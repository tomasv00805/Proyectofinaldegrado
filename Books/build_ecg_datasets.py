#!/usr/bin/env python
"""
Pipeline de preparación de datasets ECG para autoencoders no supervisados.

Procesa:
  1. PTB-XL (12 derivaciones, 10s, 500 Hz) extraído a II/V1/V5.
  2. MIMIC-IV-ECG (subset diagnostico) filtrando diagnósticos normales/anómalos.

Produce salidas consistentes en `data/`:
  - Artefactos individuales por dataset (crudo, filtrado, min-max, metadatos).
  - Dataset combinado PTB+MIMIC para entrenamiento/validación/test.
  - Muestras pequeñas (10 normales / 10 anómalos) para pruebas rápidas.
  - Ejemplos antes/después de limpieza para documentación.

Uso:
  python build_ecg_datasets.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch


# =====================================================================================
# Configuración y constantes
# =====================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PTB_ROOT = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
MIMIC_ROOT = PROJECT_ROOT / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"

PTB_OUTPUT = DATA_DIR / "ptbxl_500hz_iv1v5"
MIMIC_OUTPUT = DATA_DIR / "mimic_500hz_iv1v5"
COMBINED_OUTPUT = DATA_DIR / "combined_ptbxl_mimic_500hz_iv1v5"

TARGET_LEADS = ["II", "V1", "V5"]
SAMPLING_RATE = 500
EXPECTED_SIGNAL_DURATION = 10  # segundos
EXPECTED_SAMPLES = SAMPLING_RATE * EXPECTED_SIGNAL_DURATION
BLOCK_SIZE = 256
MIMIC_CHUNK_SIZE = 10_000
PTB_MAX_ANOM = 400
MIMIC_MAX_ANOM = 600
MIN_REPORT_WORDS = 1
RNG_SEED = 12345
ANOM_HOLDOUT_PER_SPLIT = 10

ALLOWED_MIMIC_NORMAL_REPORTS = {
    "sinus rhythm",
    "normal ecg",
    "normal ecg except for rate",
    "within normal limits",
}

# Substrings que consideramos indicativos de registros normales aunque el reporte incluya texto adicional.
ALLOWED_MIMIC_SUBSTRINGS = [
    "sinus rhythm",
    "normal ecg",
    "normal electrocardiogram",
    "within normal limits",
]

BANNED_KEYWORDS = {
    # Arritmias
    "tachy",
    "brady",
    "atrial",
    "ventric",
    "junctional",
    "ectopic",
    "flutter",
    "fibrillation",
    "av block",
    "a-v block",
    "block",
    "svt",
    "pvc",
    "pac",
    "aberrant",
    # Conducción
    "bundle",
    "axis deviation",
    "conduction defect",
    "iv conduction",
    "lbbb",
    "rbbb",
    "leftward axis",
    "rightward axis",
    "left axis",
    "right axis",
    # Estructurales/isquémicas
    "infarct",
    "ischemia",
    "injury",
    "st-t",
    "st elevation",
    "st depression",
    "t wave",
    "hypertrophy",
    "lvh",
    "rvh",
    "strain",
    "prolonged qt",
    # Ambiguos
    "borderline",
    "possible",
    "consider",
    "cannot rule out",
    "summary",
    "abnormal",
    # Calidad
    "unsuitable",
    "analysis error",
    "noise",
}


# =====================================================================================
# Utils
# =====================================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_lead_name(name: str) -> str:
    return name.replace("-", "").replace(" ", "").upper()


def lead_indices(sig_names: Sequence[str], desired: Sequence[str]) -> List[int]:
    name_map = {normalize_lead_name(n): idx for idx, n in enumerate(sig_names)}
    idxs: List[int] = []
    missing: List[str] = []
    for lead in desired:
        key = normalize_lead_name(lead)
        if key not in name_map:
            missing.append(lead)
        else:
            idxs.append(name_map[key])
    if missing:
        raise KeyError(f"Derivaciones faltantes {missing} en {sig_names}")
    return idxs


def get_sig_names(meta) -> List[str]:
    if isinstance(meta, dict):
        return meta["sig_name"]
    if hasattr(meta, "sig_name"):
        return list(meta.sig_name)
    raise TypeError(f"Metadatos wfdb inesperados: {type(meta)}")


def apply_notch_50(x: np.ndarray, fs: int = SAMPLING_RATE, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=50.0 / (fs / 2.0), Q=q)
    return filtfilt(b, a, x, axis=0)


def apply_bandpass(
    x: np.ndarray,
    fs: int = SAMPLING_RATE,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs * 0.5
    lowc, highc = low / nyquist, high / nyquist
    b, a = butter(order, [lowc, highc], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def minmax_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ptp = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    xm = (x - mins) / ptp
    return xm.astype(np.float32), mins.astype(np.float32), maxs.astype(np.float32)


def preprocess_block(
    batch_raw: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    filt_batch: List[np.ndarray] = []
    mm_batch: List[np.ndarray] = []
    for rec in batch_raw:
        rec_f = apply_notch_50(rec, fs=fs)
        rec_f = apply_bandpass(rec_f, fs=fs)
        rec_mm, _, _ = minmax_scale(rec_f)
        filt_batch.append(rec_f.astype(np.float32))
        mm_batch.append(rec_mm.astype(np.float32))
    return np.stack(filt_batch, axis=0), np.stack(mm_batch, axis=0)


def summarize_signal(x: np.ndarray, leads: Sequence[str]) -> pd.DataFrame:
    rows = []
    for idx, lead in enumerate(leads):
        col = x[:, idx]
        rows.append(
            {
                "lead": lead,
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
            }
        )
    return pd.DataFrame(rows).set_index("lead")


def save_summary_report(
    output_dir: Path,
    prefix: str,
    raw: np.ndarray,
    filt: np.ndarray,
    mm: np.ndarray,
    leads: Sequence[str],
) -> None:
    ensure_dir(output_dir)
    summary = {
        "raw_stats": summarize_signal(raw, leads).to_dict(),
        "filt_stats": summarize_signal(filt, leads).to_dict(),
        "mm_stats": summarize_signal(mm, leads).to_dict(),
        "raw_head": raw[:20].tolist(),
        "filt_head": filt[:20].tolist(),
        "mm_head": mm[:20].tolist(),
    }
    with open(output_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def normalize_report_text(text: str) -> str:
    clean = text.strip().lower()
    clean = clean.replace("’", "'")
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def contains_banned_keyword(normalized_report: str) -> bool:
    for token in BANNED_KEYWORDS:
        if token in normalized_report:
            return True
    return False


def contains_allowed_phrase(normalized_report: str) -> bool:
    if normalized_report in ALLOWED_MIMIC_NORMAL_REPORTS:
        return True
    return any(allowed in normalized_report for allowed in ALLOWED_MIMIC_SUBSTRINGS)


def open_memmap_known_shape(
    path: Path, T: int, C: int, mode: str = "r"
) -> Tuple[np.memmap, int]:
    bytes_total = path.stat().st_size
    bytes_per_record = 4 * T * C
    if bytes_total % bytes_per_record != 0:
        raise ValueError(
            f"Tamaño incompatible para {path}. bytes={bytes_total}, esperado múltiplo de {bytes_per_record}"
        )
    N = bytes_total // bytes_per_record
    mmap = np.memmap(path, dtype=np.float32, mode=mode, shape=(N, T, C))
    return mmap, N


# =====================================================================================
# PTB-XL pipeline
# =====================================================================================

def prepare_ptbxl_dataset(
    overwrite: bool = False,
    block_size: int = BLOCK_SIZE,
    max_anom: int = PTB_MAX_ANOM,
) -> Dict[str, object]:
    ensure_dir(PTB_OUTPUT)

    dims_json = PTB_OUTPUT / "dims.json"
    X_norm_mm_path = PTB_OUTPUT / "X_norm_mm.dat"

    if X_norm_mm_path.exists() and not overwrite:
        print("[PTB-XL] Artefactos existentes, omitiendo extracción (usar --overwrite-ptb para regenerar).")
        with open(dims_json, "r", encoding="utf-8") as fp:
            dims = json.load(fp)
        return {
            "dataset": "ptbxl",
            "status": "skipped",
            "dims": dims,
            "output_root": str(PTB_OUTPUT),
        }

    print("[PTB-XL] Iniciando preparación…")

    db_csv = PTB_ROOT / "ptbxl_database.csv"
    scp_csv = PTB_ROOT / "scp_statements.csv"
    assert db_csv.exists(), f"No se encontró {db_csv}"
    assert scp_csv.exists(), f"No se encontró {scp_csv}"

    Y = pd.read_csv(db_csv, index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(lambda s: ast_literal_eval(s))

    agg_df = pd.read_csv(scp_csv, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]

    def aggregate_superclass(code_dict: Dict[str, float]) -> List[str]:
        classes: List[str] = []
        for k in code_dict.keys():
            if k in agg_df.index:
                classes.append(agg_df.loc[k, "diagnostic_class"])
        return sorted(set(classes))

    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_superclass)
    Y["is_norm"] = Y["diagnostic_superclass"].apply(
        lambda classes: len(classes) == 1 and classes[0] == "NORM"
    )
    Y["is_anom"] = ~Y["is_norm"]

    norm_ids = Y[Y["is_norm"]].index.to_list()
    anom_ids = Y[Y["is_anom"]].index.to_list()[:max_anom]

    print(f"[PTB-XL] Registros normales: {len(norm_ids)} | muestra anómala: {len(anom_ids)}")
    assert norm_ids, "[PTB-XL] No hay registros normales."

    first_sig, meta0 = read_ptb_record(norm_ids[0], Y)
    T, C = first_sig.shape
    assert T == EXPECTED_SAMPLES, f"[PTB-XL] Se esperaban {EXPECTED_SAMPLES} muestras y se obtuvo {T}"
    assert C == len(TARGET_LEADS), f"[PTB-XL] Se esperaban {len(TARGET_LEADS)} derivaciones."

    norm_mm_path = PTB_OUTPUT / "X_norm_raw.dat"
    X_norm_raw_mm = np.memmap(norm_mm_path, dtype=np.float32, mode="w+", shape=(len(norm_ids), T, C))

    meta_norm_rows = []
    n_blocks = math.ceil(len(norm_ids) / block_size)
    for b in range(n_blocks):
        a = b * block_size
        z = min(a + block_size, len(norm_ids))
        batch_ids = norm_ids[a:z]
        for idx_local, ecg_id in enumerate(batch_ids):
            sig_sel, _ = read_ptb_record(ecg_id, Y)
            X_norm_raw_mm[a + idx_local] = sig_sel
            meta_norm_rows.append(
                {
                    "ecg_id": int(ecg_id),
                    "filename_hr": Y.loc[ecg_id, "filename_hr"],
                    "strat_fold": int(Y.loc[ecg_id, "strat_fold"]),
                    "sex": Y.loc[ecg_id].get("sex", np.nan),
                    "age": Y.loc[ecg_id].get("age", np.nan),
                    "diagnostic_superclass": Y.loc[ecg_id, "diagnostic_superclass"],
                }
            )
        X_norm_raw_mm.flush()
        print(f"[PTB-XL] Bloque {b+1}/{n_blocks} (normales) listo.")

    del X_norm_raw_mm

    X_anom_raw, M_anom = read_many_ptb(Y, anom_ids)
    np.save(PTB_OUTPUT / "X_anom_raw.npy", X_anom_raw)
    M_anom.to_csv(PTB_OUTPUT / "meta_anom.csv", index=True)
    np.save(PTB_OUTPUT / "y_anom.npy", np.ones(len(M_anom), dtype=np.int64))

    M_norm = pd.DataFrame(meta_norm_rows).set_index("ecg_id")
    M_norm.to_csv(PTB_OUTPUT / "meta_norm.csv", index=True)
    np.save(PTB_OUTPUT / "y_norm.npy", np.zeros(len(M_norm), dtype=np.int64))

    dims = {"T": T, "C": C}
    with open(dims_json, "w", encoding="utf-8") as fp:
        json.dump(dims, fp)

    X_norm_raw = np.memmap(norm_mm_path, dtype=np.float32, mode="r", shape=(len(M_norm), T, C))
    X_norm_filt_path = PTB_OUTPUT / "X_norm_filt.dat"
    X_norm_mm_path = PTB_OUTPUT / "X_norm_mm.dat"
    X_norm_filt = np.memmap(X_norm_filt_path, dtype=np.float32, mode="w+", shape=(len(M_norm), T, C))
    X_norm_mm = np.memmap(X_norm_mm_path, dtype=np.float32, mode="w+", shape=(len(M_norm), T, C))

    n_blocks = math.ceil(len(M_norm) / block_size)
    for b in range(n_blocks):
        a = b * block_size
        z = min(a + block_size, len(M_norm))
        Xf, Xm = preprocess_block(X_norm_raw[a:z])
        X_norm_filt[a:z] = Xf
        X_norm_mm[a:z] = Xm
        X_norm_filt.flush()
        X_norm_mm.flush()
        print(f"[PTB-XL] Bloque {b+1}/{n_blocks} (filtro + minmax) listo.")

    del X_norm_filt, X_norm_mm

    Xa_filt = Xa_mm = None
    if len(X_anom_raw):
        Xa_filt, Xa_mm = preprocess_block(X_anom_raw)
        np.save(PTB_OUTPUT / "X_anom_filt.npy", Xa_filt)
        np.save(PTB_OUTPUT / "X_anom_mm.npy", Xa_mm)

    X_norm_filt_mem, _ = open_memmap_known_shape(X_norm_filt_path, T, C, mode="r")
    X_norm_mm_mem, _ = open_memmap_known_shape(X_norm_mm_path, T, C, mode="r")
    save_summary_report(
        PTB_OUTPUT,
        "ptbxl_norm_example",
        np.array(X_norm_raw[0]),
        np.array(X_norm_filt_mem[0]),
        np.array(X_norm_mm_mem[0]),
        TARGET_LEADS,
    )
    del X_norm_filt_mem, X_norm_mm_mem, X_norm_raw

    build_ptb_splits(M_norm, M_anom, PTB_OUTPUT)

    print("[PTB-XL] Preparación finalizada.")
    return {
        "dataset": "ptbxl",
        "status": "generated",
        "normals": len(M_norm),
        "anomalies": int(len(M_anom)),
        "T": T,
        "C": C,
        "output_root": str(PTB_OUTPUT),
    }


def ast_literal_eval(text: str):
    import ast

    return ast.literal_eval(text)


def read_ptb_record(ecg_id: int, metadata: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    rel = metadata.loc[ecg_id, "filename_hr"]
    rec_path = str(PTB_ROOT / rel)
    sig, meta = wfdb.rdsamp(rec_path)
    sig_names = get_sig_names(meta)
    idxs = lead_indices(sig_names, TARGET_LEADS)
    selected = sig[:, idxs].astype(np.float32)
    assert selected.shape[0] == EXPECTED_SAMPLES, f"[PTB-XL] Registro {ecg_id} longitud inesperada {selected.shape}"
    return selected, meta


def read_many_ptb(metadata: pd.DataFrame, ids: Sequence[int]) -> Tuple[np.ndarray, pd.DataFrame]:
    xs: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    for ecg_id in ids:
        sig_sel, _ = read_ptb_record(ecg_id, metadata)
        xs.append(sig_sel)
        rows.append(
            {
                "ecg_id": int(ecg_id),
                "filename_hr": metadata.loc[ecg_id, "filename_hr"],
                "strat_fold": int(metadata.loc[ecg_id, "strat_fold"]),
                "sex": metadata.loc[ecg_id].get("sex", np.nan),
                "age": metadata.loc[ecg_id].get("age", np.nan),
                "diagnostic_superclass": metadata.loc[ecg_id, "diagnostic_superclass"],
            }
        )
    X = np.stack(xs, axis=0) if xs else np.zeros((0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32)
    M = pd.DataFrame(rows).set_index("ecg_id") if rows else pd.DataFrame()
    return X, M


def build_ptb_splits(M_norm: pd.DataFrame, M_anom: pd.DataFrame, output_root: Path) -> None:
    DF_norm = M_norm.copy()
    DF_norm["y"] = 0
    DF_norm["source"] = "norm"

    DF_anom = M_anom.copy()
    if not DF_anom.empty:
        DF_anom["y"] = 1
        DF_anom["source"] = "anom"

    DF = pd.concat([DF_norm, DF_anom], axis=0)
    DF = DF.reset_index().rename(columns={"index": "ecg_id"})
    DF["row_in_source"] = DF.groupby("source").cumcount()

    def strat_fold_split(sf: int) -> str:
        if sf == 10:
            return "test"
        if sf == 9:
            return "val"
        return "train"

    DF["split"] = DF["strat_fold"].apply(strat_fold_split)

    split_dir = output_root / "splits"
    ensure_dir(split_dir)

    summary: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        part = DF[DF["split"] == split]
        idx_norm = part[part["source"] == "norm"]["row_in_source"].astype(int).to_numpy()
        idx_anom = part[part["source"] == "anom"]["row_in_source"].astype(int).to_numpy()
        y_split = part["y"].astype(int).to_numpy()
        part[["ecg_id", "source", "row_in_source", "y", "strat_fold", "diagnostic_superclass"]].to_csv(
            split_dir / f"meta_{split}.csv", index=False
        )
        np.save(split_dir / f"idx_norm_{split}.npy", idx_norm)
        np.save(split_dir / f"idx_anom_{split}.npy", idx_anom)
        np.save(split_dir / f"y_{split}.npy", y_split)
        summary[split] = {
            "n_samples": int(len(part)),
            "n_norm": int(len(idx_norm)),
            "n_anom": int(len(idx_anom)),
        }

    with open(split_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


# =====================================================================================
# MIMIC-IV-ECG pipeline
# =====================================================================================

@dataclass
class MIMICRecord:
    subject_id: int
    study_id: int
    cart_id: int
    ecg_time: str
    path: str
    reports_raw: List[str]
    filtering: str
    bandwidth: str
    rr_interval: Optional[float]
    p_onset: Optional[float]
    p_end: Optional[float]
    qrs_onset: Optional[float]
    qrs_end: Optional[float]
    t_end: Optional[float]
    p_axis: Optional[float]
    qrs_axis: Optional[float]
    t_axis: Optional[float]

    def normalized_reports(self) -> List[str]:
        return [normalize_report_text(rep) for rep in self.reports_raw if rep and rep.strip()]


def select_mimic_records(max_anom: int = MIMIC_MAX_ANOM) -> Tuple[List[MIMICRecord], List[MIMICRecord]]:
    machine_csv = MIMIC_ROOT / "machine_measurements.csv"
    record_list_csv = MIMIC_ROOT / "record_list.csv"
    assert machine_csv.exists(), f"No se encontró {machine_csv}"
    assert record_list_csv.exists(), f"No se encontró {record_list_csv}"

    record_df = pd.read_csv(record_list_csv)
    path_map: Dict[Tuple[int, int], str] = {
        (int(row.subject_id), int(row.study_id)): row.path for row in record_df.itertuples(index=False)
    }

    report_cols = [col for col in pd.read_csv(machine_csv, nrows=0).columns if col.startswith("report_")]

    normals: List[MIMICRecord] = []
    anomalies: List[MIMICRecord] = []
    skipped_missing_path = 0
    skipped_empty_reports = 0
    skipped_no_allowed = 0
    skipped_banned = 0

    for chunk in pd.read_csv(machine_csv, chunksize=5000):
        for row in chunk.itertuples(index=False):
            key = (int(row.subject_id), int(row.study_id))
            if key not in path_map:
                skipped_missing_path += 1
                continue

            reports = []
            for col in report_cols:
                val = getattr(row, col)
                if isinstance(val, str) and val.strip():
                    reports.append(val.strip())
            if not reports:
                skipped_empty_reports += 1
                continue

            record = MIMICRecord(
                subject_id=int(row.subject_id),
                study_id=int(row.study_id),
                cart_id=int(getattr(row, "cart_id")),
                ecg_time=str(getattr(row, "ecg_time")),
                path=path_map[key],
                reports_raw=reports,
                filtering=str(getattr(row, "filtering", "")),
                bandwidth=str(getattr(row, "bandwidth", "")),
                rr_interval=float_or_none(getattr(row, "rr_interval", np.nan)),
                p_onset=float_or_none(getattr(row, "p_onset", np.nan)),
                p_end=float_or_none(getattr(row, "p_end", np.nan)),
                qrs_onset=float_or_none(getattr(row, "qrs_onset", np.nan)),
                qrs_end=float_or_none(getattr(row, "qrs_end", np.nan)),
                t_end=float_or_none(getattr(row, "t_end", np.nan)),
                p_axis=float_or_none(getattr(row, "p_axis", np.nan)),
                qrs_axis=float_or_none(getattr(row, "qrs_axis", np.nan)),
                t_axis=float_or_none(getattr(row, "t_axis", np.nan)),
            )

            normalized_reports = record.normalized_reports()

            has_banned = any(contains_banned_keyword(rep) for rep in normalized_reports)
            has_allowed = any(contains_allowed_phrase(rep) for rep in normalized_reports)

            if has_banned:
                skipped_banned += 1
                if len(anomalies) < max_anom:
                    anomalies.append(record)
                continue

            if normalized_reports and has_allowed:
                normals.append(record)
            else:
                skipped_no_allowed += 1
                if len(anomalies) < max_anom:
                    anomalies.append(record)

    print(f"[MIMIC] Seleccionados {len(normals)} normales | {len(anomalies)} anómalos (limit {max_anom}).")
    print(
        "[MIMIC] Saltados → path faltante:",
        skipped_missing_path,
        "| sin reportes:",
        skipped_empty_reports,
        "| con palabras prohibidas:",
        skipped_banned,
        "| sin frases normales:",
        skipped_no_allowed,
    )
    return normals, anomalies


def float_or_none(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def read_mimic_signal(record: MIMICRecord) -> Optional[np.ndarray]:
    relative_path = Path(record.path)
    candidates = [
        MIMIC_ROOT / relative_path,
        PROJECT_ROOT / relative_path,
        relative_path,
    ]

    last_exc: Optional[Exception] = None
    for candidate in candidates:
        try:
            sig, meta = wfdb.rdsamp(str(candidate))
        except Exception as exc:
            last_exc = exc
            continue

        sig_names = get_sig_names(meta)
        if len(sig_names) < 12:
            return None
        if sig.shape[0] != EXPECTED_SAMPLES:
            return None
        fs = getattr(meta, "fs", None) or meta.get("fs") if isinstance(meta, dict) else None
        if fs and int(fs) != SAMPLING_RATE:
            return None

        try:
            idxs = lead_indices(sig_names, TARGET_LEADS)
        except KeyError:
            return None
        return sig[:, idxs].astype(np.float32)

    if last_exc is not None:
        sys.stderr.write(f"[MIMIC] Error leyendo {record.path}: {last_exc}\n")
    return None


def _count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as fh:
        next(fh, None)  # saltar header
        return sum(1 for _ in fh)


def write_mimic_arrays(
    normals: Sequence[MIMICRecord],
    anomalies: Sequence[MIMICRecord],
    block_size: int = MIMIC_CHUNK_SIZE,
) -> Dict[str, object]:
    ensure_dir(MIMIC_OUTPUT)

    meta_norm_path = MIMIC_OUTPUT / "meta_norm.csv"
    norm_raw_path = MIMIC_OUTPUT / "X_norm_raw.dat"
    norm_filt_path = MIMIC_OUTPUT / "X_norm_filt.dat"
    norm_mm_path = MIMIC_OUTPUT / "X_norm_mm.dat"
    progress_path = MIMIC_OUTPUT / "progress_norm.json"

    total_normals_input = len(normals)
    existing_normals = _count_csv_rows(meta_norm_path)
    if existing_normals:
        print(f"[MIMIC] Reanudando desde {existing_normals} registros normales ya procesados.")

    normals_remaining = normals[existing_normals:]
    total_chunks = math.ceil(len(normals_remaining) / block_size) if normals_remaining else 0
    if not normals_remaining:
        print("[MIMIC] No hay registros nuevos para procesar (ya están todos en disco).")
    processed_normals = existing_normals
    chunk_index = 0
    skipped_total = 0
    skipped_examples: List[str] = []

    header_needed = not meta_norm_path.exists()

    while normals_remaining:
        chunk_records = normals_remaining[:block_size]
        normals_remaining = normals_remaining[block_size:]
        chunk_index += 1
        end_estimate = min(processed_normals + len(chunk_records), total_normals_input)
        print(
            f"[MIMIC] Bloque {chunk_index}/{total_chunks} "
            f"(índices ~{processed_normals}-{end_estimate})"
        )

        raw_list: List[np.ndarray] = []
        filt_list: List[np.ndarray] = []
        mm_list: List[np.ndarray] = []
        meta_list: List[Dict[str, object]] = []
        records_seen = 0

        for record in chunk_records:
            sig = read_mimic_signal(record)
            if sig is None:
                skipped_total += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append(record.path)
                continue
            raw_list.append(sig)
            meta_list.append(record_to_meta(record))

            xf = apply_notch_50(sig)
            xf = apply_bandpass(xf)
            xm, _, _ = minmax_scale(xf)
            filt_list.append(xf.astype(np.float32))
            mm_list.append(xm.astype(np.float32))
            records_seen += 1
            if records_seen % 1000 == 0:
                print(
                    f"    [MIMIC]   → registros leídos en bloque {chunk_index}: {records_seen}/{len(chunk_records)}"
                )

        if not raw_list:
            print(f"[MIMIC] Bloque {chunk_index}: todos los registros del bloque fallaron, continuando.")
            continue

        raw_chunk = np.stack(raw_list, axis=0).astype(np.float32)
        filt_chunk = np.stack(filt_list, axis=0).astype(np.float32)
        mm_chunk = np.stack(mm_list, axis=0).astype(np.float32)
        meta_chunk = pd.DataFrame(meta_list)

        chunk_size = raw_chunk.shape[0]
        start_idx = processed_normals
        end_idx = processed_normals + chunk_size

        # Ajustar tamaño del archivo y escribir bloque
        def _append_chunk(path: Path, data: np.ndarray, dtype=np.float32):
            if not path.exists():
                mm = np.memmap(path, dtype=dtype, mode="w+", shape=data.shape)
                mm[:] = data
                mm.flush()
                del mm
            else:
                bytes_to_add = data.nbytes
                with path.open("ab") as fh:
                    fh.write(b"\x00" * bytes_to_add)
                new_shape = (end_idx, EXPECTED_SAMPLES, len(TARGET_LEADS))
                mm = np.memmap(path, dtype=dtype, mode="r+", shape=new_shape)
                mm[start_idx:end_idx] = data
                mm.flush()
                del mm

        _append_chunk(norm_raw_path, raw_chunk)
        _append_chunk(norm_filt_path, filt_chunk)
        _append_chunk(norm_mm_path, mm_chunk)

        meta_chunk["index_global"] = np.arange(start_idx, end_idx)
        meta_chunk.to_csv(
            meta_norm_path,
            mode="a",
            header=header_needed,
            index=False,
        )
        header_needed = False

        processed_normals = end_idx
        if progress_path:
            with progress_path.open("w", encoding="utf-8") as fh:
                json.dump({"normals_processed": processed_normals}, fh)

        print(
            f"[MIMIC] Bloque {chunk_index} listo "
            f"({processed_normals}/{total_normals_input} acumulados)."
        )

    if processed_normals == existing_normals:
        if processed_normals == 0:
            example_msg = f" Ejemplos fallidos: {skipped_examples}" if skipped_examples else ""
            raise RuntimeError(
                "[MIMIC] No se pudieron leer registros normales válidos "
                f"(total candidatos={len(normals)}, fallidos={skipped_total}).{example_msg}"
            )
        else:
            print("[MIMIC] No se incorporaron registros nuevos; se conserva el estado existente.")

    if skipped_total:
        print(f"[MIMIC] Advertencia: {skipped_total} registros normales descartados al leer señales.")
        if skipped_examples:
            print("[MIMIC] Ejemplos descartados:", skipped_examples)

    np.save(MIMIC_OUTPUT / "y_norm.npy", np.zeros(processed_normals, dtype=np.int64))
    with open(MIMIC_OUTPUT / "dims.json", "w", encoding="utf-8") as fp:
        json.dump({"T": EXPECTED_SAMPLES, "C": len(TARGET_LEADS)}, fp)

    # Anomalías
    anomaly_signals = []
    meta_anom_rows = []
    for record in anomalies:
        sig = read_mimic_signal(record)
        if sig is None:
            continue
        anomaly_signals.append(sig)
        meta_anom_rows.append(record_to_meta(record))

    X_anom_raw = np.stack(anomaly_signals, axis=0) if anomaly_signals else np.zeros(
        (0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32
    )
    np.save(MIMIC_OUTPUT / "X_anom_raw.npy", X_anom_raw)
    if meta_anom_rows:
        pd.DataFrame(meta_anom_rows).to_csv(MIMIC_OUTPUT / "meta_anom.csv", index=False)
    np.save(MIMIC_OUTPUT / "y_anom.npy", np.ones(len(meta_anom_rows), dtype=np.int64))

    if len(X_anom_raw):
        Xa_filt, Xa_mm = preprocess_block(X_anom_raw)
        np.save(MIMIC_OUTPUT / "X_anom_filt.npy", Xa_filt)
        np.save(MIMIC_OUTPUT / "X_anom_mm.npy", Xa_mm)

    X_norm_raw_mm = np.memmap(norm_raw_path, dtype=np.float32, mode="r", shape=(processed_normals, EXPECTED_SAMPLES, len(TARGET_LEADS)))
    X_norm_filt_mm = np.memmap(norm_filt_path, dtype=np.float32, mode="r", shape=(processed_normals, EXPECTED_SAMPLES, len(TARGET_LEADS)))
    X_norm_mm_mm = np.memmap(norm_mm_path, dtype=np.float32, mode="r", shape=(processed_normals, EXPECTED_SAMPLES, len(TARGET_LEADS)))

    sample_idx = processed_normals - 1 if processed_normals else 0
    sample_raw = np.array(X_norm_raw_mm[sample_idx])
    sample_filt = np.array(X_norm_filt_mm[sample_idx])
    sample_mm = np.array(X_norm_mm_mm[sample_idx])

    save_summary_report(
        MIMIC_OUTPUT,
        "mimic_norm_example",
        sample_raw,
        sample_filt,
        sample_mm,
        TARGET_LEADS,
    )

    del X_norm_raw_mm, X_norm_filt_mm, X_norm_mm_mm

    print(
        f"[MIMIC] Normales almacenados: {processed_normals} "
        f"| anómalos escritos: {len(meta_anom_rows)} | descartados={skipped_total}"
    )

    status = "generated" if processed_normals > existing_normals else "skipped"
    return {
        "dataset": "mimic",
        "status": status,
        "normals": int(processed_normals),
        "anomalies": int(len(meta_anom_rows)),
        "skipped_normals": int(skipped_total),
        "output_root": str(MIMIC_OUTPUT),
    }


def record_to_meta(record: MIMICRecord) -> Dict[str, object]:
    return {
        "subject_id": record.subject_id,
        "study_id": record.study_id,
        "cart_id": record.cart_id,
        "ecg_time": record.ecg_time,
        "path": record.path,
        "reports": " | ".join(record.reports_raw),
        "filtering": record.filtering,
        "bandwidth": record.bandwidth,
        "rr_interval": record.rr_interval,
        "p_onset": record.p_onset,
        "p_end": record.p_end,
        "qrs_onset": record.qrs_onset,
        "qrs_end": record.qrs_end,
        "t_end": record.t_end,
        "p_axis": record.p_axis,
        "qrs_axis": record.qrs_axis,
        "t_axis": record.t_axis,
    }


# =====================================================================================
# Dataset combinado
# =====================================================================================

def build_combined_dataset(seed: int = RNG_SEED) -> Dict[str, object]:
    ensure_dir(COMBINED_OUTPUT)

    ptb_norm_mm = PTB_OUTPUT / "X_norm_mm.dat"
    mimic_norm_mm = MIMIC_OUTPUT / "X_norm_mm.dat"
    for path in [ptb_norm_mm, mimic_norm_mm]:
        if not path.exists():
            raise FileNotFoundError(f"Falta el archivo requerido: {path}")

    with open(PTB_OUTPUT / "dims.json", "r", encoding="utf-8") as fp:
        dims_ptb = json.load(fp)
    with open(MIMIC_OUTPUT / "dims.json", "r", encoding="utf-8") as fp:
        dims_mimic = json.load(fp)
    assert dims_ptb == dims_mimic == {"T": EXPECTED_SAMPLES, "C": len(TARGET_LEADS)}

    X_ptb, N_ptb = open_memmap_known_shape(ptb_norm_mm, EXPECTED_SAMPLES, len(TARGET_LEADS))
    X_mimic, N_mimic = open_memmap_known_shape(mimic_norm_mm, EXPECTED_SAMPLES, len(TARGET_LEADS))

    total = N_ptb + N_mimic
    combined_path = COMBINED_OUTPUT / "X_norm_mm.dat"
    X_combined = np.memmap(combined_path, dtype=np.float32, mode="w+", shape=(total, EXPECTED_SAMPLES, len(TARGET_LEADS)))

    X_combined[:N_ptb] = X_ptb
    X_combined[N_ptb:] = X_mimic
    X_combined.flush()
    del X_combined
    del X_ptb, X_mimic

    meta_ptb = pd.read_csv(PTB_OUTPUT / "meta_norm.csv")
    meta_ptb["source_dataset"] = "ptbxl"
    meta_ptb["row_index"] = np.arange(len(meta_ptb))
    meta_mimic = pd.read_csv(MIMIC_OUTPUT / "meta_norm.csv")
    meta_mimic["source_dataset"] = "mimic"
    meta_mimic["row_index"] = np.arange(len(meta_mimic))

    meta_ptb["combined_index"] = np.arange(len(meta_ptb))
    meta_mimic["combined_index"] = np.arange(len(meta_mimic)) + len(meta_ptb)

    meta_combined = pd.concat([meta_ptb, meta_mimic], ignore_index=True)
    meta_combined.to_csv(COMBINED_OUTPUT / "meta_norm.csv", index=False)

    np.save(COMBINED_OUTPUT / "dims.npy", np.array([EXPECTED_SAMPLES, len(TARGET_LEADS)], dtype=np.int64))

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    train_cut = int(0.8 * total)
    val_cut = int(0.9 * total)

    idx_train = np.array(sorted(indices[:train_cut]), dtype=np.int64)
    idx_val = np.array(sorted(indices[train_cut:val_cut]), dtype=np.int64)
    idx_test = np.array(sorted(indices[val_cut:]), dtype=np.int64)

    splits_dir = COMBINED_OUTPUT / "splits"
    ensure_dir(splits_dir)

    np.save(splits_dir / "idx_norm_train.npy", idx_train)
    np.save(splits_dir / "idx_norm_val.npy", idx_val)
    np.save(splits_dir / "idx_norm_test.npy", idx_test)

    # Anomalías combinadas
    Xa_ptb = load_optional_np(PTB_OUTPUT / "X_anom_mm.npy")
    Xa_mimic = load_optional_np(MIMIC_OUTPUT / "X_anom_mm.npy")
    if Xa_ptb.size == 0 and Xa_mimic.size == 0:
        Xa_combined = np.zeros((0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32)
    else:
        Xa_combined = np.concatenate([arr for arr in [Xa_ptb, Xa_mimic] if arr.size], axis=0)
    np.save(COMBINED_OUTPUT / "X_anom_mm.npy", Xa_combined)

    meta_anom = []
    if (PTB_OUTPUT / "meta_anom.csv").exists():
        tmp = pd.read_csv(PTB_OUTPUT / "meta_anom.csv")
        tmp["source_dataset"] = "ptbxl"
        meta_anom.append(tmp)
    if (MIMIC_OUTPUT / "meta_anom.csv").exists():
        tmp = pd.read_csv(MIMIC_OUTPUT / "meta_anom.csv")
        tmp["source_dataset"] = "mimic"
        meta_anom.append(tmp)
    meta_anom_df = pd.concat(meta_anom, ignore_index=True) if meta_anom else pd.DataFrame()
    if not meta_anom_df.empty:
        meta_anom_df["combined_index"] = np.arange(len(meta_anom_df))
        meta_anom_df.to_csv(COMBINED_OUTPUT / "meta_anom.csv", index=False)

    idx_anom_train = np.zeros((0,), dtype=np.int64)
    np.save(splits_dir / "idx_anom_train.npy", idx_anom_train)

    rng_np = np.random.default_rng(seed)
    anom_total = Xa_combined.shape[0]
    if anom_total:
        perm = rng_np.permutation(anom_total)
        val_keep = min(ANOM_HOLDOUT_PER_SPLIT, anom_total)
        test_keep = min(ANOM_HOLDOUT_PER_SPLIT, anom_total - val_keep)
        idx_anom_val = np.sort(perm[:val_keep]) if val_keep else np.zeros((0,), dtype=np.int64)
        idx_anom_test = (
            np.sort(perm[val_keep : val_keep + test_keep]) if test_keep else np.zeros((0,), dtype=np.int64)
        )
    else:
        idx_anom_val = np.zeros((0,), dtype=np.int64)
        idx_anom_test = np.zeros((0,), dtype=np.int64)

    np.save(splits_dir / "idx_anom_val.npy", idx_anom_val)
    np.save(splits_dir / "idx_anom_test.npy", idx_anom_test)

    if not meta_anom_df.empty:
        split_anom_dir = splits_dir / "anom_meta"
        ensure_dir(split_anom_dir)
        if len(idx_anom_val):
            meta_anom_df.iloc[idx_anom_val].to_csv(split_anom_dir / "meta_val.csv", index=False)
        if len(idx_anom_test):
            meta_anom_df.iloc[idx_anom_test].to_csv(split_anom_dir / "meta_test.csv", index=False)

    meta_combined.iloc[idx_train].to_csv(splits_dir / "meta_train.csv", index=False)
    meta_combined.iloc[idx_val].to_csv(splits_dir / "meta_val.csv", index=False)
    meta_combined.iloc[idx_test].to_csv(splits_dir / "meta_test.csv", index=False)

    save_samples_for_debug(COMBINED_OUTPUT, COMBINED_OUTPUT / "X_norm_mm.dat", Xa_combined)

    holdout_dir = COMBINED_OUTPUT / "holdout"
    ensure_dir(holdout_dir)
    if len(idx_test):
        test_sample = np.sort(
            np.random.default_rng(seed + 1).choice(idx_test, size=min(ANOM_HOLDOUT_PER_SPLIT, len(idx_test)), replace=False)
        )
        np.save(holdout_dir / "idx_norm_test_10.npy", test_sample)
    else:
        np.save(holdout_dir / "idx_norm_test_10.npy", np.zeros((0,), dtype=np.int64))

    if len(idx_anom_test):
        anom_sample = np.sort(
            np.random.default_rng(seed + 2).choice(
                idx_anom_test, size=min(ANOM_HOLDOUT_PER_SPLIT, len(idx_anom_test)), replace=False
            )
        )
        np.save(holdout_dir / "idx_anom_test_10.npy", anom_sample)
    else:
        np.save(holdout_dir / "idx_anom_test_10.npy", np.zeros((0,), dtype=np.int64))

    summary = {
        "dataset": "combined",
        "status": "generated",
        "normals_total": total,
        "normals_ptb": int(N_ptb),
        "normals_mimic": int(N_mimic),
        "split_train": int(len(idx_train)),
        "split_val": int(len(idx_val)),
        "split_test": int(len(idx_test)),
        "anomalies_total": int(Xa_combined.shape[0]),
        "anom_val": int(len(idx_anom_val)),
        "anom_test": int(len(idx_anom_test)),
    }

    with open(COMBINED_OUTPUT / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[COMBINED] Normales totales {total} | anómalos {Xa_combined.shape[0]}")
    return summary


def load_optional_np(path: Path) -> np.ndarray:
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr)
    return np.zeros((0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32)


def save_samples_for_debug(output_root: Path, norm_path: Path, Xa_combined: np.ndarray) -> None:
    ensure_dir(output_root / "samples")
    X_norm, N_norm = open_memmap_known_shape(norm_path, EXPECTED_SAMPLES, len(TARGET_LEADS))
    rng = np.random.default_rng(RNG_SEED)
    idx_norm = rng.choice(N_norm, size=min(10, N_norm), replace=False)
    sample_norm = np.asarray(X_norm[idx_norm])
    np.save(output_root / "samples" / "norm_10.npy", sample_norm)
    del X_norm

    if Xa_combined.size:
        idx_anom = rng.choice(Xa_combined.shape[0], size=min(10, Xa_combined.shape[0]), replace=False)
        np.save(output_root / "samples" / "anom_10.npy", Xa_combined[idx_anom])


# =====================================================================================
# CLI
# =====================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye datasets ECG PTB-XL + MIMIC.")
    parser.add_argument("--overwrite-ptb", action="store_true", help="Regenerar artefactos PTB-XL.")
    parser.add_argument("--overwrite-mimic", action="store_true", help="Regenerar artefactos MIMIC.")
    parser.add_argument("--mimic-max-anom", type=int, default=MIMIC_MAX_ANOM, help="Máximo de anómalos MIMIC a guardar.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Semilla para splits aleatorios.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    reports = []

    reports.append(prepare_ptbxl_dataset(overwrite=args.overwrite_ptb))

    if args.overwrite_mimic or not (MIMIC_OUTPUT / "X_norm_mm.dat").exists():
        normals, anomalies = select_mimic_records(max_anom=args.mimic_max_anom)
        reports.append(write_mimic_arrays(normals, anomalies))
    else:
        reports.append(
            {
                "dataset": "mimic",
                "status": "skipped",
                "output_root": str(MIMIC_OUTPUT),
            }
        )

    reports.append(build_combined_dataset(seed=args.seed))

    with open(DATA_DIR / "dataset_build_report.json", "w", encoding="utf-8") as fp:
        json.dump(reports, fp, indent=2)

    print("\nResumen de ejecución:")
    for rep in reports:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

