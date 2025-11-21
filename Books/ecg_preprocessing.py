#!/usr/bin/env python
"""
Módulo unificado de preprocesamiento de señales ECG.

Este módulo integra toda la lógica de extracción, filtrado, limpieza y normalización
de ECG de los datasets PTB-XL y MIMIC-IV-ECG, proporcionando una interfaz clara
y reutilizable para el pipeline completo.

Funciones principales:
- load_raw_data: Carga datos crudos desde PTB-XL y MIMIC-IV-ECG
- filter_and_clean_signals: Aplica filtros (notch 50Hz, bandpass) y limpieza
- normalize_signals: Normaliza señales usando min-max scaling
- build_dataset: Construye el dataset final listo para entrenamiento
- train_valid_test_split: Genera splits asegurando que train solo contiene normales
"""
from __future__ import annotations

import ast
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
RNG_SEED = 42  # Seed fijo para reproducibilidad
ANOM_HOLDOUT_PER_SPLIT = 10

ALLOWED_MIMIC_NORMAL_REPORTS = {
    "sinus rhythm",
    "normal ecg",
    "normal ecg except for rate",
    "within normal limits",
}

ALLOWED_MIMIC_SUBSTRINGS = [
    "sinus rhythm",
    "normal ecg",
    "normal electrocardiogram",
    "within normal limits",
]

BANNED_KEYWORDS = {
    "tachy", "brady", "atrial", "ventric", "junctional", "ectopic",
    "flutter", "fibrillation", "av block", "a-v block", "block", "svt",
    "pvc", "pac", "aberrant", "bundle", "axis deviation", "conduction defect",
    "iv conduction", "lbbb", "rbbb", "leftward axis", "rightward axis",
    "left axis", "right axis", "infarct", "ischemia", "injury", "st-t",
    "st elevation", "st depression", "t wave", "hypertrophy", "lvh", "rvh",
    "strain", "prolonged qt", "borderline", "possible", "consider",
    "cannot rule out", "summary", "abnormal", "unsuitable", "analysis error", "noise",
}


# =====================================================================================
# Utilidades generales
# =====================================================================================

def ensure_dir(path: Path) -> None:
    """Crea un directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_lead_name(name: str) -> str:
    """Normaliza el nombre de una derivación para comparación."""
    return name.replace("-", "").replace(" ", "").upper()


def lead_indices(sig_names: Sequence[str], desired: Sequence[str]) -> List[int]:
    """Encuentra los índices de las derivaciones deseadas en la lista de nombres."""
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
    """Extrae los nombres de las señales desde metadatos wfdb."""
    if isinstance(meta, dict):
        return meta["sig_name"]
    if hasattr(meta, "sig_name"):
        return list(meta.sig_name)
    raise TypeError(f"Metadatos wfdb inesperados: {type(meta)}")


def is_normal_ptb(diagnostic_superclass: List[str]) -> bool:
    """
    Determina si un ECG de PTB-XL es normal.
    
    Criterio estricto: solo ['NORM'] sin otras clases diagnósticas.
    
    Args:
        diagnostic_superclass: Lista de clases diagnósticas superiores
        
    Returns:
        True si es normal, False en caso contrario
    """
    return len(diagnostic_superclass) == 1 and diagnostic_superclass[0] == "NORM"


def normalize_report_text(text: str) -> str:
    """Normaliza texto de reporte para comparación."""
    clean = text.strip().lower()
    clean = clean.replace("'", "'")
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def contains_banned_keyword(normalized_report: str) -> bool:
    """Verifica si un reporte contiene palabras prohibidas (anómalo)."""
    return any(token in normalized_report for token in BANNED_KEYWORDS)


def contains_allowed_phrase(normalized_report: str) -> bool:
    """Verifica si un reporte contiene frases permitidas (normal)."""
    if normalized_report in ALLOWED_MIMIC_NORMAL_REPORTS:
        return True
    return any(allowed in normalized_report for allowed in ALLOWED_MIMIC_SUBSTRINGS)


# =====================================================================================
# Filtrado y limpieza de señales
# =====================================================================================

def apply_notch_50(x: np.ndarray, fs: int = SAMPLING_RATE, q: float = 30.0) -> np.ndarray:
    """
    Aplica filtro notch a 50 Hz para eliminar ruido de línea.
    
    Args:
        x: Señal de entrada [T, C] o [T]
        fs: Frecuencia de muestreo (Hz)
        q: Factor de calidad del filtro
        
    Returns:
        Señal filtrada con la misma forma que x
    """
    b, a = iirnotch(w0=50.0 / (fs / 2.0), Q=q)
    return filtfilt(b, a, x, axis=0)


def apply_bandpass(
    x: np.ndarray,
    fs: int = SAMPLING_RATE,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Aplica filtro pasa-banda para eliminar componentes fuera del rango de interés.
    
    Args:
        x: Señal de entrada [T, C] o [T]
        fs: Frecuencia de muestreo (Hz)
        low: Frecuencia de corte inferior (Hz)
        high: Frecuencia de corte superior (Hz)
        order: Orden del filtro Butterworth
        
    Returns:
        Señal filtrada con la misma forma que x
    """
    nyquist = fs * 0.5
    lowc, highc = low / nyquist, high / nyquist
    b, a = butter(order, [lowc, highc], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def filter_and_clean_signals(
    signals: np.ndarray,
    fs: int = SAMPLING_RATE,
    apply_notch: bool = True,
    apply_bandpass: bool = True,
) -> np.ndarray:
    """
    Aplica filtrado y limpieza a un batch de señales ECG.
    
    Args:
        signals: Array de señales [N, T, C] o [T, C]
        fs: Frecuencia de muestreo (Hz)
        apply_notch: Si aplicar filtro notch a 50Hz
        apply_bandpass: Si aplicar filtro pasa-banda
        
    Returns:
        Señales filtradas con la misma forma que signals
    """
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    filtered = signals.copy().astype(np.float32)
    
    for i in range(len(filtered)):
        if apply_notch:
            filtered[i] = apply_notch_50(filtered[i], fs=fs)
        if apply_bandpass:
            filtered[i] = apply_bandpass(filtered[i], fs=fs, low=0.5, high=40.0)
    
    if squeeze_output:
        filtered = filtered[0]
    
    return filtered


# =====================================================================================
# Normalización de señales
# =====================================================================================

def minmax_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Escala una señal a [0, 1] usando min-max por derivación.
    
    Args:
        x: Señal [T, C]
        
    Returns:
        Tupla (señal escalada, mínimos, máximos)
    """
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ptp = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    xm = (x - mins) / ptp
    return xm.astype(np.float32), mins.astype(np.float32), maxs.astype(np.float32)


def normalize_signals(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza un batch de señales usando min-max por registro y derivación.
    
    Args:
        signals: Array de señales [N, T, C] o [T, C]
        
    Returns:
        Tupla (señales normalizadas, mínimos, máximos)
    """
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]
        squeeze_output = True
    else:
        squeeze_output = False
    
    normalized = []
    mins_list = []
    maxs_list = []
    
    for sig in signals:
        nm, mins, maxs = minmax_scale(sig)
        normalized.append(nm)
        mins_list.append(mins)
        maxs_list.append(maxs)
    
    normalized = np.stack(normalized, axis=0)
    mins = np.stack(mins_list, axis=0)
    maxs = np.stack(maxs_list, axis=0)
    
    if squeeze_output:
        normalized = normalized[0]
        mins = mins[0]
        maxs = maxs[0]
    
    return normalized, mins, maxs


def preprocess_block(
    batch_raw: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesa un bloque de señales: filtrado + normalización.
    
    Args:
        batch_raw: Batch de señales crudas [N, T, C]
        fs: Frecuencia de muestreo
        
    Returns:
        Tupla (señales filtradas, señales normalizadas)
    """
    filt_batch: List[np.ndarray] = []
    mm_batch: List[np.ndarray] = []
    for rec in batch_raw:
        rec_f = apply_notch_50(rec, fs=fs)
        rec_f = apply_bandpass(rec_f, fs=fs)
        rec_mm, _, _ = minmax_scale(rec_f)
        filt_batch.append(rec_f.astype(np.float32))
        mm_batch.append(rec_mm.astype(np.float32))
    return np.stack(filt_batch, axis=0), np.stack(mm_batch, axis=0)


# =====================================================================================
# Carga de datos crudos
# =====================================================================================

def load_raw_data(
    dataset: str = "combined",
    overwrite: bool = False,
    max_anom_ptb: int = PTB_MAX_ANOM,
    max_anom_mimic: int = MIMIC_MAX_ANOM,
) -> Dict[str, object]:
    """
    Carga datos crudos desde PTB-XL y/o MIMIC-IV-ECG.
    
    Args:
        dataset: "ptbxl", "mimic", o "combined"
        overwrite: Si regenerar archivos existentes
        max_anom_ptb: Máximo de anómalos a extraer de PTB-XL
        max_anom_mimic: Máximo de anómalos a extraer de MIMIC
        
    Returns:
        Diccionario con información del dataset cargado
    """
    if dataset == "ptbxl":
        return _load_ptbxl_data(overwrite, max_anom_ptb)
    elif dataset == "mimic":
        return _load_mimic_data(overwrite, max_anom_mimic)
    elif dataset == "combined":
        ptb_report = _load_ptbxl_data(overwrite, max_anom_ptb)
        mimic_report = _load_mimic_data(overwrite, max_anom_mimic)
        return _build_combined_dataset()
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")


def _load_ptbxl_data(overwrite: bool, max_anom: int) -> Dict[str, object]:
    """Carga datos de PTB-XL."""
    ensure_dir(PTB_OUTPUT)
    
    dims_json = PTB_OUTPUT / "dims.json"
    X_norm_mm_path = PTB_OUTPUT / "X_norm_mm.dat"
    
    if X_norm_mm_path.exists() and not overwrite:
        print("[PTB-XL] Artefactos existentes, omitiendo extracción.")
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
    if not db_csv.exists():
        raise FileNotFoundError(f"No se encontró {db_csv}")
    if not scp_csv.exists():
        raise FileNotFoundError(f"No se encontró {scp_csv}")
    
    Y = pd.read_csv(db_csv, index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(lambda s: ast.literal_eval(s))
    
    agg_df = pd.read_csv(scp_csv, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    
    def aggregate_superclass(code_dict: Dict[str, float]) -> List[str]:
        classes: List[str] = []
        for k in code_dict.keys():
            if k in agg_df.index:
                classes.append(agg_df.loc[k, "diagnostic_class"])
        return sorted(set(classes))
    
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_superclass)
    Y["is_norm"] = Y["diagnostic_superclass"].apply(is_normal_ptb)
    Y["is_anom"] = ~Y["is_norm"]
    
    norm_ids = Y[Y["is_norm"]].index.to_list()
    anom_ids = Y[Y["is_anom"]].index.to_list()[:max_anom]
    
    print(f"[PTB-XL] Registros normales: {len(norm_ids)} | muestra anómala: {len(anom_ids)}")
    if not norm_ids:
        raise ValueError("[PTB-XL] No hay registros normales.")
    
    first_sig, _ = _read_ptb_record(norm_ids[0], Y)
    T, C = first_sig.shape
    if T != EXPECTED_SAMPLES:
        raise ValueError(f"[PTB-XL] Se esperaban {EXPECTED_SAMPLES} muestras y se obtuvo {T}")
    if C != len(TARGET_LEADS):
        raise ValueError(f"[PTB-XL] Se esperaban {len(TARGET_LEADS)} derivaciones.")
    
    # Guardar crudos
    norm_mm_path = PTB_OUTPUT / "X_norm_raw.dat"
    X_norm_raw_mm = np.memmap(norm_mm_path, dtype=np.float32, mode="w+", shape=(len(norm_ids), T, C))
    
    meta_norm_rows = []
    n_blocks = math.ceil(len(norm_ids) / BLOCK_SIZE)
    for b in range(n_blocks):
        a = b * BLOCK_SIZE
        z = min(a + BLOCK_SIZE, len(norm_ids))
        batch_ids = norm_ids[a:z]
        for idx_local, ecg_id in enumerate(batch_ids):
            sig_sel, _ = _read_ptb_record(ecg_id, Y)
            X_norm_raw_mm[a + idx_local] = sig_sel
            meta_norm_rows.append({
                "ecg_id": int(ecg_id),
                "filename_hr": Y.loc[ecg_id, "filename_hr"],
                "strat_fold": int(Y.loc[ecg_id, "strat_fold"]),
                "sex": Y.loc[ecg_id].get("sex", np.nan),
                "age": Y.loc[ecg_id].get("age", np.nan),
                "diagnostic_superclass": str(Y.loc[ecg_id, "diagnostic_superclass"]),
            })
        X_norm_raw_mm.flush()
        print(f"[PTB-XL] Bloque {b+1}/{n_blocks} (normales) listo.")
    
    del X_norm_raw_mm
    
    X_anom_raw, M_anom = _read_many_ptb(Y, anom_ids)
    np.save(PTB_OUTPUT / "X_anom_raw.npy", X_anom_raw)
    M_anom.to_csv(PTB_OUTPUT / "meta_anom.csv", index=True)
    np.save(PTB_OUTPUT / "y_anom.npy", np.ones(len(M_anom), dtype=np.int64))
    
    M_norm = pd.DataFrame(meta_norm_rows).set_index("ecg_id")
    M_norm.to_csv(PTB_OUTPUT / "meta_norm.csv", index=True)
    np.save(PTB_OUTPUT / "y_norm.npy", np.zeros(len(M_norm), dtype=np.int64))
    
    dims = {"T": T, "C": C}
    with open(dims_json, "w", encoding="utf-8") as fp:
        json.dump(dims, fp)
    
    # Procesar: filtrado + normalización
    X_norm_raw = np.memmap(norm_mm_path, dtype=np.float32, mode="r", shape=(len(M_norm), T, C))
    X_norm_filt_path = PTB_OUTPUT / "X_norm_filt.dat"
    X_norm_mm_path = PTB_OUTPUT / "X_norm_mm.dat"
    X_norm_filt = np.memmap(X_norm_filt_path, dtype=np.float32, mode="w+", shape=(len(M_norm), T, C))
    X_norm_mm = np.memmap(X_norm_mm_path, dtype=np.float32, mode="w+", shape=(len(M_norm), T, C))
    
    n_blocks = math.ceil(len(M_norm) / BLOCK_SIZE)
    for b in range(n_blocks):
        a = b * BLOCK_SIZE
        z = min(a + BLOCK_SIZE, len(M_norm))
        Xf, Xm = preprocess_block(X_norm_raw[a:z])
        X_norm_filt[a:z] = Xf
        X_norm_mm[a:z] = Xm
        X_norm_filt.flush()
        X_norm_mm.flush()
        print(f"[PTB-XL] Bloque {b+1}/{n_blocks} (filtro + minmax) listo.")
    
    del X_norm_filt, X_norm_mm
    
    if len(X_anom_raw):
        Xa_filt, Xa_mm = preprocess_block(X_anom_raw)
        np.save(PTB_OUTPUT / "X_anom_filt.npy", Xa_filt)
        np.save(PTB_OUTPUT / "X_anom_mm.npy", Xa_mm)
    
    del X_norm_raw
    
    _build_ptb_splits(M_norm, M_anom, PTB_OUTPUT)
    
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


def _read_ptb_record(ecg_id: int, metadata: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    """Lee un registro individual de PTB-XL."""
    rel = metadata.loc[ecg_id, "filename_hr"]
    rec_path = str(PTB_ROOT / rel)
    sig, meta = wfdb.rdsamp(rec_path)
    sig_names = get_sig_names(meta)
    idxs = lead_indices(sig_names, TARGET_LEADS)
    selected = sig[:, idxs].astype(np.float32)
    if selected.shape[0] != EXPECTED_SAMPLES:
        raise ValueError(f"[PTB-XL] Registro {ecg_id} longitud inesperada {selected.shape}")
    return selected, meta


def _read_many_ptb(metadata: pd.DataFrame, ids: Sequence[int]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Lee múltiples registros de PTB-XL."""
    xs: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    for ecg_id in ids:
        sig_sel, _ = _read_ptb_record(ecg_id, metadata)
        xs.append(sig_sel)
        rows.append({
            "ecg_id": int(ecg_id),
            "filename_hr": metadata.loc[ecg_id, "filename_hr"],
            "strat_fold": int(metadata.loc[ecg_id, "strat_fold"]),
            "sex": metadata.loc[ecg_id].get("sex", np.nan),
            "age": metadata.loc[ecg_id].get("age", np.nan),
            "diagnostic_superclass": str(metadata.loc[ecg_id, "diagnostic_superclass"]),
        })
    X = np.stack(xs, axis=0) if xs else np.zeros((0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32)
    M = pd.DataFrame(rows).set_index("ecg_id") if rows else pd.DataFrame()
    return X, M


@dataclass
class MIMICRecord:
    """Representa un registro de MIMIC-IV-ECG."""
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


def _load_mimic_data(overwrite: bool, max_anom: int) -> Dict[str, object]:
    """Carga datos de MIMIC-IV-ECG."""
    ensure_dir(MIMIC_OUTPUT)
    
    if (MIMIC_OUTPUT / "X_norm_mm.dat").exists() and not overwrite:
        print("[MIMIC] Artefactos existentes, omitiendo extracción.")
        return {
            "dataset": "mimic",
            "status": "skipped",
            "output_root": str(MIMIC_OUTPUT),
        }
    
    normals, anomalies = _select_mimic_records(max_anom)
    return _write_mimic_arrays(normals, anomalies)


def _select_mimic_records(max_anom: int = MIMIC_MAX_ANOM) -> Tuple[List[MIMICRecord], List[MIMICRecord]]:
    """Selecciona registros normales y anómalos de MIMIC."""
    machine_csv = MIMIC_ROOT / "machine_measurements.csv"
    record_list_csv = MIMIC_ROOT / "record_list.csv"
    if not machine_csv.exists():
        raise FileNotFoundError(f"No se encontró {machine_csv}")
    if not record_list_csv.exists():
        raise FileNotFoundError(f"No se encontró {record_list_csv}")
    
    record_df = pd.read_csv(record_list_csv)
    path_map: Dict[Tuple[int, int], str] = {
        (int(row.subject_id), int(row.study_id)): row.path
        for row in record_df.itertuples(index=False)
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
                rr_interval=_float_or_none(getattr(row, "rr_interval", np.nan)),
                p_onset=_float_or_none(getattr(row, "p_onset", np.nan)),
                p_end=_float_or_none(getattr(row, "p_end", np.nan)),
                qrs_onset=_float_or_none(getattr(row, "qrs_onset", np.nan)),
                qrs_end=_float_or_none(getattr(row, "qrs_end", np.nan)),
                t_end=_float_or_none(getattr(row, "t_end", np.nan)),
                p_axis=_float_or_none(getattr(row, "p_axis", np.nan)),
                qrs_axis=_float_or_none(getattr(row, "qrs_axis", np.nan)),
                t_axis=_float_or_none(getattr(row, "t_axis", np.nan)),
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
        f"[MIMIC] Saltados → path faltante: {skipped_missing_path} | "
        f"sin reportes: {skipped_empty_reports} | "
        f"con palabras prohibidas: {skipped_banned} | "
        f"sin frases normales: {skipped_no_allowed}"
    )
    return normals, anomalies


def _float_or_none(value) -> Optional[float]:
    """Convierte un valor a float o None."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _read_mimic_signal(record: MIMICRecord) -> Optional[np.ndarray]:
    """Lee una señal de MIMIC."""
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


def _write_mimic_arrays(
    normals: Sequence[MIMICRecord],
    anomalies: Sequence[MIMICRecord],
    block_size: int = MIMIC_CHUNK_SIZE,
) -> Dict[str, object]:
    """Escribe arrays de MIMIC procesados."""
    ensure_dir(MIMIC_OUTPUT)
    
    meta_norm_path = MIMIC_OUTPUT / "meta_norm.csv"
    norm_raw_path = MIMIC_OUTPUT / "X_norm_raw.dat"
    norm_filt_path = MIMIC_OUTPUT / "X_norm_filt.dat"
    norm_mm_path = MIMIC_OUTPUT / "X_norm_mm.dat"
    
    total_normals_input = len(normals)
    existing_normals = _count_csv_rows(meta_norm_path)
    if existing_normals:
        print(f"[MIMIC] Reanudando desde {existing_normals} registros normales ya procesados.")
    
    normals_remaining = normals[existing_normals:]
    total_chunks = math.ceil(len(normals_remaining) / block_size) if normals_remaining else 0
    if not normals_remaining:
        print("[MIMIC] No hay registros nuevos para procesar.")
    processed_normals = existing_normals
    chunk_index = 0
    skipped_total = 0
    
    header_needed = not meta_norm_path.exists()
    
    while normals_remaining:
        chunk_records = normals_remaining[:block_size]
        normals_remaining = normals_remaining[block_size:]
        chunk_index += 1
        end_estimate = min(processed_normals + len(chunk_records), total_normals_input)
        print(f"[MIMIC] Bloque {chunk_index}/{total_chunks} (índices ~{processed_normals}-{end_estimate})")
        
        raw_list: List[np.ndarray] = []
        filt_list: List[np.ndarray] = []
        mm_list: List[np.ndarray] = []
        meta_list: List[Dict[str, object]] = []
        
        for record in chunk_records:
            sig = _read_mimic_signal(record)
            if sig is None:
                skipped_total += 1
                continue
            raw_list.append(sig)
            meta_list.append(_record_to_meta(record))
            
            xf = apply_notch_50(sig)
            xf = apply_bandpass(xf)
            xm, _, _ = minmax_scale(xf)
            filt_list.append(xf.astype(np.float32))
            mm_list.append(xm.astype(np.float32))
        
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
        meta_chunk.to_csv(meta_norm_path, mode="a", header=header_needed, index=False)
        header_needed = False
        
        processed_normals = end_idx
        print(f"[MIMIC] Bloque {chunk_index} listo ({processed_normals}/{total_normals_input} acumulados).")
    
    if processed_normals == 0:
        raise RuntimeError(f"[MIMIC] No se pudieron leer registros normales válidos (fallidos={skipped_total}).")
    
    np.save(MIMIC_OUTPUT / "y_norm.npy", np.zeros(processed_normals, dtype=np.int64))
    with open(MIMIC_OUTPUT / "dims.json", "w", encoding="utf-8") as fp:
        json.dump({"T": EXPECTED_SAMPLES, "C": len(TARGET_LEADS)}, fp)
    
    # Anomalías
    anomaly_signals = []
    meta_anom_rows = []
    for record in anomalies:
        sig = _read_mimic_signal(record)
        if sig is None:
            continue
        anomaly_signals.append(sig)
        meta_anom_rows.append(_record_to_meta(record))
    
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
    
    print(f"[MIMIC] Normales almacenados: {processed_normals} | anómalos escritos: {len(meta_anom_rows)}")
    
    return {
        "dataset": "mimic",
        "status": "generated",
        "normals": int(processed_normals),
        "anomalies": int(len(meta_anom_rows)),
        "skipped_normals": int(skipped_total),
        "output_root": str(MIMIC_OUTPUT),
    }


def _count_csv_rows(csv_path: Path) -> int:
    """Cuenta filas en un CSV (sin header)."""
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as fh:
        next(fh, None)
        return sum(1 for _ in fh)


def _record_to_meta(record: MIMICRecord) -> Dict[str, object]:
    """Convierte un MIMICRecord a diccionario de metadatos."""
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


def _build_combined_dataset() -> Dict[str, object]:
    """Construye el dataset combinado PTB+MIMIC."""
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
    del X_combined, X_ptb, X_mimic
    
    meta_ptb = pd.read_csv(PTB_OUTPUT / "meta_norm.csv")
    meta_ptb["source_dataset"] = "ptbxl"
    meta_mimic = pd.read_csv(MIMIC_OUTPUT / "meta_norm.csv")
    meta_mimic["source_dataset"] = "mimic"
    
    meta_combined = pd.concat([meta_ptb, meta_mimic], ignore_index=True)
    meta_combined.to_csv(COMBINED_OUTPUT / "meta_norm.csv", index=False)
    
    # Anomalías combinadas
    Xa_ptb = _load_optional_np(PTB_OUTPUT / "X_anom_mm.npy")
    Xa_mimic = _load_optional_np(MIMIC_OUTPUT / "X_anom_mm.npy")
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
        meta_anom_df.to_csv(COMBINED_OUTPUT / "meta_anom.csv", index=False)
    
    print(f"[COMBINED] Normales totales {total} | anómalos {Xa_combined.shape[0]}")
    
    return {
        "dataset": "combined",
        "status": "generated",
        "normals_total": total,
        "normals_ptb": int(N_ptb),
        "normals_mimic": int(N_mimic),
        "anomalies_total": int(Xa_combined.shape[0]),
        "output_root": str(COMBINED_OUTPUT),
    }


def open_memmap_known_shape(path: Path, T: int, C: int, mode: str = "r") -> Tuple[np.memmap, int]:
    """
    Abre un memmap calculando N desde el tamaño del archivo.
    
    Args:
        path: Ruta al archivo .dat
        T: Número de muestras temporales
        C: Número de canales/derivaciones
        mode: Modo de apertura ('r' para lectura, 'w+' para escritura)
        
    Returns:
        Tupla (memmap, N) donde N es el número de registros
    """
    bytes_total = path.stat().st_size
    bytes_per_record = 4 * T * C
    if bytes_total % bytes_per_record != 0:
        raise ValueError(f"Tamaño incompatible para {path}")
    N = bytes_total // bytes_per_record
    mmap = np.memmap(path, dtype=np.float32, mode=mode, shape=(N, T, C))
    return mmap, N


def _open_memmap_known_shape(path: Path, T: int, C: int, mode: str = "r") -> Tuple[np.memmap, int]:
    """Versión privada (alias para compatibilidad interna)."""
    return open_memmap_known_shape(path, T, C, mode)


def _load_optional_np(path: Path) -> np.ndarray:
    """Carga un array numpy opcional."""
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr)
    return np.zeros((0, EXPECTED_SAMPLES, len(TARGET_LEADS)), dtype=np.float32)


# =====================================================================================
# Construcción de dataset y splits
# =====================================================================================

def build_dataset(
    dataset: str = "combined",
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Construye el dataset final listo para entrenamiento.
    
    Args:
        dataset: "ptbxl", "mimic", o "combined"
        output_dir: Directorio de salida (None usa el predeterminado)
        
    Returns:
        Diccionario con información del dataset construido
    """
    if dataset == "combined":
        return _build_combined_dataset()
    elif dataset in ["ptbxl", "mimic"]:
        output = PTB_OUTPUT if dataset == "ptbxl" else MIMIC_OUTPUT
        return {
            "dataset": dataset,
            "output_root": str(output),
        }
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")


def train_valid_test_split(
    dataset: str = "combined",
    seed: int = RNG_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    ensure_train_only_normals: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Genera splits train/valid/test asegurando que train solo contiene normales.
    
    IMPORTANTE: El set de entrenamiento SOLO contiene ECG normales.
    Los sets de validación y test contienen normales y anómalos.
    
    Args:
        dataset: "ptbxl", "mimic", o "combined"
        seed: Semilla para reproducibilidad
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        ensure_train_only_normals: Si True, garantiza que train solo tiene normales
        
    Returns:
        Diccionario con índices para cada split:
        {
            'train_norm': índices de normales en train,
            'val_norm': índices de normales en val,
            'val_anom': índices de anómalos en val,
            'test_norm': índices de normales en test,
            'test_anom': índices de anómalos en test,
        }
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")
    
    rng = np.random.default_rng(seed)
    
    if dataset == "combined":
        output_dir = COMBINED_OUTPUT
    elif dataset == "ptbxl":
        output_dir = PTB_OUTPUT
    elif dataset == "mimic":
        output_dir = MIMIC_OUTPUT
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")
    
    # Cargar metadatos
    meta_norm = pd.read_csv(output_dir / "meta_norm.csv")
    n_normals = len(meta_norm)
    
    # Cargar anómalos si existen
    meta_anom_path = output_dir / "meta_anom.csv"
    if meta_anom_path.exists():
        meta_anom = pd.read_csv(meta_anom_path)
        n_anomalies = len(meta_anom)
    else:
        meta_anom = pd.DataFrame()
        n_anomalies = 0
    
    # Split de normales
    norm_indices = np.arange(n_normals)
    rng.shuffle(norm_indices)
    
    train_cut = int(train_ratio * n_normals)
    val_cut = train_cut + int(val_ratio * n_normals)
    
    idx_norm_train = np.sort(norm_indices[:train_cut])
    idx_norm_val = np.sort(norm_indices[train_cut:val_cut])
    idx_norm_test = np.sort(norm_indices[val_cut:])
    
    # Split de anómalos (solo val y test, NO train)
    if n_anomalies > 0:
        anom_indices = np.arange(n_anomalies)
        rng.shuffle(anom_indices)
        
        # Dividir anómalos entre val y test según las proporciones
        # Usamos las mismas proporciones que para normales: val_ratio y test_ratio
        val_anom_cut = int(val_ratio * n_anomalies)
        # El resto va a test (no usamos train_ratio para anómalos)
        idx_anom_val = np.sort(anom_indices[:val_anom_cut])
        idx_anom_test = np.sort(anom_indices[val_anom_cut:])
        
        # Verificar que todos los anómalos fueron asignados
        total_assigned = len(idx_anom_val) + len(idx_anom_test)
        assert total_assigned == n_anomalies, \
            f"Error: No todos los anómalos fueron asignados. Total: {n_anomalies}, Asignados: {total_assigned}"
        
        print(f"[SPLITS] Anómalos divididos: val={len(idx_anom_val)}, test={len(idx_anom_test)} (total={n_anomalies})")
    else:
        idx_anom_val = np.array([], dtype=np.int64)
        idx_anom_test = np.array([], dtype=np.int64)
        print("[SPLITS] No hay anómalos para dividir")
    
    # Verificación crítica: train solo debe tener normales
    if ensure_train_only_normals:
        assert len(idx_norm_train) > 0, "Train debe contener al menos un ECG normal"
        print(f"✓ Verificado: Train contiene {len(idx_norm_train)} ECG normales y 0 anómalos")
    
    # Guardar splits
    splits_dir = output_dir / "splits"
    ensure_dir(splits_dir)
    
    np.save(splits_dir / "idx_norm_train.npy", idx_norm_train)
    np.save(splits_dir / "idx_norm_val.npy", idx_norm_val)
    np.save(splits_dir / "idx_norm_test.npy", idx_norm_test)
    np.save(splits_dir / "idx_anom_val.npy", idx_anom_val)
    np.save(splits_dir / "idx_anom_test.npy", idx_anom_test)
    
    # Verificar que los archivos se guardaron correctamente
    assert (splits_dir / "idx_anom_val.npy").exists(), "Error: idx_anom_val.npy no se guardó"
    assert (splits_dir / "idx_anom_test.npy").exists(), "Error: idx_anom_test.npy no se guardó"
    
    # Verificar que los arrays no están vacíos si hay anómalos
    if n_anomalies > 0:
        loaded_val = np.load(splits_dir / "idx_anom_val.npy")
        loaded_test = np.load(splits_dir / "idx_anom_test.npy")
        assert len(loaded_val) > 0 or len(loaded_test) > 0, \
            f"Error: Anómalos no se dividieron correctamente. Total: {n_anomalies}, val: {len(loaded_val)}, test: {len(loaded_test)}"
        print(f"✓ Archivos guardados: val_anom={len(loaded_val)}, test_anom={len(loaded_test)}")
    
    # Guardar resumen
    summary = {
        "train": {
            "normals": int(len(idx_norm_train)),
            "anomalies": 0,  # CRÍTICO: train solo tiene normales
        },
        "val": {
            "normals": int(len(idx_norm_val)),
            "anomalies": int(len(idx_anom_val)),
        },
        "test": {
            "normals": int(len(idx_norm_test)),
            "anomalies": int(len(idx_anom_test)),
        },
    }
    
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    
    return {
        "train_norm": idx_norm_train,
        "val_norm": idx_norm_val,
        "val_anom": idx_anom_val,
        "test_norm": idx_norm_test,
        "test_anom": idx_anom_test,
    }


def _build_ptb_splits(M_norm: pd.DataFrame, M_anom: pd.DataFrame, output_root: Path) -> None:
    """Construye splits para PTB-XL usando strat_fold."""
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
    
    # Verificación: train solo debe tener normales
    if summary["train"]["n_anom"] > 0:
        print(f"⚠ ADVERTENCIA: Train contiene {summary['train']['n_anom']} anómalos. "
              "Esto puede no ser deseado para entrenamiento no supervisado.")
    else:
        print("✓ Train contiene solo normales (correcto para entrenamiento no supervisado)")


# =====================================================================================
# Funciones de utilidad para análisis
# =====================================================================================

def summarize_signal(x: np.ndarray, leads: Sequence[str]) -> pd.DataFrame:
    """Genera estadísticas descriptivas de una señal."""
    rows = []
    for idx, lead in enumerate(leads):
        col = x[:, idx]
        rows.append({
            "lead": lead,
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        })
    return pd.DataFrame(rows).set_index("lead")


def get_dataset_stats(dataset: str = "combined") -> Dict[str, object]:
    """
    Obtiene estadísticas del dataset.
    
    Returns:
        Diccionario con conteos de normales, anómalos, y distribución por split
    """
    if dataset == "combined":
        output_dir = COMBINED_OUTPUT
    elif dataset == "ptbxl":
        output_dir = PTB_OUTPUT
    elif dataset == "mimic":
        output_dir = MIMIC_OUTPUT
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")
    
    meta_norm = pd.read_csv(output_dir / "meta_norm.csv")
    n_normals = len(meta_norm)
    
    meta_anom_path = output_dir / "meta_anom.csv"
    if meta_anom_path.exists():
        meta_anom = pd.read_csv(meta_anom_path)
        n_anomalies = len(meta_anom)
    else:
        n_anomalies = 0
    
    splits_dir = output_dir / "splits"
    stats = {
        "total_normals": n_normals,
        "total_anomalies": n_anomalies,
        "total_ecg": n_normals + n_anomalies,
    }
    
    if splits_dir.exists():
        try:
            idx_norm_train = np.load(splits_dir / "idx_norm_train.npy")
            idx_norm_val = np.load(splits_dir / "idx_norm_val.npy")
            idx_norm_test = np.load(splits_dir / "idx_norm_test.npy")
            
            stats["train"] = {
                "normals": int(len(idx_norm_train)),
                "anomalies": 0,  # Train solo tiene normales
            }
            
            if (splits_dir / "idx_anom_val.npy").exists():
                idx_anom_val = np.load(splits_dir / "idx_anom_val.npy")
                stats["val"] = {
                    "normals": int(len(idx_norm_val)),
                    "anomalies": int(len(idx_anom_val)),
                }
            else:
                stats["val"] = {
                    "normals": int(len(idx_norm_val)),
                    "anomalies": 0,
                }
            
            if (splits_dir / "idx_anom_test.npy").exists():
                idx_anom_test = np.load(splits_dir / "idx_anom_test.npy")
                stats["test"] = {
                    "normals": int(len(idx_norm_test)),
                    "anomalies": int(len(idx_anom_test)),
                }
            else:
                stats["test"] = {
                    "normals": int(len(idx_norm_test)),
                    "anomalies": 0,
                }
        except FileNotFoundError:
            stats["splits"] = "No disponibles"
    
    return stats

