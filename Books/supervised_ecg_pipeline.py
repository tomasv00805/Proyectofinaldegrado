#!/usr/bin/env python
"""
Pipeline completo para preparar dataset supervisado binario de ECG (NORMAL vs ANÓMALO).

Este módulo implementa un pipeline robusto, parametrizable y documentado que:
- Etiqueta registros correctamente (MIMIC y PTB-XL)
- Filtra señales malas
- Aplica filtrado y normalización estándar
- Selecciona solo leads: II, V1 y V5
- Resamplea todo a 10 segundos y 500 Hz
- Genera datasets balanceados
- Genera train/val/test + 10 folds estratificados
- Guarda todo bajo /data/Datos_supervisados/
- Incluye funciones para visualizar y verificar

Autor: Pipeline ECG Supervisado
Fecha: 2024
"""

from __future__ import annotations

import ast
import json
import math
import multiprocessing as mp
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, resample
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


# =====================================================================================
# Configuración y constantes
# =====================================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PTB_ROOT = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
MIMIC_ROOT = PROJECT_ROOT / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
OUTPUT_DIR = DATA_DIR / "Datos_supervisados"

# Parámetros de señal
TARGET_LEADS = ["II", "V1", "V5"]  # Orden fijo: [II, V1, V5]
SAMPLING_RATE = 500  # Hz
SIGNAL_DURATION = 10  # segundos
EXPECTED_SAMPLES = SAMPLING_RATE * SIGNAL_DURATION  # 5000 muestras

# Parámetros de filtrado
NOTCH_FREQ = 50.0  # Hz (configurable 50/60)
NOTCH_Q = 30.0
BANDPASS_LOW = 0.5  # Hz
BANDPASS_HIGH = 40.0  # Hz
BANDPASS_ORDER = 4  # Orden del filtro Butterworth

# Parámetros de calidad de señal
MIN_STD = 0.001  # Desviación estándar mínima por lead
MAX_NAN_RATIO = 0.05  # Máximo 5% de NaN
MAX_FLAT_RATIO = 0.05  # Máximo 5% de valores constantes
SATURATION_THRESHOLD = 0.98  # Si 98% de valores están en min/max, es saturación

# Parámetros de splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
N_FOLDS = 10
RANDOM_STATE = 42

# Parámetros de optimización
BATCH_SIZE = 100  # Tamaño de batch para procesamiento
N_WORKERS = max(1, mp.cpu_count() - 1)  # Número de workers para paralelización
CHUNK_SIZE_MIMIC = 5000  # Tamaño de chunk para leer MIMIC CSV

# Mapeo de nombres de leads
LEAD_MAPPING = {
    "II": ["II", "MLII", "II-", "II+", "II_"],
    "V1": ["V1", "V1-", "V1+", "V1_"],
    "V5": ["V5", "V5-", "V5+", "V5_"],
}

# =====================================================================================
# Patrones de etiquetado para MIMIC
# =====================================================================================

# Patrones ANÓMALOS (label = 1) - case-insensitive
MIMIC_ANOMALY_PATTERNS = [
    # Clasificación general
    r"\babnormal\s+ecg\b",
    r"\bsummary:\s+abnormal\s+ecg\b",
    # Infartos
    r"\binfarct",
    r"\binfarction",
    # Isquemia
    r"\bischemia",
    r"\bmyocardial\s+ischemia",
    # Hipertrofia
    r"\bhypertrophy",
    # Arritmias
    r"\bpvc\b",
    r"\bventricular\s+contraction",
    r"\batrial\s+fibrillation",
    # Bloqueos
    r"\bbundle\s+branch\s+block",
    r"\blbbb\b",
    r"\brbbb\b",
    # ST-T
    r"\bst\s+elevation",
    r"\bst-t\s+changes",
    # Infartos específicos
    r"\bmi\b",  # Myocardial Infarction
    r"\bmyocardial\s+infarction",
    r"\bacute\s+st\s+elevation\s+mi\b",
    # QT prolongado
    r"\bprolonged\s+qt\s+interval",
    # Taquicardia ventricular
    r"\bventricular\s+tachycardia",
    # Bloqueo AV
    r"\ba-v\s+block\b",
]

# Patrones NORMALES (label = 0)
MIMIC_NORMAL_PATTERNS = [
    r"\bnormal\s+ecg\b",
    r"\bnormal\s+ecg\s+except\s+for\s+rate\b",
    r"\bnormal\s+ecg\s+based\s+on\s+available\s+leads\b",
    r"\bprobable\s+normal\s+variant\b",
    r"\bsinus\s+rhythm\b",  # Si no hay anormalidades
]

# Patrones DESCARTAR (ruido / no evaluable)
MIMIC_REJECT_PATTERNS = [
    r"\brecording\s+unsuitable\s+for\s+analysis\s+-\s+please\s+repeat\b",
    r"\banalysis\s+error\b",
    r"\bwarning:\s+data\s+quality\s+may\s+affect\s+interpretation\b",
    r"\blead\(s\)\s+unsuitable\s+for\s+analysis\b",
    r"\bunsuitable\s+for\s+analysis\b",
    r"\bcannot\s+rule\s+out\b.*unsuitable",
    r"\bfaulty\b",
    r"\brepeat\b.*quality",
]

# =====================================================================================
# Códigos SCP para PTB-XL
# =====================================================================================

# Códigos ANÓMALOS (label = 1)
PTB_ANOMALY_CODES = {
    # Infartos
    "IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "IPMI", "PMI", "IPLMI",
    # Isquemia
    "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCAN",
    "INJAS", "INJAL", "INJIN", "INJLA", "INJIL",
    # Hipertrofias
    "LVH", "RVH", "SEHYP", "LAO", "LAE", "RAO", "RAE",
    # Bloqueos
    "CLBBB", "CRBBB", "ILBBB", "IRBBB", "LAFB", "LPFB", "IVCD",
    "1AVB", "2AVB", "3AVB",
    # Arritmias
    "AFIB", "AFLT", "PVC", "PAC", "SVTAC", "PSVT", "SVARR",
    "BIGU", "TRIGU",
    # ST-T
    "NST_", "NDT", "STD_", "STE_", "TAB_", "INVT",
    "LOWT", "DIG", "LNGQT", "ANEUR",
    # Otros
    "ABQRS", "QWAVE", "WPW", "EL",
}

# Códigos NORMALES (label = 0)
PTB_NORMAL_CODES = {"NORM"}  # Con valor 100.0 o 80.0

# Patrones de rechazo para PTB-XL
PTB_REJECT_COLUMNS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
    "extra_beats",  # Como artefacto, no latido clínico
]


# =====================================================================================
# Utilidades generales
# =====================================================================================

def ensure_dir(path: Path) -> None:
    """Crea un directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_lead_name(name: str) -> str:
    """Normaliza el nombre de una derivación para comparación."""
    return name.replace("-", "").replace("_", "").replace(" ", "").upper()


def get_sig_names(meta: Union[dict, object]) -> List[str]:
    """Extrae los nombres de las señales desde metadatos wfdb."""
    if isinstance(meta, dict):
        return meta.get("sig_name", [])
    if hasattr(meta, "sig_name"):
        return list(meta.sig_name)
    raise TypeError(f"Metadatos wfdb inesperados: {type(meta)}")


def map_lead_names(sig_names: Sequence[str], target_leads: Sequence[str]) -> Dict[str, int]:
    """
    Mapea nombres de leads a índices, considerando equivalentes.
    
    Args:
        sig_names: Lista de nombres de leads en el registro
        target_leads: Lista de leads objetivo [II, V1, V5]
        
    Returns:
        Diccionario {lead: índice} para los leads encontrados
        
    Raises:
        KeyError: Si algún lead requerido no se encuentra
    """
    normalized_sig = {normalize_lead_name(n): idx for idx, n in enumerate(sig_names)}
    mapping = {}
    missing = []
    
    for target_lead in target_leads:
        found = False
        for variant in LEAD_MAPPING.get(target_lead, [target_lead]):
            variant_norm = normalize_lead_name(variant)
            if variant_norm in normalized_sig:
                mapping[target_lead] = normalized_sig[variant_norm]
                found = True
                break
        if not found:
            missing.append(target_lead)
    
    if missing:
        raise KeyError(f"Leads faltantes: {missing}. Leads disponibles: {sig_names}")
    
    return mapping


# =====================================================================================
# Etiquetado para MIMIC
# =====================================================================================

def normalize_text(text: str) -> str:
    """Normaliza texto para comparación (case-insensitive, limpia puntuación)."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip().lower()
    # Reemplazar caracteres especiales
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_mimic_record(
    report_text: Union[str, pd.Series],
    report_column: str = "report_1",
) -> Tuple[int, str]:
    """
    Etiqueta un registro de MIMIC como NORMAL (0), ANÓMALO (1) o DESCARTAR (-1).
    
    Args:
        report_text: Texto del reporte o Series de pandas con múltiples report_#
        report_column: Nombre de la columna si report_text es una Series
        
    Returns:
        Tupla (label, razón)
        - label: 0 (normal), 1 (anómalo), -1 (descartar)
        - razón: string explicando la decisión
    """
    # Si es una Series, concatenar todos los report_# que no sean NaN
    if isinstance(report_text, pd.Series):
        reports = []
        for col in report_text.index:
            if col.startswith("report_") and pd.notna(report_text[col]):
                reports.append(str(report_text[col]))
        text = " ".join(reports)
    else:
        text = str(report_text) if report_text is not None else ""
    
    normalized = normalize_text(text)
    
    if not normalized:
        return -1, "reporte_vacio"
    
    # 1. Verificar patrones de rechazo (prioridad máxima)
    for pattern in MIMIC_REJECT_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return -1, f"rechazo: {pattern}"
    
    # 2. Verificar patrones anómalos
    for pattern in MIMIC_ANOMALY_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return 1, f"anómalo: {pattern}"
    
    # 3. Verificar patrones normales
    # Si es "sinus rhythm", solo es normal si no hay anormalidades
    is_sinus = re.search(r"\bsinus\s+rhythm\b", normalized, re.IGNORECASE)
    if is_sinus:
        # Si hay patrón anómalo después de "sinus rhythm", es anómalo
        for pattern in MIMIC_ANOMALY_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return 1, f"anómalo_con_sinus: {pattern}"
        return 0, "normal: sinus_rhythm"
    
    for pattern in MIMIC_NORMAL_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return 0, f"normal: {pattern}"
    
    # Por defecto, si no coincide con nada, descartar
    return -1, "sin_patron_reconocido"


# =====================================================================================
# Etiquetado para PTB-XL
# =====================================================================================

def label_ptbxl_record(
    scp_codes: Union[Dict[str, float], str],
    quality_columns: Optional[pd.Series] = None,
    reject_unvalidated: bool = False,
    validated_by_human: Optional[bool] = None,
    initial_autogenerated: Optional[bool] = None,
) -> Tuple[int, str]:
    """
    Etiqueta un registro de PTB-XL como NORMAL (0), ANÓMALO (1) o DESCARTAR (-1).
    
    Args:
        scp_codes: Diccionario de códigos SCP o string que se puede evaluar
        quality_columns: Series con columnas de calidad (baseline_drift, etc.)
        reject_unvalidated: Si True, rechazar reportes no validados
        validated_by_human: Si se proporciona, usar este valor para validación
        initial_autogenerated: Si se proporciona, usar este valor para autogenerado
        
    Returns:
        Tupla (label, razón)
    """
    # Parsear scp_codes si es string
    if isinstance(scp_codes, str):
        try:
            scp_codes = ast.literal_eval(scp_codes)
        except (ValueError, SyntaxError):
            return -1, "scp_codes_invalido"
    
    if not isinstance(scp_codes, dict):
        return -1, "scp_codes_no_dict"
    
    # 1. Verificar calidad de señal (prioridad máxima)
    if quality_columns is not None:
        for col in PTB_REJECT_COLUMNS:
            if col in quality_columns.index:
                val = quality_columns[col]
                if pd.notna(val) and str(val).strip():
                    # Si hay problemas de electrodos con "alles", rechazar
                    if col == "electrodes_problems" and "alles" in str(val).lower():
                        return -1, f"rechazo: {col}=alles"
                    # Si hay problemas notables, rechazar
                    if str(val).strip():
                        return -1, f"rechazo: {col}={val}"
    
    # Verificar validación humana si está configurado
    if reject_unvalidated:
        if validated_by_human is False:
            return -1, "rechazo: no_validado_por_humano"
        if initial_autogenerated is True:
            return -1, "rechazo: reporte_autogenerado"
    
    # 2. Verificar códigos anómalos
    codes_present = set(scp_codes.keys())
    for code in codes_present:
        if code in PTB_ANOMALY_CODES:
            return 1, f"anómalo: {code}"
    
    # 3. Verificar códigos normales
    if "NORM" in codes_present:
        norm_value = scp_codes.get("NORM", 0.0)
        if norm_value in [100.0, 80.0]:
            # Solo es normal si no hay otros códigos patológicos
            has_pathological = any(c in PTB_ANOMALY_CODES for c in codes_present if c != "NORM")
            if not has_pathological:
                return 0, "normal: NORM"
    
    # Por defecto, descartar si no coincide con nada claro
    return -1, "sin_clasificacion_clara"


# =====================================================================================
# Filtros de calidad de señal
# =====================================================================================

def check_signal_quality(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    min_duration: float = SIGNAL_DURATION,
    min_std: float = MIN_STD,
    max_nan_ratio: float = MAX_NAN_RATIO,
    max_flat_ratio: float = MAX_FLAT_RATIO,
    saturation_threshold: float = SATURATION_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Verifica la calidad de una señal ECG.
    
    Args:
        signal: Array de forma [T, C] o [C, T] (se detecta automáticamente)
        fs: Frecuencia de muestreo
        min_duration: Duración mínima requerida en segundos
        min_std: Desviación estándar mínima por lead
        max_nan_ratio: Proporción máxima de NaN permitida
        max_flat_ratio: Proporción máxima de valores constantes
        saturation_threshold: Umbral para detección de saturación
        
    Returns:
        Tupla (es_valida, razon_rechazo)
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    elif signal.ndim == 2:
        # Asumir [T, C] si T > C, sino [C, T]
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
    
    T, C = signal.shape
    duration = T / fs
    
    # 1. Verificar duración
    if duration < min_duration:
        return False, f"duracion_insuficiente: {duration:.2f}s < {min_duration}s"
    
    # 2. Verificar NaN por lead
    for c in range(C):
        lead = signal[:, c]
        nan_ratio = np.isnan(lead).sum() / len(lead)
        if nan_ratio > max_nan_ratio:
            return False, f"lead_{c}_exceso_nan: {nan_ratio:.2%} > {max_nan_ratio:.2%}"
    
    # 3. Verificar flat signal por lead
    for c in range(C):
        lead = signal[:, c]
        lead_clean = lead[~np.isnan(lead)]
        if len(lead_clean) == 0:
            return False, f"lead_{c}_solo_nan"
        
        std = np.std(lead_clean)
        if std < min_std:
            return False, f"lead_{c}_flat: std={std:.6f} < {min_std}"
        
        # Verificar saturación (valores constantes)
        min_val = np.min(lead_clean)
        max_val = np.max(lead_clean)
        if min_val == max_val:
            return False, f"lead_{c}_constante"
        
        # Verificar saturación (valores pegados a min/max)
        at_min = (lead_clean == min_val).sum() / len(lead_clean)
        at_max = (lead_clean == max_val).sum() / len(lead_clean)
        if at_min > saturation_threshold or at_max > saturation_threshold:
            return False, f"lead_{c}_saturacion: min={at_min:.2%}, max={at_max:.2%}"
    
    # 4. Verificar discontinuidades severas (grandes saltos)
    for c in range(C):
        lead = signal[:, c]
        lead_clean = lead[~np.isnan(lead)]
        if len(lead_clean) < 2:
            continue
        
        diff = np.abs(np.diff(lead_clean))
        # Si hay saltos > 10x la mediana, posible artefacto
        median_diff = np.median(diff)
        if median_diff > 0:
            large_jumps = (diff > 10 * median_diff).sum()
            if large_jumps > len(diff) * 0.05:  # Más del 5% son saltos grandes
                return False, f"lead_{c}_discontinuidades_severas"
    
    return True, "ok"


# =====================================================================================
# Filtrado de señales
# =====================================================================================

def apply_notch_filter(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    freq: float = NOTCH_FREQ,
    q: float = NOTCH_Q,
) -> np.ndarray:
    """
    Aplica filtro notch a la frecuencia especificada (50 o 60 Hz).
    
    Args:
        signal: Señal de forma [T, C] o [T]
        fs: Frecuencia de muestreo
        freq: Frecuencia del notch (50 o 60 Hz)
        q: Factor de calidad
        
    Returns:
        Señal filtrada
    """
    nyquist = fs / 2.0
    w0 = freq / nyquist
    b, a = iirnotch(w0, Q=q)
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return filtfilt(b, a, signal, axis=0)


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """
    Aplica filtro pasa-banda Butterworth.
    
    Args:
        signal: Señal de forma [T, C] o [T]
        fs: Frecuencia de muestreo
        low: Frecuencia de corte inferior (Hz)
        high: Frecuencia de corte superior (Hz)
        order: Orden del filtro
        
    Returns:
        Señal filtrada
    """
    nyquist = fs / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype="bandpass")
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return filtfilt(b, a, signal, axis=0)


def filter_signal(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ,
    apply_bandpass: bool = True,
) -> np.ndarray:
    """
    Aplica filtros a una señal ECG.
    
    Args:
        signal: Señal de forma [T, C] o [T]
        fs: Frecuencia de muestreo
        apply_notch: Si aplicar filtro notch
        notch_freq: Frecuencia del notch (50 o 60 Hz)
        apply_bandpass: Si aplicar filtro pasa-banda
        
    Returns:
        Señal filtrada
    """
    filtered = signal.copy().astype(np.float32)
    
    if apply_notch:
        filtered = apply_notch_filter(filtered, fs=fs, freq=notch_freq)
    
    if apply_bandpass:
        filtered = apply_bandpass_filter(filtered, fs=fs)
    
    return filtered


# =====================================================================================
# Normalización
# =====================================================================================

def normalize_signal_minmax(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza señal usando Min-Max a [0, 1] por lead.
    
    Args:
        signal: Señal de forma [T, C] o [T]
        
    Returns:
        Tupla (señal_normalizada, mins, maxs)
        Si max == min en algún lead, se descarta (retorna NaN)
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    mins = np.min(signal, axis=0, keepdims=True)
    maxs = np.max(signal, axis=0, keepdims=True)
    
    # Verificar flat signals
    ranges = maxs - mins
    if np.any(ranges == 0):
        # Lead constante, retornar NaN
        normalized = np.full_like(signal, np.nan)
        if squeeze:
            normalized = normalized.flatten()
        return normalized, mins.squeeze(), maxs.squeeze()
    
    normalized = (signal - mins) / ranges
    normalized = normalized.astype(np.float32)
    
    if squeeze:
        normalized = normalized.flatten()
        mins = mins.flatten()
        maxs = maxs.flatten()
    
    return normalized, mins.squeeze(), maxs.squeeze()


def normalize_signal_zscore(signal: np.ndarray) -> np.ndarray:
    """
    Normaliza señal usando Z-score por lead.
    
    Args:
        signal: Señal de forma [T, C] o [T]
        
    Returns:
        Señal normalizada
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    
    # Evitar división por cero
    std = np.where(std == 0, 1.0, std)
    
    normalized = (signal - mean) / std
    normalized = normalized.astype(np.float32)
    
    if squeeze:
        normalized = normalized.flatten()
    
    return normalized


# =====================================================================================
# Resampleo temporal
# =====================================================================================

def resample_signal(
    signal: np.ndarray,
    original_fs: float,
    target_fs: float = SAMPLING_RATE,
    target_duration: float = SIGNAL_DURATION,
    strategy: str = "center",
) -> np.ndarray:
    """
    Resamplea una señal a la frecuencia y duración objetivo.
    
    Args:
        signal: Señal de forma [T, C] o [T]
        original_fs: Frecuencia de muestreo original
        target_fs: Frecuencia de muestreo objetivo
        target_duration: Duración objetivo en segundos
        strategy: "center" (ventana central), "start" (inicio), "random" (aleatorio)
        
    Returns:
        Señal resampleada de forma [target_samples, C] o [target_samples]
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    T_orig, C = signal.shape
    duration_orig = T_orig / original_fs
    target_samples = int(target_fs * target_duration)
    
    # Si la señal es más corta que la duración objetivo, rechazar
    if duration_orig < target_duration:
        raise ValueError(
            f"Duración insuficiente: {duration_orig:.2f}s < {target_duration}s"
        )
    
    # Si es más larga, recortar
    if duration_orig > target_duration:
        samples_orig_target = int(original_fs * target_duration)
        
        if strategy == "center":
            start_idx = (T_orig - samples_orig_target) // 2
            end_idx = start_idx + samples_orig_target
            signal = signal[start_idx:end_idx, :]
        elif strategy == "start":
            signal = signal[:samples_orig_target, :]
        elif strategy == "random":
            max_start = T_orig - samples_orig_target
            start_idx = np.random.randint(0, max_start + 1)
            end_idx = start_idx + samples_orig_target
            signal = signal[start_idx:end_idx, :]
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
    
    # Resamplear
    if original_fs != target_fs:
        T_current = signal.shape[0]
        num_samples_target = int(T_current * target_fs / original_fs)
        
        # Resamplear cada lead por separado
        resampled = np.zeros((num_samples_target, C), dtype=np.float32)
        for c in range(C):
            resampled[:, c] = resample(signal[:, c], num_samples_target)
        
        signal = resampled
    
    # Asegurar tamaño exacto
    if signal.shape[0] != target_samples:
        # Interpolación final si es necesario
        resampled = np.zeros((target_samples, C), dtype=np.float32)
        for c in range(C):
            resampled[:, c] = resample(signal[:, c], target_samples)
        signal = resampled
    
    if squeeze:
        signal = signal.flatten()
    
    return signal.astype(np.float32)


# =====================================================================================
# Carga y procesamiento de registros
# =====================================================================================

def load_ptbxl_record(
    record_id: int,
    metadata: pd.DataFrame,
    records_dir: Path = PTB_ROOT,
) -> Tuple[np.ndarray, dict]:
    """
    Carga un registro de PTB-XL y extrae solo los leads II, V1, V5.
    
    Args:
        record_id: ID del ECG en PTB-XL
        metadata: DataFrame con metadatos
        records_dir: Directorio raíz de PTB-XL
        
    Returns:
        Tupla (señal [T, 3], metadatos)
    """
    filename_hr = metadata.loc[record_id, "filename_hr"]
    record_path = records_dir / filename_hr
    
    signal, meta = wfdb.rdsamp(str(record_path))
    sig_names = get_sig_names(meta)
    
    # Mapear leads
    lead_mapping = map_lead_names(sig_names, TARGET_LEADS)
    
    # Extraer leads en orden: [II, V1, V5]
    indices = [lead_mapping[lead] for lead in TARGET_LEADS]
    selected = signal[:, indices].astype(np.float32)
    
    return selected, meta


def load_mimic_record(
    record_path: Path,
) -> Tuple[np.ndarray, dict]:
    """
    Carga un registro de MIMIC y extrae solo los leads II, V1, V5.
    
    Args:
        record_path: Path al archivo de registro (sin extensión)
        
    Returns:
        Tupla (señal [T, 3], metadatos)
    """
    signal, meta = wfdb.rdsamp(str(record_path))
    sig_names = get_sig_names(meta)
    
    # Mapear leads
    lead_mapping = map_lead_names(sig_names, TARGET_LEADS)
    
    # Extraer leads en orden: [II, V1, V5]
    indices = [lead_mapping[lead] for lead in TARGET_LEADS]
    selected = signal[:, indices].astype(np.float32)
    
    return selected, meta


def process_single_record(
    signal: np.ndarray,
    original_fs: float,
    apply_quality_check: bool = True,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ,
    apply_bandpass: bool = True,
    normalize_method: str = "minmax",
    resample_strategy: str = "center",
) -> Tuple[Optional[np.ndarray], str]:
    """
    Procesa un registro completo: calidad -> filtrado -> resampleo -> normalización.
    
    Args:
        signal: Señal cruda [T, 3]
        original_fs: Frecuencia de muestreo original
        apply_quality_check: Si verificar calidad
        apply_notch: Si aplicar notch
        notch_freq: Frecuencia del notch
        apply_bandpass: Si aplicar bandpass
        normalize_method: "minmax" o "zscore"
        resample_strategy: "center", "start", o "random"
        
    Returns:
        Tupla (señal_procesada, mensaje)
        Si hay error, retorna (None, mensaje_de_error)
    """
    try:
        # 1. Verificar calidad
        if apply_quality_check:
            is_valid, reason = check_signal_quality(signal, fs=original_fs)
            if not is_valid:
                return None, f"calidad_rechazada: {reason}"
        
        # 2. Filtrar
        filtered = filter_signal(
            signal,
            fs=original_fs,
            apply_notch=apply_notch,
            notch_freq=notch_freq,
            apply_bandpass=apply_bandpass,
        )
        
        # 3. Resamplear
        resampled = resample_signal(
            filtered,
            original_fs=original_fs,
            strategy=resample_strategy,
        )
        
        # 4. Normalizar
        if normalize_method == "minmax":
            normalized, _, _ = normalize_signal_minmax(resampled)
            if np.any(np.isnan(normalized)):
                return None, "normalizacion_fallida: señal_flat"
        elif normalize_method == "zscore":
            normalized = normalize_signal_zscore(resampled)
        else:
            return None, f"metodo_normalizacion_desconocido: {normalize_method}"
        
        return normalized, "ok"
    
    except Exception as e:
        return None, f"error_procesamiento: {str(e)}"


# =====================================================================================
# Dataset y balanceo
# =====================================================================================

def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
    return_indices: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Balancea dataset haciendo downsampling estratificado de la clase mayoritaria.
    
    Args:
        X: Array de señales [N, T, C]
        y: Array de etiquetas [N]
        random_state: Seed para reproducibilidad
        return_indices: Si retornar también los índices seleccionados
        
    Returns:
        Si return_indices=False: Tupla (X_balanced, y_balanced)
        Si return_indices=True: Tupla (X_balanced, y_balanced, indices)
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    
    np.random.seed(random_state)
    
    balanced_indices = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        if len(indices) > n_min:
            # Downsampling aleatorio
            selected = np.random.choice(indices, size=n_min, replace=False)
            balanced_indices.extend(selected.tolist())
        else:
            # Usar todos
            balanced_indices.extend(indices.tolist())
    
    balanced_indices = np.array(balanced_indices)
    np.random.seed(random_state)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    if return_indices:
        return X_balanced, y_balanced, balanced_indices
    else:
        return X_balanced, y_balanced


# =====================================================================================
# Splits y folds
# =====================================================================================

def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea splits train/val/test estratificados usando enfoque eficiente en memoria.
    Evita crear múltiples copias simultáneas para reducir uso de memoria.
    
    Args:
        X: Array de señales [N, T, C]
        y: Array de etiquetas [N]
        train_ratio: Proporción de train
        val_ratio: Proporción de val
        test_ratio: Proporción de test
        random_state: Seed
        
    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Usar StratifiedShuffleSplit para obtener índices sin copiar arrays
    # Primero separar test del resto
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_state
    )
    train_val_idx, test_idx = next(sss.split(X, y))
    
    # Crear test split primero (más pequeño, ~15%)
    print(f"    Creando test split ({len(test_idx)} registros)...")
    X_test = X[test_idx].copy()
    y_test = y[test_idx].copy()
    del test_idx  # Liberar memoria del índice
    
    # Ahora separar train y val del resto usando índices anidados
    # para evitar crear X_temp
    val_size = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state
    )
    
    # Usar solo los índices de train_val_idx, no crear array intermedio
    y_temp = y[train_val_idx]  # Solo necesitamos y para estratificar
    train_local_idx, val_local_idx = next(sss2.split(np.arange(len(train_val_idx)), y_temp))
    
    # Mapear índices locales a índices globales
    train_idx = train_val_idx[train_local_idx]
    val_idx = train_val_idx[val_local_idx]
    
    # Limpiar índices intermedios
    del train_val_idx, train_local_idx, val_local_idx, y_temp
    
    # Crear splits uno a la vez para minimizar memoria pico
    print(f"    Creando train split ({len(train_idx)} registros)...")
    X_train = X[train_idx].copy()
    y_train = y[train_idx].copy()
    del train_idx
    
    print(f"    Creando val split ({len(val_idx)} registros)...")
    X_val = X[val_idx].copy()
    y_val = y[val_idx].copy()
    del val_idx
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_splits_to_disk(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    chunk_size: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea splits train/val/test estratificados y los guarda directamente en disco.
    Procesa en chunks para minimizar uso de memoria.
    
    Args:
        X: Array de señales [N, T, C]
        y: Array de etiquetas [N]
        output_dir: Directorio donde guardar los splits
        train_ratio: Proporción de train
        val_ratio: Proporción de val
        test_ratio: Proporción de test
        random_state: Seed
        chunk_size: Tamaño de chunk para procesamiento
        
    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx)
        Los datos reales se guardan en disco en output_dir/numpy/
        Los índices se retornan para poder mapear metadata correctamente
    """
    ensure_dir(output_dir / "numpy")
    
    # Definir rutas de archivos
    test_path = output_dir / "numpy" / "X_test.npy"
    y_test_path = output_dir / "numpy" / "y_test.npy"
    val_path = output_dir / "numpy" / "X_val.npy"
    y_val_path = output_dir / "numpy" / "y_val.npy"
    train_path = output_dir / "numpy" / "X_train.npy"
    y_train_path = output_dir / "numpy" / "y_train.npy"
    
    # Intentar eliminar archivos existentes si existen (para evitar errores)
    # Si están en uso, simplemente los sobrescribiremos con np.save
    import os
    import gc
    for path in [test_path, y_test_path, val_path, y_val_path, train_path, y_train_path]:
        if path.exists():
            try:
                path.unlink()
            except (PermissionError, OSError):
                # Si el archivo está en uso, forzar garbage collection y reintentar
                gc.collect()
                try:
                    path.unlink()
                except (PermissionError, OSError):
                    # Si aún no se puede, np.save sobrescribirá el archivo
                    # pero primero intentamos renombrarlo
                    try:
                        temp_path = path.with_suffix('.tmp')
                        if temp_path.exists():
                            temp_path.unlink()
                        path.rename(temp_path)
                    except (PermissionError, OSError):
                        # Si todo falla, simplemente continuar - np.save debería sobrescribir
                        pass
    
    # Usar StratifiedShuffleSplit para obtener índices sin copiar arrays
    print(f"    Calculando índices de splits...")
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_state
    )
    train_val_idx, test_idx = next(sss.split(X, y))
    
    # Separar train y val
    val_size = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state
    )
    y_temp = y[train_val_idx]
    train_local_idx, val_local_idx = next(sss2.split(np.arange(len(train_val_idx)), y_temp))
    train_idx = train_val_idx[train_local_idx]
    val_idx = train_val_idx[val_local_idx]
    
    del train_val_idx, train_local_idx, val_local_idx, y_temp
    
    # Guardar test split - escribir directamente (test es pequeño, ~15%)
    print(f"    Guardando test split ({len(test_idx)} registros)...")
    X_test_data = X[test_idx]
    y_test_data = y[test_idx]
    np.save(str(test_path), X_test_data)
    np.save(str(y_test_path), y_test_data)
    
    # Guardar copia de test_idx antes de eliminarlo
    test_idx_copy = test_idx.copy()
    del X_test_data, y_test_data, test_idx
    
    # Guardar val split - escribir directamente (val es pequeño, ~15%)
    print(f"    Guardando val split ({len(val_idx)} registros)...")
    X_val_data = X[val_idx]
    y_val_data = y[val_idx]
    np.save(str(val_path), X_val_data)
    np.save(str(y_val_path), y_val_data)
    
    # Guardar copia de val_idx antes de eliminarlo
    val_idx_copy = val_idx.copy()
    del X_val_data, y_val_data, val_idx
    
    # Guardar train split en chunks (el más grande, necesita procesamiento por chunks)
    print(f"    Guardando train split ({len(train_idx)} registros) en chunks...")
    
    # Para train, procesar en chunks para evitar problemas de memoria
    # Crear lista de chunks y guardar uno por uno
    train_chunks_X = []
    train_chunks_y = []
    
    for i in range(0, len(train_idx), chunk_size):
        end_idx = min(i + chunk_size, len(train_idx))
        chunk_idx = train_idx[i:end_idx]
        train_chunks_X.append(X[chunk_idx].copy())  # Copiar para liberar referencia
        train_chunks_y.append(y[chunk_idx].copy())
    
    # Concatenar chunks (esto puede usar memoria, pero train es ~70% del total)
    X_train_data = np.concatenate(train_chunks_X, axis=0)
    y_train_data = np.concatenate(train_chunks_y, axis=0)
    np.save(str(train_path), X_train_data)
    np.save(str(y_train_path), y_train_data)
    
    # Guardar copia de train_idx antes de eliminarlo
    train_idx_copy = train_idx.copy()
    del train_chunks_X, train_chunks_y, X_train_data, y_train_data, train_idx
    
    # Guardar índices en disco
    indices_path = output_dir / "numpy" / "split_indices.npz"
    np.savez(
        str(indices_path),
        train_idx=train_idx_copy,
        val_idx=val_idx_copy,
        test_idx=test_idx_copy
    )
    
    # Cargar arrays pequeños para retornar (solo para compatibilidad)
    # Los datos reales están en disco
    print(f"    Cargando arrays en memoria (solo para compatibilidad)...")
    X_train = np.load(train_path, mmap_mode='r')
    y_train = np.load(y_train_path, mmap_mode='r')
    X_val = np.load(val_path, mmap_mode='r')
    y_val = np.load(y_val_path, mmap_mode='r')
    X_test = np.load(test_path, mmap_mode='r')
    y_test = np.load(y_test_path, mmap_mode='r')
    
    print(f"    ✓ Splits guardados en {output_dir / 'numpy'}")
    print(f"    ✓ Índices guardados en {indices_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx_copy, val_idx_copy, test_idx_copy


def create_stratified_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crea folds estratificados para cross-validation.
    
    Args:
        X: Array de señales [N, T, C]
        y: Array de etiquetas [N]
        n_splits: Número de folds
        random_state: Seed
        
    Returns:
        Lista de tuplas (train_indices, val_indices) para cada fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))
    return folds


# =====================================================================================
# Visualización
# =====================================================================================

def plot_ecg_comparison(
    raw: np.ndarray,
    filtered: Optional[np.ndarray],
    normalized: Optional[np.ndarray],
    fs: float = SAMPLING_RATE,
    lead_names: Sequence[str] = TARGET_LEADS,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Visualiza comparación de señal cruda vs filtrada vs normalizada.
    
    Args:
        raw: Señal cruda [T, C]
        filtered: Señal filtrada [T, C] (opcional)
        normalized: Señal normalizada [T, C] (opcional)
        fs: Frecuencia de muestreo
        lead_names: Nombres de los leads
        title: Título del gráfico
        save_path: Path para guardar (opcional)
        figsize: Tamaño de la figura
    """
    import matplotlib.pyplot as plt
    
    n_leads = raw.shape[1]
    time = np.arange(len(raw)) / fs
    
    n_plots = 1
    if filtered is not None:
        n_plots += 1
    if normalized is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, n_leads, figsize=figsize)
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # Raw
    for c in range(n_leads):
        axes[plot_idx, c].plot(time, raw[:, c])
        axes[plot_idx, c].set_title(f"Cruda - {lead_names[c]}")
        axes[plot_idx, c].set_xlabel("Tiempo (s)")
        axes[plot_idx, c].set_ylabel("Amplitud")
        axes[plot_idx, c].grid(True)
    plot_idx += 1
    
    # Filtered
    if filtered is not None:
        for c in range(n_leads):
            axes[plot_idx, c].plot(time[:len(filtered)], filtered[:, c])
            axes[plot_idx, c].set_title(f"Filtrada - {lead_names[c]}")
            axes[plot_idx, c].set_xlabel("Tiempo (s)")
            axes[plot_idx, c].set_ylabel("Amplitud")
            axes[plot_idx, c].grid(True)
        plot_idx += 1
    
    # Normalized
    if normalized is not None:
        time_norm = np.arange(len(normalized)) / fs
        for c in range(n_leads):
            axes[plot_idx, c].plot(time_norm, normalized[:, c])
            axes[plot_idx, c].set_title(f"Normalizada - {lead_names[c]}")
            axes[plot_idx, c].set_xlabel("Tiempo (s)")
            axes[plot_idx, c].set_ylabel("Amplitud [0,1]")
            axes[plot_idx, c].grid(True)
            axes[plot_idx, c].set_ylim(-0.1, 1.1)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ecg(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    lead_names: Sequence[str] = TARGET_LEADS,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualiza una señal ECG simple.
    
    Args:
        signal: Señal [T, C]
        fs: Frecuencia de muestreo
        lead_names: Nombres de los leads
        title: Título
        save_path: Path para guardar
    """
    import matplotlib.pyplot as plt
    
    time = np.arange(len(signal)) / fs
    n_leads = signal.shape[1]
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 3 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]
    
    for c in range(n_leads):
        axes[c].plot(time, signal[:, c])
        axes[c].set_ylabel(f"{lead_names[c]}\nAmplitud")
        axes[c].grid(True)
    
    axes[-1].set_xlabel("Tiempo (s)")
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =====================================================================================
# Procesamiento completo de datasets
# =====================================================================================

def process_ptbxl_dataset(
    overwrite: bool = False,
    apply_quality_check: bool = True,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ,
    normalize_method: str = "minmax",
    reject_unvalidated: bool = False,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Procesa el dataset completo de PTB-XL.
    
    Args:
        overwrite: Si sobrescribir archivos existentes
        apply_quality_check: Si aplicar verificación de calidad
        apply_notch: Si aplicar filtro notch
        notch_freq: Frecuencia del notch
        normalize_method: Método de normalización
        reject_unvalidated: Si rechazar reportes no validados
        max_records: Máximo de registros a procesar (None = todos)
        verbose: Si imprimir progreso
        
    Returns:
        DataFrame con metadatos de todos los registros procesados
    """
    db_csv = PTB_ROOT / "ptbxl_database.csv"
    if not db_csv.exists():
        raise FileNotFoundError(f"No se encontró {db_csv}")
    
    metadata = pd.read_csv(db_csv, index_col="ecg_id")
    metadata["scp_codes"] = metadata["scp_codes"].apply(ast.literal_eval)
    
    processed_records = []
    skipped_labels = {"rejected": 0, "quality": 0, "processing_error": 0, "missing_leads": 0}
    
    total = len(metadata) if max_records is None else min(max_records, len(metadata))
    
    if verbose:
        print(f"[PTB-XL] Procesando {total} registros...")
    
    for idx, (ecg_id, row) in enumerate(metadata.iterrows()):
        if max_records and idx >= max_records:
            break
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Procesados: {idx + 1}/{total} | "
                  f"Válidos: {len(processed_records)} | "
                  f"Rechazados: {sum(skipped_labels.values())}")
        
        try:
            # 1. Etiquetar
            quality_cols = row[PTB_REJECT_COLUMNS] if all(c in row.index for c in PTB_REJECT_COLUMNS) else None
            label, reason = label_ptbxl_record(
                row["scp_codes"],
                quality_cols,
                reject_unvalidated=reject_unvalidated,
                validated_by_human=row.get("validated_by_human"),
                initial_autogenerated=row.get("initial_autogenerated_report"),
            )
            
            if label == -1:
                skipped_labels["rejected"] += 1
                continue
            
            # 2. Cargar señal
            try:
                signal, meta = load_ptbxl_record(ecg_id, metadata)
                original_fs = meta.get("fs", 500) if isinstance(meta, dict) else getattr(meta, "fs", 500)
            except (KeyError, FileNotFoundError, ValueError) as e:
                skipped_labels["missing_leads"] += 1
                continue
            
            # 3. Procesar
            processed, proc_reason = process_single_record(
                signal,
                original_fs,
                apply_quality_check=apply_quality_check,
                apply_notch=apply_notch,
                notch_freq=notch_freq,
                normalize_method=normalize_method,
            )
            
            if processed is None:
                skipped_labels["quality"] += 1
                if "calidad" in proc_reason:
                    skipped_labels["quality"] += 1
                else:
                    skipped_labels["processing_error"] += 1
                continue
            
            # 4. Guardar registro procesado
            processed_records.append({
                "record_id": f"ptbxl_{ecg_id}",
                "source": "PTB-XL",
                "original_id": int(ecg_id),
                "label": int(label),
                "label_reason": reason,
                "shape": f"{processed.shape[0]},{processed.shape[1]}",
                "signal": processed,  # Guardaremos esto después
            })
        
        except Exception as e:
            skipped_labels["processing_error"] += 1
            if verbose:
                print(f"  Error procesando PTB-XL {ecg_id}: {e}")
            continue
    
    if verbose:
        print(f"\n[PTB-XL] Completado:")
        print(f"  Procesados exitosamente: {len(processed_records)}")
        print(f"  Rechazados por etiquetado: {skipped_labels['rejected']}")
        print(f"  Rechazados por calidad: {skipped_labels['quality']}")
        print(f"  Errores de procesamiento: {skipped_labels['processing_error']}")
        print(f"  Leads faltantes: {skipped_labels['missing_leads']}")
    
    df = pd.DataFrame(processed_records)
    return df


def process_mimic_dataset(
    overwrite: bool = False,
    report_column: str = "report_1",
    apply_quality_check: bool = True,
    apply_notch: bool = True,
    notch_freq: float = NOTCH_FREQ,
    normalize_method: str = "minmax",
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Procesa el dataset completo de MIMIC-IV-ECG.
    
    Args:
        overwrite: Si sobrescribir archivos existentes
        report_column: Nombre base de columna de reporte (se buscarán report_1, report_2, etc.)
        apply_quality_check: Si aplicar verificación de calidad
        apply_notch: Si aplicar filtro notch
        notch_freq: Frecuencia del notch
        normalize_method: Método de normalización
        max_records: Máximo de registros a procesar (None = todos)
        verbose: Si imprimir progreso
        
    Returns:
        DataFrame con metadatos de todos los registros procesados
    """
    machine_csv = MIMIC_ROOT / "machine_measurements.csv"
    record_list_csv = MIMIC_ROOT / "record_list.csv"
    
    if not machine_csv.exists():
        raise FileNotFoundError(f"No se encontró {machine_csv}")
    if not record_list_csv.exists():
        raise FileNotFoundError(f"No se encontró {record_list_csv}")
    
    record_df = pd.read_csv(record_list_csv)
    path_map = {
        (int(row.subject_id), int(row.study_id)): row.path
        for row in record_df.itertuples(index=False)
    }
    
    processed_records = []
    skipped_labels = {"rejected": 0, "quality": 0, "processing_error": 0, "missing_leads": 0, "missing_path": 0}
    
    # Leer CSV de machine measurements
    chunksize = 1000
    total_processed = 0
    
    if verbose:
        print("[MIMIC] Leyendo registros...")
    
    for chunk in pd.read_csv(machine_csv, chunksize=chunksize):
        for idx, row in chunk.iterrows():
            if max_records and total_processed >= max_records:
                break
            
            total_processed += 1
            
            if verbose and total_processed % 100 == 0:
                print(f"  Procesados: {total_processed} | "
                      f"Válidos: {len(processed_records)} | "
                      f"Rechazados: {sum(skipped_labels.values())}")
            
            try:
                subject_id = int(row["subject_id"])
                study_id = int(row["study_id"])
                
                # 1. Buscar path
                key = (subject_id, study_id)
                if key not in path_map:
                    skipped_labels["missing_path"] += 1
                    continue
                
                record_path_str = path_map[key]
                record_path = MIMIC_ROOT / record_path_str
                
                # 2. Etiquetar basado en reportes
                # Buscar todas las columnas report_#
                report_cols = [col for col in row.index if col.startswith("report_")]
                if not report_cols:
                    skipped_labels["rejected"] += 1
                    continue
                
                reports_series = row[report_cols]
                label, reason = label_mimic_record(reports_series)
                
                if label == -1:
                    skipped_labels["rejected"] += 1
                    continue
                
                # 3. Cargar señal
                try:
                    signal, meta = load_mimic_record(record_path)
                    original_fs = meta.get("fs", 250) if isinstance(meta, dict) else getattr(meta, "fs", 250)
                except (KeyError, FileNotFoundError, ValueError) as e:
                    skipped_labels["missing_leads"] += 1
                    continue
                
                # 4. Procesar
                processed, proc_reason = process_single_record(
                    signal,
                    original_fs,
                    apply_quality_check=apply_quality_check,
                    apply_notch=apply_notch,
                    notch_freq=notch_freq,
                    normalize_method=normalize_method,
                )
                
                if processed is None:
                    if "calidad" in proc_reason:
                        skipped_labels["quality"] += 1
                    else:
                        skipped_labels["processing_error"] += 1
                    continue
                
                # 5. Guardar registro procesado
                processed_records.append({
                    "record_id": f"mimic_{subject_id}_{study_id}",
                    "source": "MIMIC",
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "label": int(label),
                    "label_reason": reason,
                    "shape": f"{processed.shape[0]},{processed.shape[1]}",
                    "signal": processed,  # Guardaremos esto después
                })
            
            except Exception as e:
                skipped_labels["processing_error"] += 1
                if verbose:
                    print(f"  Error procesando MIMIC {key}: {e}")
                continue
        
        if max_records and total_processed >= max_records:
            break
    
    if verbose:
        print(f"\n[MIMIC] Completado:")
        print(f"  Procesados exitosamente: {len(processed_records)}")
        print(f"  Rechazados por etiquetado: {skipped_labels['rejected']}")
        print(f"  Rechazados por calidad: {skipped_labels['quality']}")
        print(f"  Errores de procesamiento: {skipped_labels['processing_error']}")
        print(f"  Leads faltantes: {skipped_labels['missing_leads']}")
        print(f"  Paths faltantes: {skipped_labels['missing_path']}")
    
    df = pd.DataFrame(processed_records)
    return df


def build_supervised_dataset(
    ptbxl_df: Optional[pd.DataFrame] = None,
    mimic_df: Optional[pd.DataFrame] = None,
    balance: bool = True,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Construye el dataset final combinado.
    
    Args:
        ptbxl_df: DataFrame de PTB-XL procesado
        mimic_df: DataFrame de MIMIC procesado
        balance: Si balancear el dataset
        random_state: Seed
        
    Returns:
        Tupla (X, y, metadata)
        - X: Array de señales [N, T, C]
        - y: Array de etiquetas [N]
        - metadata: DataFrame con metadatos
    """
    all_records = []
    
    if ptbxl_df is not None and len(ptbxl_df) > 0:
        all_records.append(ptbxl_df)
    
    if mimic_df is not None and len(mimic_df) > 0:
        all_records.append(mimic_df)
    
    if not all_records:
        raise ValueError("No hay registros procesados")
    
    combined_df = pd.concat(all_records, ignore_index=True)
    
    # Extraer señales y etiquetas
    signals = np.stack([rec["signal"] for rec in combined_df.to_dict("records")], axis=0)
    labels = combined_df["label"].values
    
    # Balancear si se solicita
    if balance:
        signals, labels = balance_dataset(signals, labels, random_state=random_state)
        # Re-indexar metadata
        metadata = combined_df.iloc[:len(labels)].copy()
        metadata.reset_index(drop=True, inplace=True)
    else:
        metadata = combined_df.copy()
    
    # Remover columna 'signal' del metadata (ya está en X)
    if "signal" in metadata.columns:
        metadata = metadata.drop(columns=["signal"])
    
    return signals, labels, metadata


def save_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata_train: pd.DataFrame,
    metadata_val: pd.DataFrame,
    metadata_test: pd.DataFrame,
    folds_train: List[np.ndarray],
    folds_val: List[np.ndarray],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Guarda el dataset completo en la estructura especificada.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Arrays de datos
        metadata_train, metadata_val, metadata_test: DataFrames de metadatos
        folds_train, folds_val: Listas de índices para folds
        output_dir: Directorio de salida
    """
    print("\nIniciando guardado de dataset...")
    
    ensure_dir(output_dir)
    ensure_dir(output_dir / "metadata")
    ensure_dir(output_dir / "numpy")
    ensure_dir(output_dir / "raw_examples")
    
    # Guardar arrays numpy con progreso
    # Verificar si los arrays ya están guardados en disco (memoria mapeada)
    print(f"  Guardando arrays numpy...")
    
    numpy_dir = output_dir / "numpy"
    train_path = numpy_dir / "X_train.npy"
    val_path = numpy_dir / "X_val.npy"
    test_path = numpy_dir / "X_test.npy"
    
    # Verificar si los arrays son memoria mapeada o si los archivos ya existen
    is_memmap_train = isinstance(X_train, np.memmap) or train_path.exists()
    is_memmap_val = isinstance(X_val, np.memmap) or val_path.exists()
    is_memmap_test = isinstance(X_test, np.memmap) or test_path.exists()
    
    if is_memmap_train and train_path.exists():
        print(f"    ⏭ X_train.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando X_train.npy ({X_train.shape})...")
        try:
            np.save(train_path, X_train)
            print(f"    ✓ X_train.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar X_train.npy (puede estar en uso): {e}")
    
    if is_memmap_train and (numpy_dir / "y_train.npy").exists():
        print(f"    ⏭ y_train.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando y_train.npy ({y_train.shape})...")
        try:
            np.save(numpy_dir / "y_train.npy", y_train)
            print(f"    ✓ y_train.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar y_train.npy (puede estar en uso): {e}")
    
    if is_memmap_val and val_path.exists():
        print(f"    ⏭ X_val.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando X_val.npy ({X_val.shape})...")
        try:
            np.save(val_path, X_val)
            print(f"    ✓ X_val.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar X_val.npy (puede estar en uso): {e}")
    
    if is_memmap_val and (numpy_dir / "y_val.npy").exists():
        print(f"    ⏭ y_val.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando y_val.npy ({y_val.shape})...")
        try:
            np.save(numpy_dir / "y_val.npy", y_val)
            print(f"    ✓ y_val.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar y_val.npy (puede estar en uso): {e}")
    
    if is_memmap_test and test_path.exists():
        print(f"    ⏭ X_test.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando X_test.npy ({X_test.shape})...")
        try:
            np.save(test_path, X_test)
            print(f"    ✓ X_test.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar X_test.npy (puede estar en uso): {e}")
    
    if is_memmap_test and (numpy_dir / "y_test.npy").exists():
        print(f"    ⏭ y_test.npy ya existe en disco, omitiendo guardado")
    else:
        print(f"    Guardando y_test.npy ({y_test.shape})...")
        try:
            np.save(numpy_dir / "y_test.npy", y_test)
            print(f"    ✓ y_test.npy guardado")
        except (OSError, PermissionError) as e:
            print(f"    ⚠ No se pudo guardar y_test.npy (puede estar en uso): {e}")
    
    # Guardar metadatos
    print(f"  Guardando metadatos...")
    print(f"    Guardando master_labels.csv...")
    metadata_train.to_csv(output_dir / "metadata" / "master_labels.csv", index=False)
    print(f"    ✓ master_labels.csv guardado")
    
    # Guardar metadata completa con splits
    print(f"    Guardando master_labels_full.csv...")
    metadata_full = pd.concat([
        metadata_train.assign(split="train"),
        metadata_val.assign(split="val"),
        metadata_test.assign(split="test"),
    ], ignore_index=True)
    metadata_full.to_csv(output_dir / "metadata" / "master_labels_full.csv", index=False)
    print(f"    ✓ master_labels_full.csv guardado")
    
    # Guardar folds
    print(f"  Guardando folds...")
    print(f"    Guardando folds_train_indices.npy...")
    np.save(output_dir / "metadata" / "folds_train_indices.npy", np.array(folds_train, dtype=object))
    print(f"    ✓ folds_train_indices.npy guardado")
    
    print(f"    Guardando folds_val_indices.npy...")
    np.save(output_dir / "metadata" / "folds_val_indices.npy", np.array(folds_val, dtype=object))
    print(f"    ✓ folds_val_indices.npy guardado")
    
    print(f"\n✓ Dataset guardado exitosamente en {output_dir}")
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Val: {len(X_val)} muestras")
    print(f"  Test: {len(X_test)} muestras")
    print(f"  Folds: {len(folds_train)} folds estratificados")

