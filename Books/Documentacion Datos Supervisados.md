# Documentación Completa: Pipeline de Dataset Supervisado ECG

## 📋 Resumen Ejecutivo

Este documento describe en detalle el **pipeline completo** para construir un dataset supervisado binario de ECG (NORMAL vs ANÓMALO) a partir de los datasets PTB-XL y MIMIC-IV-ECG. El pipeline procesa, etiqueta, filtra, normaliza, balancea y organiza los datos en splits de entrenamiento, validación y test con folds estratificados.

**Resultados del ejemplo completo:**
- **Total registros procesados:** 496,244 (979 PTB-XL + 495,265 MIMIC)
- **Después de balanceo:** 386,670 registros (50% normales, 50% anómalos)
- **Splits finales:**
  - Train: 270,668 (70%)
  - Val: 58,001 (15%)
  - Test: 58,001 (15%)
- **Shape final:** (386,670, 5000, 3) → 386,670 registros × 5000 muestras × 3 leads
- **Tiempo total:** ~90 minutos (procesamiento + balanceo + splits)

---

## 🔄 Flujo Completo del Pipeline

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CONFIGURACIÓN E IMPORTACIONES                            │
│    - Cargar módulos y configurar parámetros                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PROCESAR PTB-XL                                         │
│    - 21,799 registros → 979 válidos (4.5%)                 │
│    - Tiempo: 0.43 minutos                                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PROCESAR MIMIC-IV-ECG                                   │
│    - 800,035 registros → 495,265 válidos (61.9%)           │
│    - Tiempo: 54.07 minutos                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CONSTRUIR DATASET COMBINADO                             │
│    - Combinar PTB-XL + MIMIC                               │
│    - Total: 496,244 registros                              │
│    - Tiempo: 712.72 segundos                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. BALANCEAR DATASET                                       │
│    - Downsampling de clase mayoritaria                     │
│    - Resultado: 386,670 registros (50/50)                  │
│    - Tiempo: 1293.67 segundos                              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. CREAR SPLITS TRAIN/VAL/TEST                             │
│    - 70% train, 15% val, 15% test (estratificado)         │
│    - Guardado directo en disco (memoria eficiente)        │
│    - Tiempo: 773.80 segundos                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. CREAR FOLDS ESTRATIFICADOS                              │
│    - 10 folds para cross-validation                        │
│    - Tiempo: 0.21 segundos                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. GUARDAR DATASET                                         │
│    - Arrays numpy, metadatos, folds                        │
│    - Tiempo: 0.04 minutos                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Funciones Principales del Pipeline

Esta sección describe las funciones más importantes que se utilizan en cada paso del pipeline. Cada función está ubicada en los módulos `supervised_ecg_pipeline.py` o `supervised_ecg_pipeline_fast.py`.

### Funciones de Etiquetado

#### `label_ptbxl_record()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 323-389

**Propósito:** Etiqueta un registro de PTB-XL como NORMAL (0), ANÓMALO (1) o DESCARTAR (-1) basándose en códigos SCP.

**Parámetros principales:**
- `scp_codes`: Diccionario de códigos SCP o string evaluable
- `quality_columns`: Series con columnas de calidad (baseline_drift, etc.)
- `reject_unvalidated`: Si True, rechazar reportes no validados

**Parámetros usados en el pipeline:**
```python
label_ptbxl_record(
    scp_codes=row["scp_codes"],           # Códigos SCP parseados del CSV
    quality_columns=row[quality_cols],     # Columnas de calidad (baseline_drift, etc.)
    reject_unvalidated=REJECT_UNVALIDATED,  # False (no rechazar no validados)
    validated_by_human=row.get("validated_by_human"),  # Columna del CSV
    initial_autogenerated=row.get("initial_autogenerated_report"),  # Columna del CSV
)
```

**Lógica:**
1. Verifica calidad de señal (prioridad máxima) - rechaza si hay problemas de electrodos
2. Verifica códigos anómalos - retorna `1` si encuentra códigos patológicos
3. Verifica códigos normales - retorna `0` si encuentra "NORM" sin otros códigos patológicos
4. Por defecto, descarta si no coincide con nada claro

**Retorna:** Tupla `(label, razón)` donde `label` es 0, 1, o -1

---

#### `label_mimic_record()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 260-316

**Propósito:** Etiqueta un registro de MIMIC basándose en reportes de texto usando patrones regex.

**Parámetros principales:**
- `reports_series`: Series de pandas con reportes (report_1, report_2, etc.)

**Lógica:**
1. Concatena todos los reportes disponibles
2. Normaliza el texto (lowercase, elimina espacios)
3. Busca patrones anómalos usando regex - retorna `1` si encuentra alguno
4. Busca patrones normales - retorna `0` si encuentra "normal" sin anómalos
5. Por defecto, descarta si no coincide con nada claro

**Retorna:** Tupla `(label, razón)` donde `label` es 0, 1, o -1

---

### Funciones de Procesamiento de Señales

#### `process_single_record()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 796-859

**Propósito:** Procesa un registro completo: calidad → filtrado → resampleo → normalización.

**Parámetros principales:**
- `signal`: Señal cruda [T, 3]
- `original_fs`: Frecuencia de muestreo original (250 o 500 Hz)
- `apply_quality_check`: Si verificar calidad
- `apply_notch`: Si aplicar filtro notch
- `notch_freq`: Frecuencia del notch (50 o 60 Hz)
- `normalize_method`: "minmax" o "zscore"

**Parámetros usados en el pipeline:**
```python
process_single_record(
    signal=signal,                          # Señal cargada desde WFDB
    original_fs=500,                        # PTB-XL: 500 Hz, MIMIC: 250 Hz
    apply_quality_check=True,               # Verificar calidad
    apply_notch=True,                       # Aplicar filtro notch
    notch_freq=50.0,                        # 50 Hz (ruido de línea eléctrica)
    apply_bandpass=True,                    # Aplicar filtro pasa-banda (por defecto)
    normalize_method="minmax",              # Normalización min-max [0, 1]
    resample_strategy="center",             # Estrategia de recorte (ventana central)
)
```

**Proceso interno:**
1. **Verificar calidad:** Llama a `check_signal_quality()` - rechaza si la señal es mala
2. **Filtrar:** Llama a `filter_signal()` - aplica notch y bandpass
3. **Resamplear:** Llama a `resample_signal()` - ajusta a 500 Hz y 10 segundos
4. **Normalizar:** Llama a `normalize_signal_minmax()` o `normalize_signal_zscore()`

**Retorna:** Tupla `(señal_procesada, mensaje)` - Si hay error, retorna `(None, mensaje_de_error)`

---

#### `check_signal_quality()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 396-485

**Propósito:** Verifica la calidad de una señal ECG antes de procesarla.

**Parámetros principales:**
- `signal`: Array de forma [T, C] o [C, T]
- `fs`: Frecuencia de muestreo
- `min_std`: Desviación estándar mínima por lead
- `max_nan_ratio`: Proporción máxima de NaN permitida (5%)
- `max_flat_ratio`: Proporción máxima de valores constantes (10%)

**Parámetros usados en el pipeline (valores por defecto):**
```python
check_signal_quality(
    signal=signal,                         # Señal a verificar [T, 3]
    fs=500.0,                              # Frecuencia de muestreo (500 Hz objetivo)
    min_duration=10.0,                     # Duración mínima (10 segundos)
    min_std=0.001,                         # Desviación estándar mínima por lead
    max_nan_ratio=0.05,                    # Máximo 5% de NaN
    max_flat_ratio=0.10,                   # Máximo 10% de valores constantes
    saturation_threshold=0.95,             # Umbral de saturación (95% del rango)
)
```

**Checks realizados:**
1. Duración mínima (10 segundos)
2. Desviación estándar mínima por lead (0.001)
3. Proporción de NaN (máximo 5%)
4. Proporción de valores constantes (máximo 10%)
5. Detección de saturación (valores en extremos)

**Retorna:** Tupla `(es_valida, razon_rechazo)`

---

#### `filter_signal()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 545-578

**Propósito:** Aplica filtros de señal (notch y bandpass) para eliminar ruido.

**Parámetros principales:**
- `signal`: Señal [T, C]
- `fs`: Frecuencia de muestreo
- `apply_notch`: Si aplicar filtro notch
- `notch_freq`: Frecuencia notch (50 o 60 Hz)
- `apply_bandpass`: Si aplicar filtro pasa-banda

**Parámetros usados en el pipeline:**
```python
filter_signal(
    signal=signal,                         # Señal después de verificación de calidad
    fs=500.0,                              # Frecuencia de muestreo (500 Hz)
    apply_notch=True,                      # Aplicar filtro notch
    notch_freq=50.0,                       # 50 Hz (ruido de línea eléctrica)
    apply_bandpass=True,                   # Aplicar filtro pasa-banda
)
```

**Filtros aplicados:**
1. **Notch (50/60 Hz):** Elimina ruido de línea eléctrica usando `scipy.signal.iirnotch()`
   - Q-factor: 30.0
2. **Bandpass (0.5-40 Hz):** Elimina deriva de línea base y ruido de alta frecuencia usando `scipy.signal.butter()` + `filtfilt()`
   - Orden: 4 (Butterworth)
   - Frecuencias: 0.5-40 Hz

**Retorna:** Señal filtrada [T, C]

---

#### `resample_signal()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 655-738

**Propósito:** Resamplea la señal a 500 Hz y ajusta a exactamente 10 segundos (5000 muestras).

**Parámetros principales:**
- `signal`: Señal [T, C]
- `original_fs`: Frecuencia original (250 o 500 Hz)
- `target_fs`: Frecuencia objetivo (500 Hz)
- `target_duration`: Duración objetivo (10 segundos)
- `strategy`: "center", "start", o "random" (para recorte)

**Parámetros usados en el pipeline:**
```python
resample_signal(
    signal=filtered,                       # Señal después de filtrado
    original_fs=500.0,                     # PTB-XL: 500 Hz, MIMIC: 250 Hz
    target_fs=500.0,                       # Frecuencia objetivo (500 Hz)
    target_duration=10.0,                  # Duración objetivo (10 segundos)
    strategy="center",                     # Ventana central para recorte
)
```

**Proceso:**
1. Si la señal es > 10s, recorta usando ventana central
2. Si `original_fs ≠ target_fs`, resamplea usando `scipy.signal.resample()`
3. Ajusta a exactamente 5000 muestras (500 Hz × 10s)

**Retorna:** Señal resampleada [5000, C]

---

#### `normalize_signal_minmax()` / `normalize_signal_zscore()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 580-648 y 620-653

**Propósito:** Normaliza la señal para que esté en un rango estándar.

**Parámetros usados en el pipeline:**
```python
# Min-Max (método por defecto)
normalize_signal_minmax(
    signal=resampled,                      # Señal después de resampleo [5000, 3]
)
# Retorna: (normalized, mins, maxs)

# Z-Score (alternativa)
normalize_signal_zscore(
    signal=resampled,                      # Señal después de resampleo [5000, 3]
)
```

**Min-Max (por defecto):**
- Normaliza cada lead independientemente: `(signal - min) / (max - min)`
- Rango final: [0, 1]
- Si `max == min` (señal constante), retorna NaN (se rechaza)

**Z-Score:**
- Normaliza cada lead: `(signal - mean) / std`
- Media final: 0, Desviación estándar: 1

**Retorna:** Señal normalizada [T, C]

---

### Funciones de Procesamiento de Datasets

#### `process_ptbxl_dataset_fast()`
**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 190-313

**Propósito:** Procesa el dataset completo de PTB-XL en paralelo usando múltiples workers.

**Parámetros principales:**
- `overwrite`: Si sobrescribir archivos existentes
- `apply_quality_check`: Si verificar calidad
- `apply_notch`: Si aplicar filtro notch
- `normalize_method`: "minmax" o "zscore"
- `max_records`: Límite de registros (None = todos)
- `n_workers`: Número de workers paralelos (15 por defecto)

**Parámetros usados en el pipeline:**
```python
process_ptbxl_dataset_fast(
    overwrite=False,                                    # No sobrescribir archivos existentes
    apply_quality_check=APPLY_QUALITY_CHECK and not MINIMAL_QUALITY,  # True (si MINIMAL_QUALITY=False)
    apply_notch=APPLY_NOTCH,                           # True
    notch_freq=NOTCH_FREQ,                             # 50.0 Hz
    normalize_method=NORMALIZE_METHOD,                  # "minmax"
    reject_unvalidated=REJECT_UNVALIDATED,             # False
    max_records=MAX_PTB,                               # None (todos los registros)
    n_workers=N_WORKERS,                               # 15 (auto: cpu_count() - 1)
    verbose=True,                                      # Mostrar progreso
    prefilter_labels=False,                            # Procesar todos directamente
)
```

**Proceso:**
1. Lee `ptbxl_database.csv` y parsea códigos SCP
2. Prepara tareas para procesamiento paralelo
3. Usa `multiprocessing.Pool` con `_process_ptbxl_single()` como worker
4. Agrega resultados y construye arrays finales

**Retorna:** Tupla `(signals, labels, metadata_df)`
- `signals`: Array [N, 5000, 3]
- `labels`: Array [N] con 0 o 1
- `metadata_df`: DataFrame con metadatos

---

#### `process_mimic_dataset_fast()`
**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 316-458

**Propósito:** Procesa el dataset completo de MIMIC-IV-ECG en paralelo.

**Parámetros principales:** Similar a `process_ptbxl_dataset_fast()`

**Parámetros usados en el pipeline:**
```python
process_mimic_dataset_fast(
    overwrite=False,                                    # No sobrescribir archivos existentes
    apply_quality_check=APPLY_QUALITY_CHECK and not MINIMAL_QUALITY,  # True (si MINIMAL_QUALITY=False)
    apply_notch=APPLY_NOTCH,                           # True
    notch_freq=NOTCH_FREQ,                             # 50.0 Hz
    normalize_method=NORMALIZE_METHOD,                  # "minmax"
    max_records=MAX_MIMIC,                             # None (todos los registros)
    n_workers=N_WORKERS,                               # 15 (auto: cpu_count() - 1)
    verbose=True,                                      # Mostrar progreso
    prefilter_labels=False,                            # Procesar todos directamente
)
```

**Proceso:**
1. Lee `record_list.csv` y crea mapeo `(subject_id, study_id)` → `path`
2. Lee `machine_measurements.csv` en chunks de 10k registros
3. Filtra registros válidos (con path y reportes)
4. Usa `multiprocessing.Pool` con `_process_mimic_single()` como worker
5. Agrega resultados y construye arrays finales

**Retorna:** Tupla `(signals, labels, metadata_df)`

---

#### `_process_ptbxl_single()` / `_process_mimic_single()`
**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 34-110 y 113-183

**Propósito:** Funciones worker que procesan un solo registro. Ejecutadas en paralelo por cada worker.

**Parámetros internos (pasados por el pool):**
- `apply_quality_check`: `APPLY_QUALITY_CHECK and not MINIMAL_QUALITY` (True)
- `apply_notch`: `APPLY_NOTCH` (True)
- `notch_freq`: `NOTCH_FREQ` (50.0)
- `normalize_method`: `NORMALIZE_METHOD` ("minmax")
- `reject_unvalidated`: `REJECT_UNVALIDATED` (False) - solo para PTB-XL

**Proceso interno:**
1. Etiqueta el registro (`label_ptbxl_record()` o `label_mimic_record()`)
   - Para PTB-XL: `reject_unvalidated=False`
2. Carga la señal desde archivo WFDB
3. Extrae los leads necesarios (II, V1, V5)
4. Procesa la señal completa (`process_single_record()`)
   - `apply_quality_check=True`
   - `apply_notch=True`
   - `notch_freq=50.0`
   - `normalize_method="minmax"`
5. Retorna diccionario con señal, label y metadatos

**Retorna:** Diccionario con `{"signal": ..., "label": ..., "record_id": ..., ...}` o `None` si se rechaza

---

### Funciones de Balanceo y División

#### `balance_dataset()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 866-911

**Propósito:** Balancea el dataset haciendo downsampling estratificado de la clase mayoritaria.

**Parámetros principales:**
- `X`: Array de señales [N, T, C]
- `y`: Array de etiquetas [N]
- `random_state`: Seed para reproducibilidad (42 por defecto)
- `return_indices`: Si retornar también los índices seleccionados

**Parámetros usados en el pipeline:**
```python
balance_dataset(
    X=X,                                    # Dataset combinado (496,244 registros)
    y=y,                                    # Labels combinados
    random_state=42,                        # Seed para reproducibilidad
    return_indices=True,                     # Retornar índices para mapear metadata
)
```

**Algoritmo:**
1. Identifica la clase minoritaria (`n_min = counts.min()`)
2. Para cada clase:
   - Si tiene más registros que `n_min`, selecciona aleatoriamente `n_min`
   - Si tiene menos o igual, usa todos
3. Mezcla aleatoriamente los índices seleccionados
4. Extrae datos balanceados usando los índices

**Retorna:** Tupla `(X_balanced, y_balanced)` o `(X_balanced, y_balanced, indices)` si `return_indices=True`

---

#### `create_splits_to_disk()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 990-1145

**Propósito:** Crea splits train/val/test estratificados y los guarda directamente en disco (evita problemas de memoria).

**Parámetros principales:**
- `X`: Array de señales [N, T, C]
- `y`: Array de etiquetas [N]
- `output_dir`: Directorio de salida
- `train_ratio`: Proporción de train (0.70)
- `val_ratio`: Proporción de val (0.15)
- `test_ratio`: Proporción de test (0.15)
- `chunk_size`: Tamaño de chunk para guardado (10000)
- `random_state`: Seed para reproducibilidad

**Parámetros usados en el pipeline:**
```python
create_splits_to_disk(
    X=X_balanced,                           # Dataset balanceado (386,670 registros)
    y=y_balanced,                           # Labels balanceados
    output_dir=OUTPUT_DIR,                  # data/Datos_supervisados/
    train_ratio=0.70,                       # 70% entrenamiento
    val_ratio=0.15,                         # 15% validación
    test_ratio=0.15,                        # 15% test
    chunk_size=10000,                       # Procesar train en chunks de 10k
    random_state=42,                        # Seed para reproducibilidad
)
```

**Proceso:**
1. Calcula índices estratificados usando `StratifiedShuffleSplit`
2. Separa test del resto (15%)
3. Separa train y val del resto (70% / 15%)
4. Guarda test y val directamente en disco
5. Guarda train en chunks para evitar problemas de memoria
6. Carga arrays como memoria mapeada (`mmap_mode='r'`)

**Retorna:** Tupla con arrays y índices:
```python
(X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx)
```

---

#### `create_stratified_folds()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 1148-1170

**Propósito:** Crea folds estratificados para cross-validation.

**Parámetros principales:**
- `X`: Array de señales [N, T, C]
- `y`: Array de etiquetas [N]
- `n_splits`: Número de folds (10 por defecto)
- `random_state`: Seed para reproducibilidad

**Parámetros usados en el pipeline:**
```python
create_stratified_folds(
    X=X_train,                             # Dataset de entrenamiento (270,668 registros)
    y=y_train,                             # Labels de entrenamiento
    n_splits=10,                            # 10 folds para cross-validation
    random_state=42,                        # Seed para reproducibilidad
)
```

**Proceso:**
1. Usa `StratifiedKFold` de scikit-learn
2. Divide el dataset en `n_splits` folds manteniendo proporción de clases
3. Mezcla aleatoriamente antes de dividir
4. Retorna lista de tuplas `(train_idx, val_idx)` para cada fold

**Retorna:** Lista de tuplas `[(train_idx_1, val_idx_1), ..., (train_idx_n, val_idx_n)]`

---

### Funciones de Guardado

#### `save_dataset()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 1632-1766

**Propósito:** Guarda el dataset completo en la estructura especificada.

**Parámetros principales:**
- `X_train, y_train, X_val, y_val, X_test, y_test`: Arrays de datos
- `metadata_train, metadata_val, metadata_test`: DataFrames de metadatos
- `folds_train, folds_val`: Listas de índices para folds
- `output_dir`: Directorio de salida

**Parámetros usados en el pipeline:**
```python
save_dataset(
    X_train=X_train,                       # Train split (270,668 registros)
    y_train=y_train,                       # Train labels
    X_val=X_val,                           # Val split (58,001 registros)
    y_val=y_val,                           # Val labels
    X_test=X_test,                         # Test split (58,001 registros)
    y_test=y_test,                         # Test labels
    metadata_train=metadata_train,         # Metadata de train
    metadata_val=metadata_val,             # Metadata de val
    metadata_test=metadata_test,           # Metadata de test
    folds_train=folds_train,               # 10 arrays de índices train
    folds_val=folds_val,                   # 10 arrays de índices val
    output_dir=OUTPUT_DIR,                 # data/Datos_supervisados/
)
```

**Proceso:**
1. Verifica si los arrays ya están guardados (memoria mapeada)
2. Guarda arrays numpy si no existen
3. Guarda metadatos:
   - `master_labels.csv`: Solo train
   - `master_labels_full.csv`: Todos los splits con columna `split`
4. Guarda folds en formato NPZ

**Estructura creada:**
```
output_dir/
├── numpy/              # Arrays numpy
├── metadata/           # Metadatos y folds
└── raw_examples/       # Ejemplos visuales (opcional)
```

---

### Funciones Auxiliares

#### `ensure_dir()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 191-194

**Propósito:** Crea un directorio si no existe.

**Uso:** `ensure_dir(path)` - Crea el directorio y todos los padres necesarios.

---

#### `load_ptbxl_record()` / `load_mimic_record()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 739-793

**Propósito:** Carga un registro desde archivo WFDB.

**Proceso:**
1. Usa `wfdb.rdsamp()` para cargar señal y metadatos
2. Retorna señal [T, C] y metadatos (incluyendo frecuencia de muestreo)

**Retorna:** Tupla `(signal, metadata)`

---

#### `plot_ecg_comparison()`
**Ubicación:** `supervised_ecg_pipeline.py` líneas 1177-1257

**Propósito:** Genera visualizaciones de señales ECG para inspección.

**Parámetros principales:**
- `raw`: Señal cruda o normalizada [T, C]
- `filtered`: Señal filtrada (opcional)
- `normalized`: Señal normalizada (opcional)
- `save_path`: Ruta donde guardar la imagen PNG

**Retorna:** Guarda imagen PNG con las 3 derivaciones (II, V1, V5)

---

## 📝 Paso 1: Configuración e Importaciones

### 1.1. Importaciones Principales

```python
import sys
import time
from pathlib import Path
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Recargar módulo si es necesario (para incluir nuevas funciones)
import importlib
import supervised_ecg_pipeline
importlib.reload(supervised_ecg_pipeline)

# Importar módulos del pipeline
from supervised_ecg_pipeline import (
    OUTPUT_DIR,
    create_splits,
    create_splits_to_disk,
    create_stratified_folds,
    save_dataset,
    balance_dataset,
    ensure_dir,
    PTB_ROOT,
    MIMIC_ROOT,
)
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 53-77

### 1.2. Importación de Versión Optimizada

```python
# Intentar importar versión optimizada
try:
    from supervised_ecg_pipeline_fast import (
        process_mimic_dataset_fast,
        process_ptbxl_dataset_fast,
    )
    FAST_VERSION_AVAILABLE = True
    print("✓ Versión rápida (paralela) disponible")
except ImportError:
    from supervised_ecg_pipeline import (
        process_mimic_dataset,
        process_ptbxl_dataset,
    )
    FAST_VERSION_AVAILABLE = False
    print("⚠ Usando versión lenta (secuencial)")
```

**Ventaja:** El código detecta automáticamente si está disponible la versión paralela optimizada.

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 79-93

### 1.3. Parámetros de Configuración

```python
# ==================== CONFIGURACIÓN ====================
# Modifica estos valores según necesites

# Límites de registros (None = todos, o poner número para pruebas)
MAX_PTB = None      # None = todos, o poner número para pruebas (ej: 1000)
MAX_MIMIC = None    # None = todos, o poner número para pruebas (ej: 1000)

# Versión de procesamiento
USE_FAST = FAST_VERSION_AVAILABLE  # Usar versión paralela si está disponible
N_WORKERS = None                   # None = auto (cpu_count - 1)

# Calidad y filtrado
APPLY_QUALITY_CHECK = True    # Verificar calidad de señal
APPLY_NOTCH = True            # Aplicar filtro notch
NOTCH_FREQ = 50.0             # Frecuencia notch (50 o 60 Hz)
NORMALIZE_METHOD = "minmax"   # Método de normalización ("minmax" o "zscore")
REJECT_UNVALIDATED = False    # Rechazar reportes no validados

# Balanceo
DO_BALANCE = True             # Balancear el dataset

# Optimización
MINIMAL_QUALITY = False       # True = deshabilitar checks menos críticos para más velocidad
```

**Configuración automática de workers:**
```python
if N_WORKERS is None and USE_FAST:
    import multiprocessing as mp
    N_WORKERS = max(1, mp.cpu_count() - 1)  # Dejar 1 core libre
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 97-114

### 1.4. Directorio de Salida

```python
OUTPUT_DIR = DATA_DIR / "Datos_supervisados"
# Ejemplo: S:\Proyecto final\data\Datos_supervisados\
```

**Estructura que se creará:**
```
data/Datos_supervisados/
├── numpy/              # Arrays numpy con señales y labels
├── metadata/           # Metadatos y folds
└── raw_examples/       # Ejemplos visuales (opcional)
```

**Ubicación:** Definido en `supervised_ecg_pipeline.py` línea 47

---

## 📊 Paso 2: Procesar PTB-XL

### 2.1. Resumen del Procesamiento

**Input:**
- Total registros en PTB-XL: 21,799
- Archivo: `ptbxl_database.csv`
- Formato: Archivos WFDB (.dat, .hea) a 500 Hz

**Procesamiento:**
- Versión: RÁPIDA (paralela) con 15 workers
- Método: `process_ptbxl_dataset_fast()`

**Output:**
```
✓ PTB-XL COMPLETADO
  Tiempo: 0.43 minutos
  Registros: 979
  Normales: 397 (40.6%)
  Anómalos: 582 (59.4%)
  Shape: (979, 5000, 3)
  Memoria: 0.05 GB
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 135-243

### 2.2. Proceso Detallado

#### 2.2.1. Llamada a la Función

```python
if USE_FAST:
    ptbxl_signals, ptbxl_labels, ptbxl_df = process_ptbxl_dataset_fast(
        overwrite=False,
        apply_quality_check=APPLY_QUALITY_CHECK and not MINIMAL_QUALITY,
        apply_notch=APPLY_NOTCH,
        notch_freq=NOTCH_FREQ,
        normalize_method=NORMALIZE_METHOD,
        reject_unvalidated=REJECT_UNVALIDATED,
        max_records=MAX_PTB,
        n_workers=N_WORKERS,
        verbose=True,
        prefilter_labels=False,  # Procesar todos directamente
    )
```

**Parámetros clave:**
- `overwrite=False`: No sobrescribir archivos existentes
- `apply_quality_check`: Verificar calidad de señal (deshabilitado si `MINIMAL_QUALITY=True`)
- `prefilter_labels=False`: Procesar todos los registros directamente (más rápido)

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 190-313

#### 2.2.2. Procesamiento Paralelo Interno

**Función:** `_process_ptbxl_single()` ejecutada por cada worker

**Pasos internos:**

1. **Etiquetado:**
   ```python
   label, reason = label_ptbxl_record(
       scp_codes,
       quality_cols,
       reject_unvalidated=reject_unvalidated,
       validated_by_human=row.get("validated_by_human"),
       initial_autogenerated=row.get("initial_autogenerated_report"),
   )
   ```
   - Retorna `0` (normal), `1` (anómalo), o `-1` (rechazar)
   - **Ubicación:** `supervised_ecg_pipeline.py` líneas 323-389

2. **Carga de señal:**
   ```python
   record_path = PTB_ROOT / filename_hr
   signal, meta = wfdb.rdsamp(str(record_path), channels=None)
   ```
   - Carga archivo WFDB completo
   - **Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 69-72

3. **Extracción de leads:**
   ```python
   sig_names = get_sig_names(meta)
   lead_mapping = map_lead_names(sig_names, TARGET_LEADS)
   indices = [lead_mapping[lead] for lead in TARGET_LEADS]
   signal = signal[:, indices].astype(np.float32)
   ```
   - Extrae solo II, V1, V5
   - **Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 74-78

4. **Procesamiento completo:**
   ```python
   processed, proc_reason = process_single_record(
       signal,
       original_fs=500,
       apply_quality_check=apply_quality_check,
       apply_notch=apply_notch,
       notch_freq=notch_freq,
       normalize_method=normalize_method,
   )
   ```
   - Ejecuta: calidad → filtrado → resampleo → normalización
   - **Ubicación:** `supervised_ecg_pipeline.py` líneas 796-859

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 34-110

#### 2.2.3. Agregación de Resultados

Después del procesamiento paralelo:

```python
# Extraer señales
signals_list = [r["signal"] for r in processed_results]
signals = np.stack(signals_list, axis=0)
del signals_list  # Liberar memoria

# Extraer labels
labels = np.array([r["label"] for r in processed_results])

# Crear DataFrame sin señales (ahorrar memoria)
metadata_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "signal"}
    for r in processed_results
])
```

**Resultado:**
- `signals`: Array `(979, 5000, 3)` - 979 registros × 5000 muestras × 3 leads
- `labels`: Array `(979,)` - Etiquetas 0 (normal) o 1 (anómalo)
- `metadata_df`: DataFrame con información de cada registro

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 280-313

### 2.3. Estadísticas Detalladas

| Métrica | Valor | Porcentaje |
|---------|-------|------------|
| **Total procesados** | 979 | 4.5% |
| **Rechazados** | 20,820 | 95.5% |
| **Normales (label=0)** | 397 | 40.6% |
| **Anómalos (label=1)** | 582 | 59.4% |
| **Shape** | (979, 5000, 3) | - |
| **Memoria** | 0.05 GB | - |

**Ver documentación detallada:** `DOCUMENTACION_PROCESAMIENTO_PTBXL.md`

---

## 📊 Paso 3: Procesar MIMIC-IV-ECG

### 3.1. Resumen del Procesamiento

**Input:**
- Total registros en MIMIC: 800,035
- Archivos: `machine_measurements.csv`, `record_list.csv`
- Formato: Archivos WFDB (.dat, .hea) a 250 Hz (resampleados a 500 Hz)

**Procesamiento:**
- Versión: RÁPIDA (paralela) con 15 workers
- Método: `process_mimic_dataset_fast()`

**Output:**
```
✓ MIMIC COMPLETADO
  Tiempo: 54.07 minutos
  Registros: 495,265
  Normales: 192,938 (39.0%)
  Anómalos: 302,327 (61.0%)
  Shape: (495,265, 5000, 3)
  Memoria: 27.68 GB
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 255-366

### 3.2. Proceso Detallado

#### 3.2.1. Preparación de Datos

**Paso 1: Leer lista de registros**
```python
record_df = pd.read_csv(record_list_csv)
path_map = {
    (int(row.subject_id), int(row.study_id)): row.path
    for row in record_df.itertuples(index=False)
}
```

Crea un diccionario que mapea `(subject_id, study_id)` → `path` del archivo.

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 340-344

**Paso 2: Leer machine measurements en chunks**
```python
chunksize = 10000
valid_rows = []

for chunk in pd.read_csv(machine_csv, chunksize=chunksize):
    for idx, row in chunk.iterrows():
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])
        key = (subject_id, study_id)
        
        if key not in path_map:
            continue
        
        record_path_str = path_map[key]
        valid_rows.append((row.to_dict(), record_path_str))
```

Lee el CSV en chunks de 10,000 registros para evitar cargar todo en memoria.

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 350-372

#### 3.2.2. Procesamiento Paralelo

**Función:** `_process_mimic_single()` ejecutada por cada worker

**Pasos internos:**

1. **Búsqueda de path:**
   ```python
   record_path = MIMIC_ROOT / record_path_str
   ```
   - Construye la ruta completa al archivo

2. **Etiquetado basado en reportes:**
   ```python
   report_cols = {k: v for k, v in row_dict.items() if k.startswith("report_")}
   reports_series = pd.Series(report_cols)
   label, reason = label_mimic_record(reports_series)
   ```
   - Busca todas las columnas `report_1`, `report_2`, etc.
   - Concatena todos los reportes y los etiqueta
   - **Ubicación:** `supervised_ecg_pipeline.py` líneas 260-316

3. **Carga de señal:**
   ```python
   signal, meta = load_mimic_record(record_path)
   original_fs = meta.get("fs", 250)  # MIMIC típicamente 250 Hz
   ```
   - Carga archivo WFDB
   - **Ubicación:** `supervised_ecg_pipeline.py` líneas 771-793

4. **Procesamiento completo:**
   ```python
   processed, proc_reason = process_single_record(
       signal,
       original_fs,  # 250 Hz → se resamplea a 500 Hz
       apply_quality_check=apply_quality_check,
       apply_notch=apply_notch,
       notch_freq=notch_freq,
       normalize_method=normalize_method,
   )
   ```
   - Resamplea de 250 Hz → 500 Hz durante el procesamiento
   - **Ubicación:** `supervised_ecg_pipeline.py` líneas 796-859

**Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 113-183

#### 3.2.3. Diferencias con PTB-XL

**MIMIC tiene mayor tasa de aceptación (61.9% vs 4.5%) porque:**
1. **Reportes de texto más estructurados:** Los reportes de MIMIC son más consistentes
2. **Menos códigos ambiguos:** No usa códigos SCP complejos
3. **Mejor calidad general:** Señales de mejor calidad en promedio
4. **Patrones de etiquetado más flexibles:** Los patrones regex son más permisivos

**Procesamiento más lento (54 min vs 0.43 min) porque:**
1. **40× más registros:** 800k vs 22k
2. **Estructura de carpetas más compleja:** Archivos distribuidos en subdirectorios
3. **Resampleo necesario:** 250 Hz → 500 Hz requiere procesamiento adicional
4. **Lectura de CSV en chunks:** Overhead adicional de I/O

### 3.3. Estadísticas Detalladas

| Métrica | Valor | Porcentaje |
|---------|-------|------------|
| **Total procesados** | 495,265 | 61.9% |
| **Rechazados** | 304,770 | 38.1% |
| **Normales (label=0)** | 192,938 | 39.0% |
| **Anómalos (label=1)** | 302,327 | 61.0% |
| **Shape** | (495,265, 5000, 3) | - |
| **Memoria** | 27.68 GB | - |

---

## 📊 Paso 4: Construir Dataset Combinado

### 4.1. Proceso de Combinación

**Función:** Combinación manual en el notebook

**Paso 1: Preparar listas**
```python
all_signals = []
all_labels = []
all_metadata = []

if ptbxl_signals is not None and len(ptbxl_signals) > 0:
    print(f"  Agregando PTB-XL: {len(ptbxl_signals)} registros")
    all_signals.append(ptbxl_signals)
    all_labels.append(ptbxl_labels)
    if ptbxl_df is not None and len(ptbxl_df) > 0:
        all_metadata.append(ptbxl_df)

if mimic_signals is not None and len(mimic_signals) > 0:
    print(f"  Agregando MIMIC: {len(mimic_signals)} registros")
    all_signals.append(mimic_signals)
    all_labels.append(mimic_labels)
    if mimic_df is not None and len(mimic_df) > 0:
        all_metadata.append(mimic_df)
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 422-439

**Paso 2: Combinar arrays**
```python
print(f"  Combinando señales (esto puede tardar si hay muchas)...")
X = np.concatenate(all_signals, axis=0)
print(f"  ✓ Señales combinadas: {X.shape}")

print(f"  Combinando labels...")
y = np.concatenate(all_labels, axis=0)
print(f"  ✓ Labels combinados: {len(y)} labels")
```

**Operación `np.concatenate()`:**
- Combina arrays a lo largo del eje 0 (registros)
- Requiere que todos tengan la misma forma en los otros ejes: `(N, 5000, 3)`
- **Tiempo:** ~712 segundos para 496k registros
- **Memoria:** Crea una copia completa en memoria

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 441-447

**Paso 3: Combinar metadatos**
```python
if all_metadata:
    print(f"  Combinando metadatos ({len(all_metadata)} dataframes)...")
    metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"  ✓ Metadatos combinados: {len(metadata)} registros")
else:
    metadata = pd.DataFrame()
```

**Operación `pd.concat()`:**
- Combina DataFrames verticalmente
- `ignore_index=True`: Reinicia el índice desde 0
- Mantiene todas las columnas de ambos DataFrames

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 449-454

**Paso 4: Liberar memoria**
```python
# Liberar memoria de listas intermedias
del all_signals, all_labels, all_metadata
if ptbxl_signals is not None:
    del ptbxl_signals, ptbxl_labels
if mimic_signals is not None:
    del mimic_signals, mimic_labels
```

Libera las referencias a los arrays originales para liberar memoria.

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 456-461

### 4.2. Resultados

```
✓ Dataset combinado
  Tiempo: 712.72 segundos (11.88 minutos)
  Total registros: 496,244
  Normales: 193,335 (39.0%)
  Anómalos: 302,909 (61.0%)
  Shape: (496,244, 5000, 3)
  Memoria: 27.73 GB
```

### 4.3. Distribución por Fuente

| Fuente | Registros | Normales | Anómalos | % del Total |
|--------|-----------|----------|----------|-------------|
| **PTB-XL** | 979 | 397 | 582 | 0.2% |
| **MIMIC** | 495,265 | 192,938 | 302,327 | 99.8% |
| **TOTAL** | 496,244 | 193,335 | 302,909 | 100% |

**Nota:** MIMIC domina el dataset (99.8%) debido a su mayor tamaño y tasa de aceptación.

### 4.4. Detalles Técnicos de la Combinación

**Memoria requerida:**
- PTB-XL: 0.05 GB
- MIMIC: 27.68 GB
- **Total combinado:** 27.73 GB (ligeramente más por overhead de concatenación)

**Tiempo de concatenación:**
- `np.concatenate()` para 496k registros: ~712 segundos
- Operación I/O bound: limitada por velocidad de memoria RAM

---

## 📊 Paso 5: Balancear Dataset

### 5.1. Algoritmo de Balanceo

**Función:** `balance_dataset()`

**Ubicación:** `supervised_ecg_pipeline.py` líneas 866-911

#### 5.1.1. Proceso Paso a Paso

**Paso 1: Identificar clase minoritaria**
```python
unique_labels, counts = np.unique(y, return_counts=True)
n_min = counts.min()  # Tamaño de la clase más pequeña
```

**Ejemplo:**
- Normales: 193,335
- Anómalos: 302,909
- `n_min = 193,335` (clase minoritaria: normales)

**Paso 2: Seleccionar registros por clase**
```python
np.random.seed(random_state)

balanced_indices = []
for label in unique_labels:
    indices = np.where(y == label)[0]  # Índices de esta clase
    if len(indices) > n_min:
        # Downsampling aleatorio: seleccionar n_min registros
        selected = np.random.choice(indices, size=n_min, replace=False)
        balanced_indices.extend(selected.tolist())
    else:
        # Usar todos si la clase es menor o igual
        balanced_indices.extend(indices.tolist())
```

**Algoritmo:**
- Para cada clase, si tiene más registros que `n_min`, selecciona aleatoriamente `n_min`
- Si tiene menos o igual, usa todos
- **Resultado:** Todas las clases tienen exactamente `n_min` registros

**Paso 3: Mezclar aleatoriamente**
```python
balanced_indices = np.array(balanced_indices)
np.random.seed(random_state)
np.random.shuffle(balanced_indices)  # Mezclar para evitar orden por clase
```

**Paso 4: Extraer datos balanceados**
```python
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]
```

### 5.2. Resultados

```
✓ Dataset balanceado
  Tiempo: 1293.67 segundos (21.56 minutos)
  
  Antes: 496,244 registros
    Normales: 193,335 (39.0%)
    Anómalos: 302,909 (61.0%)
  
  Después: 386,670 registros
    Normales: 193,335 (50.0%)
    Anómalos: 193,335 (50.0%)
```

### 5.3. Estadísticas Detalladas

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| **Total registros** | 496,244 | 386,670 | -109,574 (-22.1%) |
| **Normales** | 193,335 | 193,335 | 0 (mantenidos) |
| **Anómalos** | 302,909 | 193,335 | -109,574 (-36.2%) |
| **Balance** | 39% / 61% | 50% / 50% | ✓ Balanceado |

**Registros eliminados:** 109,574 anómalos (seleccionados aleatoriamente con `random_state=42`)

### 5.4. Detalles Técnicos

**Tiempo de ejecución:**
- Indexación de arrays grandes: `X[balanced_indices]` crea una copia
- Para 496k registros × 5000 muestras × 3 leads: ~27 GB
- **Tiempo:** ~1293 segundos (21.56 minutos) - operación limitada por I/O de memoria

**Memoria:**
- Crea una copia completa del dataset balanceado
- Memoria pico: ~55 GB (dataset original + balanceado simultáneamente)
- Después de balancear, se libera el dataset original

**Reproducibilidad:**
- Usa `random_state=42` para selección aleatoria
- Mismo `random_state` → mismos registros seleccionados

---

## 📊 Paso 6: Crear Splits Train/Val/Test

### 6.1. Método: División Estratificada con Guardado en Disco

**Función:** `create_splits_to_disk()`

**Ventaja:** Guarda directamente en disco para evitar problemas de memoria con datasets grandes.

**Ubicación:** `supervised_ecg_pipeline.py` líneas 990-1145

### 6.2. Proceso Detallado

#### 6.2.1. Preparación de Archivos

```python
ensure_dir(output_dir / "numpy")

# Definir rutas de archivos
test_path = output_dir / "numpy" / "X_test.npy"
y_test_path = output_dir / "numpy" / "y_test.npy"
val_path = output_dir / "numpy" / "X_val.npy"
y_val_path = output_dir / "numpy" / "y_val.npy"
train_path = output_dir / "numpy" / "X_train.npy"
y_train_path = output_dir / "numpy" / "y_train.npy"
```

**Limpieza de archivos existentes:**
```python
import os
import gc
for path in [test_path, y_test_path, val_path, y_val_path, train_path, y_train_path]:
    if path.exists():
        try:
            path.unlink()  # Eliminar archivo
        except (PermissionError, OSError):
            # Si está en uso, forzar garbage collection y reintentar
            gc.collect()
            try:
                path.unlink()
            except (PermissionError, OSError):
                # Si aún falla, renombrar a .tmp
                temp_path = path.with_suffix('.tmp')
                if temp_path.exists():
                    temp_path.unlink()
                path.rename(temp_path)
```

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1019-1052

#### 6.2.2. Cálculo de Índices Estratificados

**Paso 1: Separar test del resto**
```python
sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=test_ratio,  # 0.15 (15%)
    random_state=random_state
)
train_val_idx, test_idx = next(sss.split(X, y))
```

**Resultado:**
- `test_idx`: Índices para test (~15% = 58,001 registros)
- `train_val_idx`: Índices para train+val (~85% = 328,669 registros)

**Paso 2: Separar train y val**
```python
val_size = val_ratio / (train_ratio + val_ratio)  # 0.15 / 0.85 = 0.176
sss2 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=val_size,  # 0.176 (17.6% de train_val = 15% del total)
    random_state=random_state
)
y_temp = y[train_val_idx]  # Solo necesitamos y para estratificar
train_local_idx, val_local_idx = next(sss2.split(np.arange(len(train_val_idx)), y_temp))
```

**Mapeo de índices locales a globales:**
```python
train_idx = train_val_idx[train_local_idx]  # Índices globales para train
val_idx = train_val_idx[val_local_idx]      # Índices globales para val
```

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1054-1075

#### 6.2.3. Guardado en Disco

**Test split (pequeño, ~15%):**
```python
print(f"    Guardando test split ({len(test_idx)} registros)...")
X_test_data = X[test_idx]      # Extraer datos
y_test_data = y[test_idx]
np.save(str(test_path), X_test_data)      # Guardar en disco
np.save(str(y_test_path), y_test_data)
test_idx_copy = test_idx.copy()            # Guardar índices
del X_test_data, y_test_data, test_idx    # Liberar memoria
```

**Val split (pequeño, ~15%):**
```python
print(f"    Guardando val split ({len(val_idx)} registros)...")
X_val_data = X[val_idx]
y_val_data = y[val_idx]
np.save(str(val_path), X_val_data)
np.save(str(y_val_path), y_val_data)
val_idx_copy = val_idx.copy()
del X_val_data, y_val_data, val_idx
```

**Train split (grande, ~70%, procesado en chunks):**
```python
print(f"    Guardando train split ({len(train_idx)} registros) en chunks...")

train_chunks_X = []
train_chunks_y = []

# Procesar en chunks de 10k registros
for i in range(0, len(train_idx), chunk_size):
    end_idx = min(i + chunk_size, len(train_idx))
    chunk_idx = train_idx[i:end_idx]
    train_chunks_X.append(X[chunk_idx].copy())  # Copiar para liberar referencia
    train_chunks_y.append(y[chunk_idx].copy())

# Concatenar chunks
X_train_data = np.concatenate(train_chunks_X, axis=0)
y_train_data = np.concatenate(train_chunks_y, axis=0)

# Guardar en disco
np.save(str(train_path), X_train_data)
np.save(str(y_train_path), y_train_data)

train_idx_copy = train_idx.copy()
del train_chunks_X, train_chunks_y, X_train_data, y_train_data, train_idx
```

**Ventaja del procesamiento en chunks:**
- Evita cargar todo el train split en memoria de una vez
- Procesa en chunks de 10k registros
- Reduce memoria pico

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1077-1121

#### 6.2.4. Guardado de Índices

```python
indices_path = output_dir / "numpy" / "split_indices.npz"
np.savez(
    str(indices_path),
    train_idx=train_idx_copy,
    val_idx=val_idx_copy,
    test_idx=test_idx_copy
)
```

Guarda los índices en formato NPZ para poder mapear metadata correctamente.

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1123-1130

#### 6.2.5. Carga como Memoria Mapeada

```python
print(f"    Cargando arrays en memoria (solo para compatibilidad)...")
X_train = np.load(train_path, mmap_mode='r')  # Solo lectura, no carga en RAM
y_train = np.load(y_train_path, mmap_mode='r')
X_val = np.load(val_path, mmap_mode='r')
y_val = np.load(y_val_path, mmap_mode='r')
X_test = np.load(test_path, mmap_mode='r')
y_test = np.load(y_test_path, mmap_mode='r')
```

**Memoria mapeada (`mmap_mode='r'`):**
- No carga los datos en RAM
- Accede directamente al archivo en disco cuando se necesita
- **Ventaja:** Permite trabajar con datasets grandes sin cargar todo en memoria

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1132-1140

### 6.3. Resultados

```
✓ Splits creados y guardados (tiempo: 773.80 segundos = 12.90 minutos)

  Train: 270,668 (70.0%)
    Normales: 135,334
    Anómalos: 135,334
    Guardado en: X_train.npy (15.12 GB)
  
  Val: 58,001 (15.0%)
    Normales: 29,000
    Anómalos: 29,001
    Guardado en: X_val.npy (3.24 GB)
  
  Test: 58,001 (15.0%)
    Normales: 29,001
    Anómalos: 29,000
    Guardado en: X_test.npy (3.24 GB)
```

### 6.4. Estadísticas Detalladas

| Split | Registros | % Total | Normales | Anómalos | Memoria | Archivo |
|-------|-----------|---------|----------|----------|---------|---------|
| **Train** | 270,668 | 70.0% | 135,334 | 135,334 | 15.12 GB | `X_train.npy` |
| **Val** | 58,001 | 15.0% | 29,000 | 29,001 | 3.24 GB | `X_val.npy` |
| **Test** | 58,001 | 15.0% | 29,001 | 29,000 | 3.24 GB | `X_test.npy` |
| **TOTAL** | 386,670 | 100% | 193,335 | 193,335 | 21.60 GB | - |

**Nota:** Los arrays están guardados como memoria mapeada (`mmap_mode='r'`) para acceso eficiente sin cargar todo en RAM.

### 6.5. Mapeo de Metadata

Después de crear los splits, se mapea la metadata usando los índices:

```python
metadata_train = metadata.iloc[train_idx].copy().reset_index(drop=True)
metadata_val = metadata.iloc[val_idx].copy().reset_index(drop=True)
metadata_test = metadata.iloc[test_idx].copy().reset_index(drop=True)
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 763-767

---

## 📊 Paso 7: Crear Folds Estratificados

### 7.1. Proceso

**Función:** `create_stratified_folds()`

**Método:** Cross-validation estratificada con `StratifiedKFold`

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1148-1170

#### 7.1.1. Algoritmo

```python
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
folds = []
for train_idx, val_idx in skf.split(X, y):
    folds.append((train_idx, val_idx))
```

**`StratifiedKFold`:**
- Divide el dataset en `n_splits` folds
- **Estratificado:** Mantiene la proporción de clases en cada fold
- **Shuffle:** Mezcla aleatoriamente antes de dividir
- **Reproducible:** Usa `random_state=42`

**Proceso interno:**
1. Mezcla los datos aleatoriamente (manteniendo proporción de clases)
2. Divide en 10 partes iguales
3. Para cada fold:
   - Train: 9 partes (90%)
   - Val: 1 parte (10%)

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1166-1169

#### 7.1.2. Extracción de Folds

```python
folds_train = [fold[0] for fold in folds]  # Lista de arrays de índices train
folds_val = [fold[1] for fold in folds]    # Lista de arrays de índices val
```

**Estructura:**
- `folds_train`: Lista de 10 arrays, cada uno con índices de train para ese fold
- `folds_val`: Lista de 10 arrays, cada uno con índices de val para ese fold

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 841-842

### 7.2. Resultados

```
✓ 10 folds creados (tiempo: 0.21 segundos)

  Fold 1: Train=243,601, Val=27,067
  Fold 2: Train=243,601, Val=27,067
  Fold 3: Train=243,601, Val=27,067
  ... (10 folds total)
```

### 7.3. Estadísticas por Fold

Cada fold mantiene la proporción 50/50:

| Fold | Train | Val | Train Normales | Train Anómalos | Val Normales | Val Anómalos |
|------|-------|-----|----------------|----------------|--------------|--------------|
| 1-10 | 243,601 | 27,067 | ~121,800 | ~121,800 | ~13,533 | ~13,534 |

**Cálculo:**
- Train: 270,668 × 0.9 = 243,601 registros
- Val: 270,668 × 0.1 = 27,067 registros
- Cada fold mantiene ~50% normales y ~50% anómalos

**Archivos generados:**
- `metadata/folds_train_indices.npy` (18.59 MB) - Array de objetos con 10 arrays
- `metadata/folds_val_indices.npy` (2.07 MB) - Array de objetos con 10 arrays

---

## 📊 Paso 8: Guardar Dataset

### 8.1. Función Principal

**Función:** `save_dataset()`

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1632-1766

### 8.2. Proceso Detallado

#### 8.2.1. Verificación de Archivos Existentes

```python
# Verificar si los arrays son memoria mapeada o si los archivos ya existen
is_memmap_train = isinstance(X_train, np.memmap) or train_path.exists()
is_memmap_val = isinstance(X_val, np.memmap) or val_path.exists()
is_memmap_test = isinstance(X_test, np.memmap) or test_path.exists()

if is_memmap_train and train_path.exists():
    print(f"    ⏭ X_train.npy ya existe en disco, omitiendo guardado")
else:
    print(f"    Guardando X_train.npy ({X_train.shape})...")
    np.save(train_path, X_train)
```

**Lógica:**
- Si el array es memoria mapeada (`np.memmap`), ya está en disco
- Si el archivo existe, no lo sobrescribe
- Solo guarda si es necesario

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1671-1734

#### 8.2.2. Guardado de Metadatos

**Master labels (solo train):**
```python
metadata_train.to_csv(output_dir / "metadata" / "master_labels.csv", index=False)
```

**Master labels full (todos los splits):**
```python
metadata_full = pd.concat([
    metadata_train.assign(split="train"),
    metadata_val.assign(split="val"),
    metadata_test.assign(split="test"),
], ignore_index=True)
metadata_full.to_csv(output_dir / "metadata" / "master_labels_full.csv", index=False)
```

**Columnas en metadata:**
- `record_id`: ID único del registro (ej: `ptbxl_12345` o `mimic_12345_67890`)
- `source`: Fuente del dato (`PTB-XL` o `MIMIC`)
- `label`: Etiqueta (0=normal, 1=anómalo)
- `label_reason`: Razón de la etiqueta (ej: `"anómalo: IMI"`)
- `split`: A qué split pertenece (`train`, `val`, `test`)

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1736-1750

#### 8.2.3. Guardado de Folds

```python
np.save(
    output_dir / "metadata" / "folds_train_indices.npy",
    np.array(folds_train, dtype=object)  # dtype=object permite arrays de diferentes tamaños
)
np.save(
    output_dir / "metadata" / "folds_val_indices.npy",
    np.array(folds_val, dtype=object)
)
```

**Formato:**
- Array de objetos (dtype=object) que contiene 10 arrays numpy
- Cada array interno contiene los índices para ese fold

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1752-1759

### 8.3. Estructura de Archivos Final

```
data/Datos_supervisados/
├── numpy/
│   ├── X_train.npy              (15,487.75 MB)
│   ├── y_train.npy              (2.07 MB)
│   ├── X_val.npy                (3,318.84 MB)
│   ├── y_val.npy                (0.44 MB)
│   ├── X_test.npy               (3,318.84 MB)
│   ├── y_test.npy               (0.44 MB)
│   └── split_indices.npz        (índices de splits)
├── metadata/
│   ├── master_labels.csv        (23.29 MB) - Solo train
│   ├── master_labels_full.csv   (35.31 MB) - Todos los splits
│   ├── folds_train_indices.npy  (18.59 MB)
│   └── folds_val_indices.npy    (2.07 MB)
└── raw_examples/
    └── (ejemplos visuales PNG)
```

**Tamaño total:** ~22.5 GB

### 8.4. Tiempo de Guardado

```
✓ Dataset guardado exitosamente
  Tiempo: 0.04 minutos
  Ubicación: S:\Proyecto final\data\Datos_supervisados
```

**Nota:** Los arrays numpy ya estaban guardados en el paso 6, por lo que este paso solo guarda metadatos y folds (muy rápido).

---

## 📊 Paso 9: Verificar Datos Guardados

### 9.1. Proceso de Verificación

```python
files_to_check = [
    "numpy/X_train.npy",
    "numpy/y_train.npy",
    "numpy/X_val.npy",
    "numpy/y_val.npy",
    "numpy/X_test.npy",
    "numpy/y_test.npy",
    "metadata/master_labels.csv",
    "metadata/master_labels_full.csv",
    "metadata/folds_train_indices.npy",
    "metadata/folds_val_indices.npy",
]

for file_path in files_to_check:
    full_path = OUTPUT_DIR / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / 1024**2
        print(f"  ✓ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {file_path} - NO ENCONTRADO")
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 990-1024

### 9.2. Resultados

```
✓ Todos los archivos se guardaron correctamente

  ✓ numpy/X_train.npy (15487.75 MB)
  ✓ numpy/y_train.npy (2.07 MB)
  ✓ numpy/X_val.npy (3318.84 MB)
  ✓ numpy/y_val.npy (0.44 MB)
  ✓ numpy/X_test.npy (3318.84 MB)
  ✓ numpy/y_test.npy (0.44 MB)
  ✓ metadata/master_labels.csv (23.29 MB)
  ✓ metadata/master_labels_full.csv (35.31 MB)
  ✓ metadata/folds_train_indices.npy (18.59 MB)
  ✓ metadata/folds_val_indices.npy (2.07 MB)
```

---

## 📊 Paso 10: Visualizar Ejemplos (Opcional)

### 10.1. Generación de Ejemplos

**Función:** `plot_ecg_comparison()`

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1177-1257

#### 10.1.1. Proceso

```python
from supervised_ecg_pipeline import plot_ecg_comparison

n_examples = 3
ensure_dir(OUTPUT_DIR / "raw_examples")

# Seleccionar índices aleatorios
indices = np.random.choice(len(X_test), size=min(n_examples, len(X_test)), replace=False)

for i, idx in enumerate(indices):
    signal = X_test[idx]      # Extraer señal [5000, 3]
    label = y_test[idx]       # Extraer etiqueta
    record_id = metadata_test.iloc[idx]["record_id"]
    
    title = f"{record_id} - {'NORMAL' if label == 0 else 'ANÓMALO'}"
    save_path = OUTPUT_DIR / "raw_examples" / f"example_{i+1:03d}_{str(record_id).replace('/', '_')}.png"
    
    plot_ecg_comparison(
        raw=signal,           # Señal normalizada (ya procesada)
        filtered=None,         # No mostrar filtrada
        normalized=signal,     # Mostrar normalizada
        title=title,
        save_path=save_path,
    )
```

**Ubicación:** `build_supervised_ecg_dataset.ipynb` líneas 1053-1082

#### 10.1.2. Función de Visualización

```python
def plot_ecg_comparison(
    raw: np.ndarray,          # [T, C] - Señal cruda o normalizada
    filtered: Optional[np.ndarray],
    normalized: Optional[np.ndarray],
    fs: float = 500.0,
    lead_names: Sequence[str] = ["II", "V1", "V5"],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
```

**Proceso interno:**
1. Crea figura con subplots: `n_plots` filas × `n_leads` columnas
2. Para cada lead, grafica la señal en el tiempo
3. Guarda como PNG con resolución 150 DPI

**Ubicación:** `supervised_ecg_pipeline.py` líneas 1177-1257

### 10.2. Resultados

```
✓ 3 ejemplos visuales creados
  Ubicación: S:\Proyecto final\data\Datos_supervisados\raw_examples\
  
  Ejemplos:
    - example_001_mimic_13744125_41662601.png
    - example_002_mimic_18832132_48870396.png
    - example_003_mimic_12182445_49330246.png
```

Cada imagen muestra las 3 derivaciones (II, V1, V5) normalizadas en el rango [0, 1].

---

## 📈 Resumen Final del Pipeline

### Estadísticas Totales

| Métrica | Valor |
|---------|-------|
| **Registros PTB-XL procesados** | 979 (4.5% de 21,799) |
| **Registros MIMIC procesados** | 495,265 (61.9% de 800,035) |
| **Total combinado** | 496,244 |
| **Después de balanceo** | 386,670 (50/50) |
| **Train** | 270,668 (70%) |
| **Val** | 58,001 (15%) |
| **Test** | 58,001 (15%) |
| **Folds** | 10 folds estratificados |
| **Shape final** | (386,670, 5000, 3) |
| **Memoria total** | ~22.5 GB en disco |

### Tiempos de Ejecución

| Paso | Tiempo | % del Total | Operación Principal |
|------|--------|------------|---------------------|
| **1. Configuración** | < 1 s | < 0.1% | Importaciones |
| **2. Procesar PTB-XL** | 0.43 min | 0.5% | Procesamiento paralelo (15 workers) |
| **3. Procesar MIMIC** | 54.07 min | 60.0% | Procesamiento paralelo (15 workers) |
| **4. Combinar datasets** | 11.88 min | 13.2% | `np.concatenate()` |
| **5. Balancear** | 21.56 min | 23.9% | Indexación y copia de arrays |
| **6. Crear splits** | 12.90 min | 14.3% | División estratificada + guardado en disco |
| **7. Crear folds** | 0.21 s | < 0.1% | `StratifiedKFold` |
| **8. Guardar dataset** | 0.04 min | < 0.1% | Guardado de metadatos y folds |
| **TOTAL** | ~90 minutos | 100% | - |

### Características del Dataset Final

- **Formato:** NumPy arrays (float32)
- **Estructura:** (N, T, C) donde:
  - N = número de registros (386,670)
  - T = 5000 muestras (10 segundos)
  - C = 3 leads (II, V1, V5)
- **Frecuencia de muestreo:** 500 Hz
- **Duración:** 10 segundos por registro
- **Normalización:** Min-Max [0, 1]
- **Balance:** 50% normales, 50% anómalos
- **Splits:** Estratificados (mantienen proporción 50/50)
- **Folds:** 10 folds estratificados para cross-validation

---

## 🔧 Parámetros Configurables

### Parámetros de Procesamiento

```python
# Límites
MAX_PTB = None          # Límite de registros PTB-XL (None = todos)
MAX_MIMIC = None        # Límite de registros MIMIC (None = todos)

# Calidad
APPLY_QUALITY_CHECK = True   # Verificar calidad de señal
MINIMAL_QUALITY = False      # Calidad mínima (más rápido, menos checks)

# Filtrado
APPLY_NOTCH = True           # Aplicar filtro notch
NOTCH_FREQ = 50.0           # Frecuencia notch (50 o 60 Hz)

# Normalización
NORMALIZE_METHOD = "minmax"  # "minmax" o "zscore"

# Validación
REJECT_UNVALIDATED = False   # Rechazar reportes no validados

# Balanceo
DO_BALANCE = True            # Balancear dataset

# Paralelización
USE_FAST = True              # Versión paralela
N_WORKERS = None             # Auto (cpu_count() - 1)
```

### Parámetros de Splits

```python
train_ratio = 0.70    # 70% entrenamiento
val_ratio = 0.15      # 15% validación
test_ratio = 0.15     # 15% test
random_state = 42     # Reproducibilidad
chunk_size = 10000    # Tamaño de chunk para guardado (registros)
```

### Parámetros de Folds

```python
n_splits = 10         # 10 folds
random_state = 42     # Reproducibilidad
```

---

## 🔍 Detalles Técnicos

### Estructura de Datos Final

**Array de señales (`X_train`, `X_val`, `X_test`):**
- **Shape:** `(N, 5000, 3)`
  - `N`: Número de registros (varía por split)
  - `5000`: Muestras temporales (500 Hz × 10 s)
  - `3`: Canales/leads (II, V1, V5)
- **Dtype:** `float32`
- **Rango:** [0, 1] (normalización minmax)
- **Memoria mapeada:** Cargados con `mmap_mode='r'` para acceso eficiente

**Array de etiquetas (`y_train`, `y_val`, `y_test`):**
- **Shape:** `(N,)`
- **Dtype:** `int64`
- **Valores:** `0` (normal) o `1` (anómalo)

**DataFrame de metadatos:**
- Columnas: `record_id`, `source`, `label`, `label_reason`, `split`, etc.
- **Tamaño:** N filas (una por registro)

### Optimizaciones Implementadas

1. **Procesamiento paralelo:** Usa `multiprocessing.Pool` con 15 workers
2. **Guardado en disco:** `create_splits_to_disk()` evita problemas de memoria
3. **Memoria mapeada:** Arrays cargados con `mmap_mode='r'` para acceso eficiente
4. **Procesamiento en chunks:** Train split procesado en chunks de 10k registros
5. **Liberación de memoria:** Elimina referencias a arrays grandes después de usar

---

## 🐛 Troubleshooting

### Problema: Error de Memoria al Crear Splits

**Síntoma:**
```
MemoryError: Unable to allocate 3.24 GiB for an array with shape (58001, 5000, 3)
```

**Causa:**
- `create_splits()` intenta cargar todos los splits en memoria simultáneamente
- Con datasets grandes (>300k registros), esto excede la RAM disponible

**Solución:**
- Usar `create_splits_to_disk()` en lugar de `create_splits()`
- Esta función guarda directamente en disco sin cargar todo en RAM

**Código:**
```python
from supervised_ecg_pipeline import create_splits_to_disk

result = create_splits_to_disk(
    X, y,
    output_dir=OUTPUT_DIR,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    chunk_size=10000,  # Procesar en chunks de 10k
    random_state=42,
)

X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = result
```

**Ubicación:** `supervised_ecg_pipeline.py` líneas 990-1145

### Problema: Procesamiento Muy Lento

**Causas:**
1. Demasiados registros para procesar
2. Workers consumen mucha CPU/RAM
3. Checks de calidad muy estrictos

**Soluciones:**
1. **Reducir registros para pruebas:**
   ```python
   MAX_PTB = 1000      # Procesar solo 1000 registros de PTB-XL
   MAX_MIMIC = 10000   # Procesar solo 10k registros de MIMIC
   ```

2. **Usar calidad mínima:**
   ```python
   MINIMAL_QUALITY = True  # Deshabilitar checks menos críticos
   ```

3. **Reducir workers:**
   ```python
   N_WORKERS = 8  # Usar menos workers si hay problemas de memoria
   ```

4. **Deshabilitar checks de calidad:**
   ```python
   APPLY_QUALITY_CHECK = False  # Más rápido, pero menos calidad
   ```

### Problema: Muy Pocos Registros Procesados

**Causas:**
1. Criterios de etiquetado muy estrictos
2. Filtros de calidad muy restrictivos
3. `REJECT_UNVALIDATED = True` rechaza muchos registros

**Soluciones:**
- Ajustar `REJECT_UNVALIDATED = False`
- Relajar `APPLY_QUALITY_CHECK` o usar `MINIMAL_QUALITY = True`
- Revisar patrones de etiquetado en:
  - `PTB_ANOMALY_CODES` y `PTB_NORMAL_CODES` (PTB-XL)
  - `MIMIC_ANOMALY_PATTERNS` y `MIMIC_NORMAL_PATTERNS` (MIMIC)

**Ubicación:** `supervised_ecg_pipeline.py` líneas 92-175

### Problema: Archivos en Uso al Guardar

**Síntoma:**
```
PermissionError: [WinError 32] El proceso no puede obtener acceso al archivo porque está siendo utilizado por otro proceso
```

**Causa:**
- Archivo `.npy` está abierto en otro proceso (Jupyter, Python, etc.)

**Solución:**
1. Cerrar todas las referencias al array
2. Forzar garbage collection:
   ```python
   import gc
   del X_train, y_train  # Eliminar referencias
   gc.collect()          # Forzar liberación
   ```
3. Reiniciar el kernel de Jupyter si es necesario

---

## 📚 Referencias

### Archivos Principales

- **`build_supervised_ecg_dataset.ipynb`**: Notebook principal (este documento)
- **`build_supervised_ecg_dataset.py`**: Script equivalente
- **`supervised_ecg_pipeline.py`**: Funciones base de procesamiento
- **`supervised_ecg_pipeline_fast.py`**: Versión paralela optimizada
- **`DOCUMENTACION_PROCESAMIENTO_PTBXL.md`**: Documentación detallada de PTB-XL

### Funciones Clave

- `process_ptbxl_dataset_fast()`: Procesar PTB-XL en paralelo
  - **Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 190-313
- `process_mimic_dataset_fast()`: Procesar MIMIC en paralelo
  - **Ubicación:** `supervised_ecg_pipeline_fast.py` líneas 316-458
- `balance_dataset()`: Balancear dataset
  - **Ubicación:** `supervised_ecg_pipeline.py` líneas 866-911
- `create_splits_to_disk()`: Crear splits guardando en disco
  - **Ubicación:** `supervised_ecg_pipeline.py` líneas 990-1145
- `create_stratified_folds()`: Crear folds estratificados
  - **Ubicación:** `supervised_ecg_pipeline.py` líneas 1148-1170
- `save_dataset()`: Guardar dataset completo
  - **Ubicación:** `supervised_ecg_pipeline.py` líneas 1632-1766

---

## 📝 Ejemplo de Uso Completo

```python
# 1. Configuración
from supervised_ecg_pipeline_fast import (
    process_ptbxl_dataset_fast,
    process_mimic_dataset_fast,
)
from supervised_ecg_pipeline import (
    balance_dataset,
    create_splits_to_disk,
    create_stratified_folds,
    save_dataset,
    OUTPUT_DIR,
)

# 2. Procesar PTB-XL
ptbxl_signals, ptbxl_labels, ptbxl_df = process_ptbxl_dataset_fast(
    overwrite=False,
    apply_quality_check=True,
    apply_notch=True,
    notch_freq=50.0,
    normalize_method="minmax",
    reject_unvalidated=False,
    max_records=None,
    n_workers=15,
    verbose=True,
)

# 3. Procesar MIMIC
mimic_signals, mimic_labels, mimic_df = process_mimic_dataset_fast(
    overwrite=False,
    apply_quality_check=True,
    apply_notch=True,
    notch_freq=50.0,
    normalize_method="minmax",
    max_records=None,
    n_workers=15,
    verbose=True,
)

# 4. Combinar
X = np.concatenate([ptbxl_signals, mimic_signals], axis=0)
y = np.concatenate([ptbxl_labels, mimic_labels], axis=0)
metadata = pd.concat([ptbxl_df, mimic_df], ignore_index=True)

# 5. Balancear
X_balanced, y_balanced, balanced_indices = balance_dataset(
    X, y,
    random_state=42,
    return_indices=True
)
metadata = metadata.iloc[balanced_indices].reset_index(drop=True)

# 6. Crear splits
result = create_splits_to_disk(
    X_balanced, y_balanced,
    output_dir=OUTPUT_DIR,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    chunk_size=10000,
    random_state=42,
)
X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = result

# 7. Mapear metadata
metadata_train = metadata.iloc[train_idx].reset_index(drop=True)
metadata_val = metadata.iloc[val_idx].reset_index(drop=True)
metadata_test = metadata.iloc[test_idx].reset_index(drop=True)

# 8. Crear folds
folds = create_stratified_folds(X_train, y_train, n_splits=10, random_state=42)
folds_train = [fold[0] for fold in folds]
folds_val = [fold[1] for fold in folds]

# 9. Guardar
save_dataset(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    metadata_train, metadata_val, metadata_test,
    folds_train, folds_val,
    output_dir=OUTPUT_DIR,
)
```

---

## ✅ Checklist de Verificación

Antes de ejecutar el pipeline completo:

- [ ] Datasets descargados:
  - [ ] PTB-XL en `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
  - [ ] MIMIC-IV-ECG en `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/`
- [ ] Archivos CSV presentes:
  - [ ] `ptbxl_database.csv`
  - [ ] `machine_measurements.csv`
  - [ ] `record_list.csv`
- [ ] Dependencias instaladas: `wfdb`, `numpy`, `pandas`, `scipy`, `sklearn`
- [ ] Memoria suficiente: 32+ GB RAM recomendado
- [ ] Espacio en disco: 25+ GB libres
- [ ] CPU con múltiples cores: Para paralelización (15 workers recomendado)

---

**Última actualización:** 2025-01-XX  
**Autor:** Sistema de documentación automática  
**Versión:** 2.0  
**Basado en:** Ejecución real del notebook `build_supervised_ecg_dataset.ipynb`
