# Documentación Completa: Pipeline de Dataset Supervisado ECG

## 📋 Resumen Ejecutivo

Este documento describe el **pipeline completo** para construir un dataset supervisado binario de ECG (NORMAL vs ANÓMALO) a partir de los datasets PTB-XL y MIMIC-IV-ECG. El pipeline procesa, etiqueta, filtra, normaliza y organiza los datos en splits de entrenamiento, validación y test con folds estratificados.

**Resultados del ejemplo completo:**
- **Total registros procesados:** 496,244 (979 PTB-XL + 495,265 MIMIC)
- **Después de balanceo:** 386,670 registros (50% normales, 50% anómalos)
- **Splits finales:**
  - Train: 270,668 (70%)
  - Val: 58,001 (15%)
  - Test: 58,001 (15%)
- **Shape final:** (386,670, 5000, 3) → 386,670 registros × 5000 muestras × 3 leads
- **Tiempo total:** ~55 minutos (procesamiento) + ~13 minutos (balanceo) + ~13 minutos (splits)

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

## 📝 Paso 1: Configuración e Importaciones

### 1.1. Importaciones Principales

```python
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Módulos del pipeline
from supervised_ecg_pipeline import (
    OUTPUT_DIR,
    create_splits,
    create_splits_to_disk,
    create_stratified_folds,
    save_dataset,
    balance_dataset,
    ensure_dir,
)

# Versión optimizada (paralela)
from supervised_ecg_pipeline_fast import (
    process_mimic_dataset_fast,
    process_ptbxl_dataset_fast,
)
```

### 1.2. Parámetros de Configuración

```python
# Límites de registros (None = todos)
MAX_PTB = None      # Procesar todos los registros de PTB-XL
MAX_MIMIC = None    # Procesar todos los registros de MIMIC

# Procesamiento
USE_FAST = True              # Usar versión paralela optimizada
N_WORKERS = None             # Auto: cpu_count() - 1 = 15 workers
APPLY_QUALITY_CHECK = True   # Verificar calidad de señal
APPLY_NOTCH = True           # Aplicar filtro notch
NOTCH_FREQ = 50.0            # Frecuencia notch (50 Hz)
NORMALIZE_METHOD = "minmax"  # Método de normalización
REJECT_UNVALIDATED = False   # No rechazar no validados
DO_BALANCE = True            # Balancear dataset
MINIMAL_QUALITY = False      # Calidad mínima (más rápido)
```

### 1.3. Directorio de Salida

```
S:\Proyecto final\data\Datos_supervisados\
```

**Ubicación:** Definido en `supervised_ecg_pipeline.py` como `OUTPUT_DIR`

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

### 2.2. Estadísticas Detalladas

| Métrica | Valor | Porcentaje |
|---------|-------|------------|
| **Total procesados** | 979 | 4.5% |
| **Rechazados** | 20,820 | 95.5% |
| **Normales (label=0)** | 397 | 40.6% |
| **Anómalos (label=1)** | 582 | 59.4% |
| **Shape** | (979, 5000, 3) | - |
| **Memoria** | 0.05 GB | - |

### 2.3. Razones de Rechazo (Estimadas)

Basado en el código y patrones típicos:

1. **Sin clasificación clara (~60-70%):**
   - Códigos SCP ambiguos
   - Múltiples códigos contradictorios
   - Códigos no incluidos en patrones de etiquetado

2. **Problemas de calidad (~15-20%):**
   - Señales con ruido excesivo
   - Deriva de línea base
   - Problemas con electrodos

3. **Leads faltantes (~5-10%):**
   - Archivos incompletos
   - Nombres de leads no reconocidos

4. **Errores de procesamiento (~5-10%):**
   - Archivos corruptos
   - Errores de lectura WFDB
   - Señales demasiado cortas

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

### 3.2. Estadísticas Detalladas

| Métrica | Valor | Porcentaje |
|---------|-------|------------|
| **Total procesados** | 495,265 | 61.9% |
| **Rechazados** | 304,770 | 38.1% |
| **Normales (label=0)** | 192,938 | 39.0% |
| **Anómalos (label=1)** | 302,327 | 61.0% |
| **Shape** | (495,265, 5000, 3) | - |
| **Memoria** | 27.68 GB | - |

### 3.3. Diferencias con PTB-XL

**MIMIC tiene mayor tasa de aceptación (61.9% vs 4.5%) porque:**
1. Reportes de texto más estructurados
2. Menos códigos ambiguos
3. Mejor calidad general de señales
4. Patrones de etiquetado más flexibles

**Procesamiento más lento (54 min vs 0.43 min) porque:**
1. 40× más registros (800k vs 22k)
2. Archivos distribuidos en estructura de carpetas más compleja
3. Necesita resampleo de 250 Hz → 500 Hz

---

## 📊 Paso 4: Construir Dataset Combinado

### 4.1. Proceso de Combinación

```python
# Combinar señales
X = np.concatenate([ptbxl_signals, mimic_signals], axis=0)

# Combinar labels
y = np.concatenate([ptbxl_labels, mimic_labels], axis=0)

# Combinar metadatos
metadata = pd.concat([ptbxl_df, mimic_df], ignore_index=True)
```

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

---

## 📊 Paso 5: Balancear Dataset

### 5.1. Proceso de Balanceo

El balanceo realiza **downsampling estratificado** de la clase mayoritaria:

```python
X_balanced, y_balanced, balanced_indices = balance_dataset(
    X, y,
    random_state=42,
    return_indices=True
)
```

**Algoritmo:**
1. Identificar la clase minoritaria (normales: 193,335)
2. Para cada clase, seleccionar aleatoriamente `n_min` registros
3. Mezclar aleatoriamente los registros seleccionados

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

### 5.3. Estadísticas

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| **Total registros** | 496,244 | 386,670 | -109,574 (-22.1%) |
| **Normales** | 193,335 | 193,335 | 0 (mantenidos) |
| **Anómalos** | 302,909 | 193,335 | -109,574 (-36.2%) |
| **Balance** | 39% / 61% | 50% / 50% | ✓ Balanceado |

**Registros eliminados:** 109,574 anómalos (seleccionados aleatoriamente)

---

## 📊 Paso 6: Crear Splits Train/Val/Test

### 6.1. Proceso de División

**Método:** División estratificada 70/15/15 usando `create_splits_to_disk()`

**Ventaja:** Guarda directamente en disco para evitar problemas de memoria con datasets grandes.

```python
result = create_splits_to_disk(
    X, y,
    output_dir=OUTPUT_DIR,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    chunk_size=10000,  # Procesar en chunks de 10k
    random_state=42,
)
```

### 6.2. Resultados

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

### 6.3. Estadísticas Detalladas

| Split | Registros | % Total | Normales | Anómalos | Memoria | Archivo |
|-------|-----------|---------|----------|----------|---------|---------|
| **Train** | 270,668 | 70.0% | 135,334 | 135,334 | 15.12 GB | `X_train.npy` |
| **Val** | 58,001 | 15.0% | 29,000 | 29,001 | 3.24 GB | `X_val.npy` |
| **Test** | 58,001 | 15.0% | 29,001 | 29,000 | 3.24 GB | `X_test.npy` |
| **TOTAL** | 386,670 | 100% | 193,335 | 193,335 | 21.60 GB | - |

### 6.4. Archivos Generados

```
data/Datos_supervisados/numpy/
├── X_train.npy      (15,487.75 MB)
├── y_train.npy      (2.07 MB)
├── X_val.npy        (3,318.84 MB)
├── y_val.npy        (0.44 MB)
├── X_test.npy       (3,318.84 MB)
├── y_test.npy       (0.44 MB)
└── split_indices.npz (índices de división)
```

**Nota:** Los arrays están guardados como memoria mapeada (`mmap_mode='r'`) para acceso eficiente sin cargar todo en RAM.

---

## 📊 Paso 7: Crear Folds Estratificados

### 7.1. Proceso

**Método:** Cross-validation estratificada con 10 folds usando `StratifiedKFold`

```python
folds = create_stratified_folds(
    X_train, y_train,
    n_splits=10,
    random_state=42
)
```

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

**Archivos generados:**
- `metadata/folds_train_indices.npy` (18.59 MB)
- `metadata/folds_val_indices.npy` (2.07 MB)

---

## 📊 Paso 8: Guardar Dataset

### 8.1. Estructura de Archivos

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

### 8.2. Contenido de Archivos

**`master_labels.csv`:**
- Metadatos del conjunto de entrenamiento
- Columnas: `record_id`, `source`, `label`, `label_reason`, etc.

**`master_labels_full.csv`:**
- Metadatos de todos los splits (train/val/test)
- Incluye columna `split` indicando a qué conjunto pertenece

**`folds_train_indices.npy` / `folds_val_indices.npy`:**
- Arrays de índices para cada fold
- Formato: Lista de arrays numpy

### 8.3. Tiempo de Guardado

```
✓ Dataset guardado exitosamente
  Tiempo: 0.04 minutos
  Ubicación: S:\Proyecto final\data\Datos_supervisados
```

**Nota:** Los arrays numpy ya estaban guardados en el paso 6, por lo que este paso solo guarda metadatos y folds.

---

## 📊 Paso 9: Verificar Datos Guardados

### 9.1. Verificación de Archivos

Todos los archivos se verifican para confirmar que se guardaron correctamente:

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

**Tamaño total:** ~22.5 GB

---

## 📊 Paso 10: Visualizar Ejemplos (Opcional)

### 10.1. Generación de Ejemplos

Se generan ejemplos visuales de señales procesadas:

```python
plot_ecg_comparison(
    raw=signal,
    filtered=None,
    normalized=signal,
    title=f"{record_id} - {'NORMAL' if label == 0 else 'ANÓMALO'}",
    save_path=OUTPUT_DIR / "raw_examples" / f"example_{i}.png"
)
```

### 10.2. Resultados

```
✓ 3 ejemplos visuales creados
  Ubicación: S:\Proyecto final\data\Datos_supervisados\raw_examples\
  
  Ejemplos:
    - example_001_mimic_13744125_41662601.png
    - example_002_mimic_18832132_48870396.png
    - example_003_mimic_12182445_49330246.png
```

Cada imagen muestra las 3 derivaciones (II, V1, V5) normalizadas.

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

| Paso | Tiempo | % del Total |
|------|--------|------------|
| **1. Configuración** | < 1 s | < 0.1% |
| **2. Procesar PTB-XL** | 0.43 min | 0.5% |
| **3. Procesar MIMIC** | 54.07 min | 60.0% |
| **4. Combinar datasets** | 11.88 min | 13.2% |
| **5. Balancear** | 21.56 min | 23.9% |
| **6. Crear splits** | 12.90 min | 14.3% |
| **7. Crear folds** | 0.21 s | < 0.1% |
| **8. Guardar dataset** | 0.04 min | < 0.1% |
| **TOTAL** | ~90 minutos | 100% |

### Características del Dataset Final

- **Formato:** NumPy arrays (float32)
- **Estructura:** (N, T, C) donde:
  - N = número de registros
  - T = 5000 muestras (10 segundos)
  - C = 3 leads (II, V1, V5)
- **Frecuencia de muestreo:** 500 Hz
- **Duración:** 10 segundos por registro
- **Normalización:** Min-Max [0, 1]
- **Balance:** 50% normales, 50% anómalos
- **Splits:** Estratificados (mantienen proporción 50/50)

---

## 🔧 Parámetros Configurables

### Parámetros de Procesamiento

```python
# Límites
MAX_PTB = None          # Límite de registros PTB-XL
MAX_MIMIC = None        # Límite de registros MIMIC

# Calidad
APPLY_QUALITY_CHECK = True   # Verificar calidad de señal
MINIMAL_QUALITY = False      # Calidad mínima (más rápido)

# Filtrado
APPLY_NOTCH = True           # Filtro notch
NOTCH_FREQ = 50.0           # 50 o 60 Hz

# Normalización
NORMALIZE_METHOD = "minmax"  # "minmax" o "zscore"

# Validación
REJECT_UNVALIDATED = False   # Rechazar no validados

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
```

### Parámetros de Folds

```python
n_splits = 10         # 10 folds
random_state = 42     # Reproducibilidad
```

---

## 🐛 Troubleshooting

### Problema: Error de Memoria al Crear Splits

**Síntoma:**
```
MemoryError: Unable to allocate 3.24 GiB for an array
```

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
    chunk_size=10000,
    random_state=42,
)
```

### Problema: Procesamiento Muy Lento

**Soluciones:**
1. Reducir `MAX_PTB` o `MAX_MIMIC` para pruebas
2. Usar `MINIMAL_QUALITY = True` para deshabilitar checks menos críticos
3. Reducir `N_WORKERS` si hay problemas de memoria
4. Procesar en batches más pequeños

### Problema: Muy Pocos Registros Procesados

**Causas:**
1. Criterios de etiquetado muy estrictos
2. Filtros de calidad muy restrictivos
3. `REJECT_UNVALIDATED = True` rechaza muchos registros

**Soluciones:**
- Ajustar `REJECT_UNVALIDATED = False`
- Relajar `APPLY_QUALITY_CHECK` o usar `MINIMAL_QUALITY = True`
- Revisar patrones de etiquetado en el código

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
- `process_mimic_dataset_fast()`: Procesar MIMIC en paralelo
- `balance_dataset()`: Balancear dataset
- `create_splits_to_disk()`: Crear splits guardando en disco
- `create_stratified_folds()`: Crear folds estratificados
- `save_dataset()`: Guardar dataset completo

---

## ✅ Checklist de Verificación

Antes de ejecutar el pipeline completo:

- [ ] Datasets descargados:
  - [ ] PTB-XL en `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
  - [ ] MIMIC-IV-ECG en `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/`
- [ ] Dependencias instaladas: `wfdb`, `numpy`, `pandas`, `scipy`, `sklearn`
- [ ] Memoria suficiente: 32+ GB RAM recomendado
- [ ] Espacio en disco: 25+ GB libres
- [ ] CPU con múltiples cores: Para paralelización (15 workers recomendado)

---

**Última actualización:** 2025-01-XX  
**Autor:** Sistema de documentación automática  
**Versión:** 1.0  
**Basado en:** Ejecución real del notebook `build_supervised_ecg_dataset.ipynb`

