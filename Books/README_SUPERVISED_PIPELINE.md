# Pipeline de Dataset Supervisado ECG (NORMAL vs ANÓMALO)

Este pipeline completo procesa los datasets PTB-XL y MIMIC-IV-ECG para crear un dataset binario supervisado.

## Archivos Creados

1. **`supervised_ecg_pipeline.py`**: Módulo principal con todas las funciones del pipeline
2. **`build_supervised_ecg_dataset.py`**: Script principal para ejecutar el pipeline completo

## Características

### Etiquetado

**MIMIC-IV-ECG:**
- Anómalos (label=1): Detecta patrones como infartos, isquemia, arritmias, bloqueos, ST-T changes, etc.
- Normales (label=0): Detecta "Normal ECG", "Sinus rhythm" (sin anormalidades), etc.
- Descarta: Registros con problemas de calidad o "unsuitable for analysis"

**PTB-XL:**
- Anómalos (label=1): Basado en códigos SCP (IMI, ISC_, LVH, AFIB, CLBBB, etc.)
- Normales (label=0): Código NORM con valor 100.0 o 80.0, sin otros códigos patológicos
- Descarta: Registros con problemas de calidad (baseline_drift, noise, etc.)

### Procesamiento de Señales

- **Selección de leads**: Solo II, V1, V5 (con mapeo automático de variantes)
- **Filtrado**:
  - Notch 50/60 Hz (configurable)
  - Bandpass 0.5-40 Hz (Butterworth orden 4)
- **Normalización**: Min-Max a [0,1] por lead (o Z-score opcional)
- **Resampleo**: A 500 Hz y 10 segundos (5000 muestras)
- **Filtros de calidad**:
  - Flat signal detection
  - Saturation detection
  - Discontinuity detection
  - NaN ratio check
  - Minimum duration check

### Balanceo y Splits

- **Balanceo**: Downsampling estratificado a la clase minoritaria (opcional)
- **Splits**: Train (70%) / Val (15%) / Test (15%) estratificados
- **Cross-validation**: 10 folds estratificados sobre el conjunto de entrenamiento

### Estructura de Salida

```
data/
  Datos_supervisados/
    metadata/
      master_labels.csv          # Metadatos completos
      master_labels_full.csv     # Con información de splits
      folds_train_indices.npy    # Índices de train para cada fold
      folds_val_indices.npy      # Índices de val para cada fold
    numpy/
      X_train.npy, y_train.npy
      X_val.npy, y_val.npy
      X_test.npy, y_test.npy
    raw_examples/                # Ejemplos visuales (opcional)
      example_001_*.png
      ...
```

## Uso

### Ejecución Básica

```bash
cd Books
python build_supervised_ecg_dataset.py
```

### Opciones Principales

```bash
# Procesar solo un subconjunto (útil para pruebas)
python build_supervised_ecg_dataset.py --max-ptb 1000 --max-mimic 1000

# No balancear el dataset (usar todos los registros)
python build_supervised_ecg_dataset.py --no-balance

# Usar notch a 60 Hz en lugar de 50 Hz
python build_supervised_ecg_dataset.py --notch-freq 60

# Crear ejemplos visuales
python build_supervised_ecg_dataset.py --create-examples --n-examples 20

# Rechazar reportes no validados en PTB-XL
python build_supervised_ecg_dataset.py --reject-unvalidated

# Deshabilitar verificación de calidad (no recomendado)
python build_supervised_ecg_dataset.py --no-quality-check

# Sobrescribir archivos existentes
python build_supervised_ecg_dataset.py --overwrite
```

### Uso Programático

```python
from supervised_ecg_pipeline import (
    process_ptbxl_dataset,
    process_mimic_dataset,
    build_supervised_dataset,
    create_splits,
    create_stratified_folds,
    save_dataset,
    plot_ecg_comparison,
)

# Procesar PTB-XL
ptbxl_df = process_ptbxl_dataset(max_records=1000, verbose=True)

# Procesar MIMIC
mimic_df = process_mimic_dataset(max_records=1000, verbose=True)

# Construir dataset combinado
X, y, metadata = build_supervised_dataset(
    ptbxl_df=ptbxl_df,
    mimic_df=mimic_df,
    balance=True,
)

# Crear splits
X_train, y_train, X_val, y_val, X_test, y_test = create_splits(X, y)

# Crear folds
folds = create_stratified_folds(X_train, y_train, n_splits=10)

# Guardar
save_dataset(
    X_train, y_train, X_val, y_val, X_test, y_test,
    metadata_train, metadata_val, metadata_test,
    folds_train, folds_val,
)
```

## Visualización

### Función Simple

```python
from supervised_ecg_pipeline import plot_ecg

# Cargar un ejemplo
X_test = np.load("data/Datos_supervisados/numpy/X_test.npy")
signal = X_test[0]

# Visualizar
plot_ecg(signal, title="Ejemplo ECG Normalizado")
```

### Comparación Completa

```python
from supervised_ecg_pipeline import plot_ecg_comparison

# Comparar crudo, filtrado y normalizado
plot_ecg_comparison(
    raw=raw_signal,
    filtered=filtered_signal,
    normalized=normalized_signal,
    title="Procesamiento ECG",
    save_path="comparison.png",
)
```

## Parámetros Configurables

Todos los parámetros están definidos al inicio de `supervised_ecg_pipeline.py`:

- `TARGET_LEADS`: Leads objetivo (por defecto: ["II", "V1", "V5"])
- `SAMPLING_RATE`: Frecuencia de muestreo objetivo (500 Hz)
- `SIGNAL_DURATION`: Duración objetivo (10 segundos)
- `NOTCH_FREQ`: Frecuencia del notch (50 Hz)
- `BANDPASS_LOW`, `BANDPASS_HIGH`: Rangos del filtro pasa-banda (0.5-40 Hz)
- `MIN_STD`, `MAX_NAN_RATIO`, etc.: Umbrales de calidad
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Proporciones de splits
- `N_FOLDS`: Número de folds para CV (10)

## Notas

1. **Memoria**: El procesamiento de datasets grandes puede requerir mucha RAM. Usa `--max-ptb` y `--max-mimic` para pruebas iniciales.

2. **Tiempo**: El procesamiento completo puede tardar varias horas dependiendo del tamaño del dataset y hardware.

3. **Reproducibilidad**: Todos los procesos aleatorios usan `random_state=42` por defecto.

4. **Paths**: Asegúrate de que los directorios de datasets estén correctamente configurados en `supervised_ecg_pipeline.py`.

## Ejemplo de Salida

```
================================================================================
PIPELINE DE DATASET SUPERVISADO ECG
================================================================================

[PTB-XL] Procesando 1000 registros...
  Procesados: 100/1000 | Válidos: 85 | Rechazados: 15
  ...
✓ PTB-XL: 850 registros procesados
  - Normales: 425
  - Anómalos: 425

[MIMIC] Procesando 1000 registros...
  ...
✓ MIMIC: 720 registros procesados
  - Normales: 360
  - Anómalos: 360

✓ Dataset combinado: 1570 registros
  - Normales: 785
  - Anómalos: 785

✓ Dataset balanceado: 1570 registros
  - Normales: 785
  - Anómalos: 785

✓ Splits creados:
  - Train: 1099 (70.0%)
  - Val: 236 (15.0%)
  - Test: 235 (15.0%)

✓ 10 folds estratificados creados

✓ Dataset guardado en data/Datos_supervisados/
```

