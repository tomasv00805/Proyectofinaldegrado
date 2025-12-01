# 📚 Documentación General del Proyecto ECG

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
3. [Pipeline de Datos](#pipeline-de-datos)
4. [Modelos de Clasificación Supervisada](#modelos-de-clasificación-supervisada)
5. [Modelos de Detección de Anomalías](#modelos-de-detección-de-anomalías)
6. [Utilidades y Herramientas](#utilidades-y-herramientas)
7. [Guía de Uso Rápida](#guía-de-uso-rápida)
8. [Estructura de Archivos](#estructura-de-archivos)
9. [Requisitos y Configuración](#requisitos-y-configuración)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Introducción

Este proyecto implementa un sistema completo de análisis de señales ECG (electrocardiogramas) utilizando técnicas de aprendizaje profundo. El objetivo principal es desarrollar modelos capaces de:

1. **Clasificación Supervisada**: Clasificar ECG como **NORMAL** (0) o **ANÓMALO** (1) usando etiquetas de entrenamiento
2. **Detección de Anomalías**: Detectar ECG anómalos mediante autoencoders entrenados solo con ejemplos normales

### Datasets Utilizados

- **PTB-XL**: Dataset público de ECG con 21,799 registros etiquetados con códigos SCP
- **MIMIC-IV-ECG**: Subset de MIMIC-IV con reportes de diagnóstico de ECG

### Características Principales

- ✅ Procesamiento robusto de señales ECG (filtrado, normalización, resampleo)
- ✅ Múltiples arquitecturas de modelos (CNN, LSTM, Transformer, Autoencoders)
- ✅ Integración con MLflow para tracking de experimentos
- ✅ Orquestación con Prefect 2.x
- ✅ Soporte para GPU (RTX 5080 compatible)
- ✅ Pipeline completo desde datos crudos hasta modelos entrenados

---

## 🏗️ Arquitectura del Proyecto

### Flujo General del Proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATOS CRUDOS                                 │
│  PTB-XL + MIMIC-IV-ECG (archivos .hea, .dat, CSV)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE DE PREPROCESAMIENTO                        │
│  • Etiquetado (NORMAL vs ANÓMALO)                               │
│  • Selección de leads (II, V1, V5)                              │
│  • Filtrado (notch 50Hz, bandpass 0.5-40Hz)                     │
│  • Normalización (Min-Max)                                       │
│  • Resampleo (10 seg @ 500Hz → 5000 muestras)                   │
│  • Control de calidad                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────────┐          ┌───────────────────────┐
│  DATOS SUPERVISADOS   │          │ DATOS NO SUPERVISADOS │
│  (Clasificación)      │          │ (Detección Anomalías) │
│                       │          │                       │
│  • Train: Normales +  │          │  • Train: Solo        │
│    Anómalos           │          │    normales           │
│  • Val/Test: Mezcla   │          │  • Val/Test: Mezcla   │
│  • Balanceado 50/50   │          │  • Sin balancear      │
└───────────────────────┘          └───────────────────────┘
        ↓                                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DOWNSAMPLING (Opcional)                       │
│  500Hz (5000 muestras) → 200Hz (2000 muestras)                 │
└─────────────────────────────────────────────────────────────────┘
        ↓                                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRENAMIENTO DE MODELOS                     │
│                                                                 │
│  CLASIFICACIÓN SUPERVISADA:        DETECCIÓN DE ANOMALÍAS:     │
│  • CNN1D                           • CNN1D Autoencoder         │
│  • CNN1D + LSTM                    • CNN1D + LSTM Autoencoder  │
│  • CNN1D + Transformer             • LSTM Autoencoder          │
│  • LSTM                            • Selección de umbral        │
└─────────────────────────────────────────────────────────────────┘
        ↓                                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUACIÓN Y MÉTRICAS                         │
│  • Accuracy, Precision, Recall, F1                              │
│  • Matrices de confusión                                         │
│  • Curvas de entrenamiento                                       │
│  • MLflow tracking                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

1. **Módulos de Procesamiento**:
   - `supervised_ecg_pipeline.py`: Pipeline completo para datos supervisados
   - `ecg_preprocessing.py`: Funciones de preprocesamiento reutilizables
   - `evaluation_threshold_tuning.py`: Evaluación y búsqueda de umbrales

2. **Notebooks de Construcción de Datos**:
   - `build_supervised_ecg_dataset.ipynb`: Crea dataset supervisado
   - `build_unsupervised_ecg_dataset.ipynb`: Crea dataset no supervisado

3. **Notebooks de Downsampling**:
   - `downsample_supervised_data.ipynb`: Reduce frecuencia de datos supervisados
   - `downsample_unsupervised_data.ipynb`: Reduce frecuencia de datos no supervisados

4. **Notebooks de Modelos**:
   - Clasificación: `cnn1d_*`, `cnn1d_lstm_*`, `cnn1d_transformer_*`, `lstm_*`
   - Anomalías: `cnn1d_autoencoder_*`, `cnn1d_lstm_autoencoder_*`, `lstm_autoencoder_*`

---

## 📊 Pipeline de Datos

### 1. Pipeline de Datos Supervisados

**Archivo principal**: `build_supervised_ecg_dataset.ipynb` / `build_supervised_ecg_dataset.py`

**Propósito**: Crear un dataset binario balanceado (NORMAL vs ANÓMALO) para entrenamiento supervisado.

**Proceso**:

1. **Procesamiento PTB-XL**:
   - Lee registros desde archivos `.hea` y `.dat`
   - Etiqueta usando códigos SCP (NORM=normal, IMI/ISC/LVH/etc=anómalo)
   - Filtra por calidad de señal

2. **Procesamiento MIMIC-IV-ECG**:
   - Lee registros desde archivos `.hea` y `.dat`
   - Etiqueta usando reportes de texto (patrones regex)
   - Filtra por calidad de señal

3. **Preprocesamiento de Señales**:
   - Selección de leads: **II, V1, V5** (con mapeo automático de variantes)
   - Filtrado:
     - Notch 50/60 Hz (configurable)
     - Bandpass 0.5-40 Hz (Butterworth orden 4)
   - Normalización: Min-Max a [0,1] por lead
   - Resampleo: A 500 Hz y 10 segundos (5000 muestras)

4. **Control de Calidad**:
   - Detección de señales planas
   - Detección de saturación
   - Detección de discontinuidades
   - Verificación de ratio de NaN
   - Verificación de duración mínima

5. **Balanceo y Splits**:
   - Balanceo: Downsampling estratificado a la clase minoritaria
   - Splits: Train (70%) / Val (15%) / Test (15%) estratificados
   - Cross-validation: 10 folds estratificados sobre train

**Salida**:
```
data/Datos_supervisados/
├── numpy/
│   ├── X_train.npy, y_train.npy
│   ├── X_val.npy, y_val.npy
│   └── X_test.npy, y_test.npy
├── tensors/ (opcional, para PyTorch)
│   ├── X_train.pt, y_train.pt
│   └── ...
├── tensors_200hz/ (si se aplica downsampling)
│   └── ...
└── metadata/
    ├── master_labels.csv
    ├── master_labels_full.csv
    ├── folds_train_indices.npy
    └── folds_val_indices.npy
```

**Documentación detallada**: Ver `Documentacion Datos Supervisados.md`

### 2. Pipeline de Datos No Supervisados

**Archivo principal**: `build_unsupervised_ecg_dataset.ipynb`

**Propósito**: Crear un dataset para entrenamiento de autoencoders (solo normales en train).

**Proceso**:

1. **Carga desde datos supervisados**: Reutiliza los datos ya procesados del pipeline supervisado
2. **Separación especial**:
   - **Train**: Solo ECG normales (label == 0) - para entrenar el autoencoder
   - **Val/Test**: Mezcla de normales y anómalos (con labels) - para evaluación
3. **Sin balanceo**: Mantiene la distribución natural de los datos

**Salida**:
```
data/Datos_no_supervisados/
├── numpy/
│   ├── X_train.npy, y_train.npy  (solo normales)
│   ├── X_val.npy, y_val.npy
│   └── X_test.npy, y_test.npy
└── tensors_200hz/ (si se aplica downsampling)
    └── ...
```

### 3. Downsampling

**Archivos**: `downsample_supervised_data.ipynb`, `downsample_unsupervised_data.ipynb`

**Propósito**: Reducir la frecuencia de muestreo de 500Hz a 200Hz para:
- Reducir el tamaño de archivos (~2.5x)
- Acelerar el entrenamiento
- Mantener el preprocesado original

**Proceso**:
- Usa `scipy.signal.resample` con filtrado anti-aliasing automático
- Convierte 5000 muestras → 2000 muestras (mantiene 10 segundos)
- Guarda en carpetas separadas (`numpy_200hz/`, `tensors_200hz/`)

---

## 🧠 Modelos de Clasificación Supervisada

Todos los modelos de clasificación supervisada comparten:
- **Input**: Datos desde `Datos_supervisados/tensors_200hz/` (archivos `.pt`)
- **Output**: Modelo entrenado + métricas en MLflow
- **Etiquetas**: 0 = NORMAL, 1 = ANÓMALO
- **Evaluación**: Accuracy, Precision, Recall, F1, matrices de confusión

### 1. CNN1D (`cnn1d_classification_supervised.ipynb`)

**Arquitectura**: CNN1D pura para extracción de características locales.

**Características**:
- Múltiples capas convolucionales 1D
- Pooling para reducción dimensional
- Capas fully connected al final
- Optimizado para capturar patrones locales en señales temporales

**Uso recomendado**: Baseline rápido, bueno para comparación.

### 2. CNN1D + LSTM (`cnn1d_lstm_classification_supervised.ipynb`)

**Arquitectura**: Híbrida - CNN1D para características locales + LSTM para dependencias temporales.

**Características**:
- CNN1D extrae características locales
- LSTM captura dependencias temporales largas
- Combina lo mejor de ambas arquitecturas

**Uso recomendado**: **Recomendado para clasificación** - balance entre rendimiento y complejidad.

### 3. CNN1D + Transformer (`cnn1d_transformer_classification_supervised.ipynb`)

**Arquitectura**: CNN1D + Transformer para atención global.

**Características**:
- CNN1D para características locales
- Transformer con self-attention para relaciones globales
- Captura dependencias complejas en toda la señal

**Uso recomendado**: Cuando se necesita el mejor rendimiento posible (más lento de entrenar).

### 4. LSTM (`lstm_classification_supervised.ipynb`)

**Arquitectura**: LSTM puro para secuencias temporales.

**Características**:
- Múltiples capas LSTM
- Captura dependencias temporales largas
- Sin convoluciones

**Uso recomendado**: Comparación con arquitecturas híbridas.

---

## 🔍 Modelos de Detección de Anomalías

Todos los modelos de detección de anomalías comparten:
- **Entrenamiento**: Solo con ejemplos normales (no supervisado)
- **Input**: Datos desde `Datos_no_supervisados/tensors_200hz/`
- **Detección**: Basada en error de reconstrucción
- **Umbral**: Selección automática o manual del umbral óptimo

### 1. CNN1D Autoencoder (`cnn1d_autoencoder_anomaly_detection.ipynb`)

**Arquitectura**: Encoder-decoder CNN1D puro.

**Características**:
- Encoder: Capas convolucionales que comprimen la señal
- Decoder: Capas de transposición convolucional o upsampling que reconstruyen
- Entrenamiento: Minimiza error de reconstrucción en normales
- Detección: ECG con error alto → anómalo

**Uso recomendado**: Baseline rápido para detección de anomalías.

### 2. CNN1D + LSTM Autoencoder (`cnn1d_lstm_autoencoder_anomaly_detection.ipynb`)

**Arquitectura**: Híbrida - CNN1D + LSTM en encoder y decoder.

**Características**:
- Combina capacidades de CNN y LSTM
- Mejor captura de patrones temporales complejos
- Reconstrucción más precisa

**Uso recomendado**: **Recomendado para detección de anomalías** - mejor balance rendimiento/complejidad.

### 3. LSTM Autoencoder (`lstm_autoencoder_pipeline.ipynb`)

**Arquitectura**: LSTM puro en encoder y decoder.

**Características**:
- Encoder LSTM comprime la secuencia
- Decoder LSTM reconstruye
- Enfocado en dependencias temporales

**Uso recomendado**: Comparación con arquitecturas híbridas.

### Selección de Umbral

**Archivo**: `evaluation_threshold_tuning.py`

**Métodos**:
1. **Automático (recomendado)**: `find_optimal_threshold()` prueba varios percentiles y selecciona el mejor según F2-score
2. **Manual**: Define un umbral fijo basado en estadísticas
3. **Basado en percentiles**: `threshold = np.percentile(errors, 95)`

**Lógica**:
- Si `error_reconstrucción > umbral` → ECG es **ANÓMALO** (clase 1)
- Si `error_reconstrucción <= umbral` → ECG es **NORMAL** (clase 0)

---

## 🛠️ Utilidades y Herramientas

### Scripts de Utilidad

1. **`cleanup_splits.py`**: Limpia archivos de splits antiguos
2. **`create_splits_disk.py`**: Crea splits guardando directamente en disco (eficiente en memoria)
3. **`evaluation_threshold_tuning.py`**: Funciones para evaluación y búsqueda de umbrales

### Integración con MLflow

Todos los notebooks de entrenamiento integran MLflow para:
- Tracking de hiperparámetros
- Logging de métricas durante entrenamiento
- Guardado de modelos
- Comparación de experimentos

**Ubicación de runs**: `Books/mlruns/`

### Orquestación con Prefect

Los notebooks principales usan Prefect 2.x para:
- Orquestación del flujo de entrenamiento
- Manejo de errores y reintentos
- Logging estructurado

---

## 🚀 Guía de Uso Rápida

### Flujo Completo Recomendado

#### Paso 1: Preparar Datos Supervisados

```bash
cd Books
# Opción 1: Ejecutar notebook
jupyter notebook build_supervised_ecg_dataset.ipynb

# Opción 2: Ejecutar script
python build_supervised_ecg_dataset.py
```

**Tiempo estimado**: 1-2 horas (depende del tamaño del dataset)

#### Paso 2: (Opcional) Downsampling a 200Hz

```bash
jupyter notebook downsample_supervised_data.ipynb
```

**Tiempo estimado**: 30-60 minutos

#### Paso 3: Preparar Datos No Supervisados

```bash
jupyter notebook build_unsupervised_ecg_dataset.ipynb
```

**Tiempo estimado**: 10-20 minutos

#### Paso 4: (Opcional) Downsampling Datos No Supervisados

```bash
jupyter notebook downsample_unsupervised_data.ipynb
```

**Tiempo estimado**: 30-60 minutos

#### Paso 5: Entrenar Modelos

**Clasificación Supervisada** (elige uno):
```bash
# Recomendado: CNN1D + LSTM
jupyter notebook cnn1d_lstm_classification_supervised.ipynb

# Otras opciones:
jupyter notebook cnn1d_classification_supervised.ipynb
jupyter notebook cnn1d_transformer_classification_supervised.ipynb
jupyter notebook lstm_classification_supervised.ipynb
```

**Detección de Anomalías** (elige uno):
```bash
# Recomendado: CNN1D + LSTM Autoencoder
jupyter notebook cnn1d_lstm_autoencoder_anomaly_detection.ipynb

# Otras opciones:
jupyter notebook cnn1d_autoencoder_anomaly_detection.ipynb
jupyter notebook lstm_autoencoder_pipeline.ipynb
```

**Tiempo estimado por modelo**: 2-4 horas (depende de GPU y tamaño de datos)

### Configuración Inicial (Primera Vez)

1. **Setup CUDA (Windows)**:
   - Ejecuta la celda "Setup CUDA y Dependencias" en cualquier notebook
   - Esto configura las DLLs de CUDA para PyTorch
   - **IMPORTANTE**: Reinicia el kernel después de ejecutar esta celda

2. **Configurar Rutas**:
   - Ajusta `DATA_DIR` en cada notebook según tu estructura de carpetas
   - Debe apuntar a `Datos_supervisados/tensors_200hz` o `Datos_no_supervisados/tensors_200hz`

3. **Verificar GPU**:
   - Los notebooks detectan automáticamente si hay GPU disponible
   - Si no hay GPU, usarán CPU (más lento)

### Ejecución Rápida de un Modelo

1. Abre el notebook deseado
2. Ejecuta la celda de **Setup CUDA** (si es primera vez)
3. Configura `DATA_DIR` en la sección de configuración
4. Ejecuta todas las celdas en orden
5. Revisa resultados en MLflow UI: `mlflow ui` en terminal

---

## 📁 Estructura de Archivos

```
Books/
├── 📄 DOCUMENTACION_GENERAL.md          # Este archivo
├── 📄 Documentacion Datos Supervisados.md  # Documentación detallada del pipeline
│
├── 🔧 Módulos Python
│   ├── supervised_ecg_pipeline.py       # Pipeline principal supervisado
│   ├── supervised_ecg_pipeline_fast.py  # Versión paralela optimizada
│   ├── ecg_preprocessing.py             # Funciones de preprocesamiento
│   ├── evaluation_threshold_tuning.py   # Evaluación y umbrales
│   ├── build_supervised_ecg_dataset.py # Script ejecutable pipeline
│   ├── cleanup_splits.py               # Utilidad limpieza
│   └── create_splits_disk.py            # Creación eficiente de splits
│
├── 📊 Notebooks de Construcción de Datos
│   ├── build_supervised_ecg_dataset.ipynb
│   └── build_unsupervised_ecg_dataset.ipynb
│
├── 🔽 Notebooks de Downsampling
│   ├── downsample_supervised_data.ipynb
│   └── downsample_unsupervised_data.ipynb
│
├── 🧠 Notebooks de Clasificación Supervisada
│   ├── cnn1d_classification_supervised.ipynb
│   ├── cnn1d_lstm_classification_supervised.ipynb      ⭐ Recomendado
│   ├── cnn1d_transformer_classification_supervised.ipynb
│   └── lstm_classification_supervised.ipynb
│
├── 🔍 Notebooks de Detección de Anomalías
│   ├── cnn1d_autoencoder_anomaly_detection.ipynb
│   ├── cnn1d_lstm_autoencoder_anomaly_detection.ipynb  ⭐ Recomendado
│   └── lstm_autoencoder_pipeline.ipynb
│
├── 📦 Modelos Guardados
│   ├── models/
│   │   ├── cnn1d_ecg_v1.pt
│   │   ├── cnn1d_lstm_ecg_v1.pt
│   │   └── cnn_transformer_ecg_v1.pt
│   └── sagemaker_models/  # Modelos para AWS SageMaker
│
├── 📈 MLflow Tracking
│   └── mlruns/  # Runs de experimentos
│
└── 📤 Outputs
    └── outputs/  # Gráficos, matrices de confusión, etc.
```

### Estructura de Datos Generados

```
data/
├── Datos_supervisados/
│   ├── numpy/              # Arrays numpy (500Hz)
│   ├── tensors/            # Tensores PyTorch (500Hz)
│   ├── numpy_200hz/        # Arrays numpy (200Hz)
│   ├── tensors_200hz/      # Tensores PyTorch (200Hz) ⭐ Usado por modelos
│   └── metadata/           # Metadatos, labels, folds
│
└── Datos_no_supervisados/
    ├── numpy/              # Arrays numpy (500Hz)
    ├── numpy_200hz/        # Arrays numpy (200Hz)
    └── tensors_200hz/      # Tensores PyTorch (200Hz) ⭐ Usado por modelos
```

---

## ⚙️ Requisitos y Configuración

### Requisitos de Hardware

- **GPU**: Recomendado (RTX 5080 compatible, CUDA 12.8)
- **RAM**: Mínimo 16GB, recomendado 32GB+ para datasets grandes
- **Disco**: ~50-100GB libres para datasets procesados

### Requisitos de Software

- **Python**: 3.11+
- **PyTorch**: Nightly build con CUDA 12.8 (instalado automáticamente en notebooks)
- **Librerías principales**:
  - `torch`, `torchvision`, `torchaudio`
  - `mlflow>=2.16`
  - `prefect>=3`
  - `scikit-learn`
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `wfdb` (para leer archivos PTB-XL/MIMIC)
  - `scipy`

### Configuración de Entorno

Los notebooks instalan automáticamente las dependencias necesarias. Para instalación manual:

```bash
pip install mlflow>=2.16 prefect>=3 scikit-learn matplotlib pandas numpy seaborn ipywidgets wfdb scipy

# PyTorch (para RTX 5080 / CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Configuración de Rutas

Asegúrate de que las rutas en los módulos Python apunten correctamente:

```python
# En supervised_ecg_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PTB_ROOT = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
MIMIC_ROOT = PROJECT_ROOT / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
```

---

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Error de DLLs CUDA en Windows

**Síntoma**: `OSError: [WinError 126] No se puede encontrar el módulo especificado`

**Solución**:
1. Ejecuta la celda "Setup CUDA y Dependencias" en el notebook
2. **Reinicia el kernel de Jupyter** (Kernel → Restart Kernel)
3. Ejecuta la celda de nuevo

#### 2. Out of Memory (OOM)

**Síntoma**: `RuntimeError: CUDA out of memory`

**Soluciones**:
- Reduce `batch_size` en la configuración del notebook
- Usa datos downsampled (200Hz en lugar de 500Hz)
- Procesa en chunks más pequeños
- Cierra otros programas que usen GPU

#### 3. Datos No Encontrados

**Síntoma**: `FileNotFoundError` al cargar datos

**Solución**:
- Verifica que `DATA_DIR` apunta correctamente
- Asegúrate de haber ejecutado el pipeline de construcción de datos primero
- Verifica que los archivos `.pt` existen en `tensors_200hz/`

#### 4. MLflow No Inicia

**Síntoma**: Errores al inicializar MLflow

**Solución**:
- Verifica que `mlflow>=2.16` está instalado
- Asegúrate de tener permisos de escritura en `Books/mlruns/`
- Intenta ejecutar `mlflow ui` manualmente para verificar

#### 5. Prefect No Funciona

**Síntoma**: Errores con Prefect flows

**Solución**:
- Verifica que `prefect>=3` está instalado
- Algunos notebooks pueden funcionar sin Prefect (comenta las secciones de Prefect)

### Verificación de Setup

Ejecuta este código para verificar tu configuración:

```python
import torch
import sys
from pathlib import Path

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Verificar rutas
project_root = Path.cwd().parent
print(f"\nProject root: {project_root}")
print(f"PTB-XL existe: {(project_root / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3').exists()}")
print(f"MIMIC existe: {(project_root / 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0').exists()}")
```

---

## 📝 Notas Adicionales

### Mejores Prácticas

1. **Siempre ejecuta el pipeline de datos primero** antes de entrenar modelos
2. **Usa downsampling a 200Hz** para acelerar entrenamiento sin perder mucho rendimiento
3. **Guarda modelos regularmente** - los notebooks guardan automáticamente en `models/`
4. **Revisa MLflow** para comparar experimentos y encontrar mejores hiperparámetros
5. **Usa GPU** siempre que sea posible - el entrenamiento es mucho más rápido

### Comparación de Modelos

| Modelo | Velocidad | Rendimiento | Complejidad | Recomendado Para |
|--------|-----------|-------------|--------------|------------------|
| CNN1D | ⚡⚡⚡ | ⭐⭐ | Baja | Baseline rápido |
| CNN1D+LSTM | ⚡⚡ | ⭐⭐⭐⭐ | Media | **Uso general** |
| CNN1D+Transformer | ⚡ | ⭐⭐⭐⭐⭐ | Alta | Máximo rendimiento |
| LSTM | ⚡⚡ | ⭐⭐⭐ | Media | Comparación |

### Siguientes Pasos

1. **Hiperparameter Tuning**: Usa MLflow para optimizar hiperparámetros
2. **Ensemble Methods**: Combina múltiples modelos para mejor rendimiento
3. **Transfer Learning**: Usa modelos pre-entrenados si están disponibles
4. **Deployment**: Exporta modelos para producción (ver `sagemaker_models/`)

---

## 📚 Referencias y Documentación Adicional

- **Documentación detallada del pipeline supervisado**: `Documentacion Datos Supervisados.md`
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Prefect Documentation**: https://docs.prefect.io/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html

---

**Última actualización**: 2025-01-XX
**Versión**: 1.0

