# Proyecto Final de Grado - Análisis de ECG con Autoencoders

Este proyecto implementa un sistema completo para el análisis de señales ECG utilizando autoencoders 1D CNN para la detección de anomalías. El sistema procesa datos de los datasets PTB-XL y MIMIC-IV-ECG, aplica filtrado y normalización, y entrena modelos de deep learning para clasificación binaria (NORMAL vs ANÓMALO).

## 📋 Requisitos Previos

- Python 3.8 o superior
- CUDA 12.8+ (opcional, para aceleración GPU con PyTorch)
- Git

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tomasv00805/Proyectofinaldegrado.git
cd Proyectofinaldegrado
```

### 2. Crear un entorno virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Instalación de PyTorch con CUDA (opcional, para GPU)

Si tienes una GPU NVIDIA y quieres usar CUDA, instala PyTorch con soporte CUDA:

**Para CUDA 12.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Para CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Solo CPU:**
```bash
pip install torch torchvision
```

## 📁 Estructura del Proyecto

```
Proyectofinaldegrado/
├── Books/                          # Scripts y notebooks principales
│   ├── supervised_ecg_pipeline.py  # Pipeline principal de procesamiento
│   ├── supervised_ecg_pipeline_fast.py  # Versión optimizada paralela
│   ├── build_supervised_ecg_dataset.py  # Script para construir dataset
│   ├── evaluation_threshold_tuning.py   # Evaluación y búsqueda de umbral
│   ├── ecg_preprocessing.py        # Funciones de preprocesamiento
│   └── *.ipynb                     # Notebooks de Jupyter
├── config/                         # Archivos de configuración
│   └── ae1d_config.json           # Configuración del autoencoder
├── data/                           # Datos procesados (no incluido en repo)
│   └── Datos_supervisados/        # Dataset final preparado
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## 🔧 Uso

### Preparar el Dataset

1. **Descargar los datasets originales:**
   - PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
   - MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg-diagnostic/1.0/

2. **Colocar los datasets en el directorio raíz:**
   - `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`
   - `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/`

3. **Ejecutar el pipeline de construcción del dataset:**

```bash
cd Books
python build_supervised_ecg_dataset.py
```

O usar el notebook interactivo:
```bash
jupyter notebook build_supervised_ecg_dataset.ipynb
```

### Entrenar el Modelo

Abre el notebook principal de entrenamiento:

```bash
jupyter notebook Books/d1CNN_AE_pipeline.ipynb
```

El notebook incluye:
- Configuración de hiperparámetros
- Carga de datos
- Definición del modelo Autoencoder 1D CNN
- Entrenamiento con MLflow y Prefect
- Evaluación y búsqueda de umbral óptimo

### Evaluar el Modelo

Usa el script de evaluación:

```bash
python Books/evaluation_threshold_tuning.py
```

## 📊 Características Principales

- **Procesamiento de señales ECG**: Filtrado, normalización, selección de leads (II, V1, V5)
- **Etiquetado automático**: Clasificación binaria NORMAL vs ANÓMALO basada en diagnósticos
- **Balanceo de datos**: Generación de datasets balanceados
- **Splits estratificados**: Train/Val/Test (70/15/15) + 10 folds para validación cruzada
- **Autoencoder 1D CNN**: Arquitectura profunda para detección de anomalías
- **Tracking de experimentos**: Integración con MLflow para logging y artefactos
- **Optimización de umbral**: Búsqueda automática del umbral óptimo para clasificación

## 🛠️ Dependencias Principales

- **numpy, pandas**: Manipulación de datos
- **scipy, wfdb**: Procesamiento de señales ECG
- **scikit-learn**: Métricas y validación
- **torch**: Deep learning (PyTorch)
- **mlflow**: Tracking de experimentos
- **prefect**: Orquestación de pipelines
- **matplotlib**: Visualización
- **jupyter**: Notebooks interactivos

Ver `requirements.txt` para la lista completa con versiones.

## 📝 Notas Importantes

- Los **datasets originales** y los **modelos entrenados** no están incluidos en el repositorio debido a su tamaño
- Los datos procesados se guardan en `data/Datos_supervisados/`
- Los artefactos de MLflow se guardan en `mlflow_artifacts/` y `mlflow.db`
- Para usar GPU, asegúrate de tener los drivers NVIDIA y CUDA instalados correctamente

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras, por favor abre un issue en el repositorio.

## 📄 Licencia

Este proyecto utiliza datasets públicos (PTB-XL y MIMIC-IV-ECG) que tienen sus propias licencias. Consulta los archivos LICENSE en cada directorio de dataset.

## 👤 Autor

Tomas V00805

## 📧 Contacto

Para preguntas sobre el proyecto, abre un issue en GitHub.

---

**Nota**: Este proyecto requiere acceso a los datasets PTB-XL y MIMIC-IV-ECG, que deben descargarse por separado desde PhysioNet (requiere registro y aceptación de términos de uso).

