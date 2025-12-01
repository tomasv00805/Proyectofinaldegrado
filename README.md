# 🫀 Proyecto Final de Grado - Análisis de ECG con Deep Learning

Sistema completo para el análisis de señales ECG utilizando modelos de deep learning para la detección de anomalías. El proyecto incluye procesamiento de datos, entrenamiento de múltiples arquitecturas (CNN1D, LSTM, Transformer), despliegue en AWS SageMaker, y un frontend React para interactuar con el modelo.

## 📋 Características Principales

- **Procesamiento de señales ECG**: Filtrado, normalización, downsampling, selección de leads (I, II, III)
- **Múltiples arquitecturas de modelos**: CNN1D, CNN1D+LSTM, CNN1D+Transformer, Autoencoders
- **Datos supervisados y no supervisados**: Pipelines completos para ambos enfoques
- **Despliegue en producción**: AWS SageMaker Serverless + Lambda + API Gateway
- **Frontend interactivo**: Aplicación React + Vite para demo y pruebas en tiempo real
- **Tracking de experimentos**: Integración con MLflow para seguimiento de entrenamientos
- **Análisis comparativo**: Comparación de costos computacionales entre modelos
- **Pipeline completo**: Desde datos crudos hasta modelo en producción

## 🚀 Inicio Rápido

### Requisitos Previos

- **Python 3.8+** para el backend/ML
- **Node.js 18+** para el frontend (opcional)
- **CUDA 12.8+** (opcional, para aceleración GPU)
- **Cuenta AWS** (para despliegue en producción)
- **Git**

### Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tomasv00805/Proyectofinaldegrado.git
cd Proyectofinaldegrado
```

2. **Configurar entorno Python:**
```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate
# Activar (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Instalar PyTorch con CUDA (opcional):**
```bash
# CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Solo CPU
pip install torch torchvision
```

4. **Configurar frontend (opcional):**
```bash
cd Frontend
npm install
cp .env.example .env
# Editar .env y agregar tu URL de API Gateway cuando esté configurada
```

## 📁 Estructura del Proyecto

```
Proyectofinaldegrado/
├── Books/                          # Scripts y notebooks de ML
│   ├── build_supervised_ecg_dataset.py      # Pipeline datos supervisados
│   ├── build_unsupervised_ecg_dataset.ipynb  # Pipeline datos no supervisados
│   ├── cnn1d_classification_supervised.ipynb
│   ├── cnn1d_lstm_classification_supervised.ipynb ⭐
│   ├── cnn1d_transformer_classification_supervised.ipynb
│   ├── cnn1d_autoencoder_anomaly_detection.ipynb
│   ├── cnn1d_lstm_autoencoder_anomaly_detection.ipynb ⭐
│   ├── lstm_autoencoder_pipeline.ipynb
│   ├── deploy_sagemaker_serverless.ipynb    # Despliegue en AWS
│   ├── evaluation_threshold_tuning.py       # Evaluación de modelos
│   ├── ecg_preprocessing.py                 # Funciones de preprocesamiento
│   ├── models/                              # Metadatos de modelos entrenados
│   ├── sagemaker_models/                    # Modelos preparados para SageMaker
│   ├── DOCUMENTACION_*.md                   # Documentación técnica completa
│   └── README_NOTEBOOKS.md                  # Guía detallada de notebooks
│
├── Frontend/                       # Aplicación web React
│   ├── src/
│   │   ├── App.jsx                 # Componente principal
│   │   ├── ECGVisualization.jsx    # Visualización de señales ECG
│   │   ├── api/client.js           # Cliente API Gateway
│   │   └── data/ecg_samples.json   # Ejemplos de ECG para demo
│   ├── lambda_function.py          # Función Lambda para AWS
│   ├── generate_ecg_samples.py     # Generar ejemplos de ECG
│   ├── package.json
│   ├── vite.config.js
│   └── README.md                   # Documentación del frontend
│
├── config/                         # Configuraciones
│   └── ae1d_config.json           # Configuración autoencoder
│
├── data/                           # Datos procesados (no en repo)
│   ├── Datos_supervisados/        # Datasets supervisados
│   └── Datos_no_supervisados/     # Datasets no supervisados
│
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## 🔧 Uso

### 1. Preparar los Datos

**Descargar datasets:**
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg-diagnostic/1.0/

**Colocar en el directorio raíz:**
```
ptb-xl-a-large-publicly-available-electrocardiogram-dataset-1.0.3/
mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/
```

**Procesar datos supervisados:**
```bash
cd Books
python build_supervised_ecg_dataset.py
# O usar el notebook interactivo:
jupyter notebook build_supervised_ecg_dataset.ipynb
```

**Procesar datos no supervisados:**
```bash
jupyter notebook build_unsupervised_ecg_dataset.ipynb
```

### 2. Entrenar Modelos

**Clasificación Supervisada (Recomendado):**
- `cnn1d_lstm_classification_supervised.ipynb` ⭐ - CNN1D + LSTM (mejor rendimiento)
- `cnn1d_classification_supervised.ipynb` - CNN1D puro
- `cnn1d_transformer_classification_supervised.ipynb` - CNN1D + Transformer

**Detección de Anomalías - No Supervisado (Recomendado):**
- `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` ⭐ - Autoencoder CNN1D + LSTM
- `cnn1d_autoencoder_anomaly_detection.ipynb` - Autoencoder CNN1D
- `lstm_autoencoder_pipeline.ipynb` - Autoencoder LSTM

**Comparar modelos:**
```bash
jupyter notebook compare_models_computational_cost.ipynb
```

### 3. Evaluar Modelos

```bash
python Books/evaluation_threshold_tuning.py
```

### 4. Desplegar en AWS

Ver la guía completa en `Books/deploy_sagemaker_serverless.ipynb` o `Books/DOCUMENTACION_DESPLIEGUE_SAGEMAKER.md`

**Pasos principales:**
1. Preparar modelo para SageMaker
2. Crear endpoint serverless en SageMaker
3. Configurar Lambda function (`Frontend/lambda_function.py`)
4. Crear API Gateway HTTP API
5. Configurar CORS

### 5. Usar el Frontend

```bash
cd Frontend
npm install
npm run dev
```

Abre `http://localhost:5173` en tu navegador.

**Configurar API Gateway:**
1. Crea `.env` desde `.env.example`
2. Agrega tu URL de API Gateway: `VITE_API_URL=https://tu-api.execute-api.us-east-1.amazonaws.com`
3. Reinicia el servidor de desarrollo

Ver `Frontend/README.md` para más detalles.

## 📊 Arquitecturas de Modelos

### Clasificación Supervisada

1. **CNN1D**: Red convolucional 1D pura para extracción de características
2. **CNN1D + LSTM**: Convolución seguida de capas LSTM para capturar dependencias temporales
3. **CNN1D + Transformer**: Convolución con atención Transformer para relaciones de largo alcance

### Detección de Anomalías (No Supervisado)

1. **Autoencoder CNN1D**: Encoder-decoder convolucional para reconstrucción
2. **Autoencoder LSTM**: Encoder-decoder con LSTM para secuencias temporales
3. **Autoencoder CNN1D + LSTM**: Arquitectura híbrida (recomendada)

### Formato de Entrada
- **Forma**: `[batch_size, 2000, 3]`
  - 2000 muestras temporales (10 segundos a 200 Hz)
  - 3 canales (I, II, III)
- **Frecuencia**: 200 Hz (downsampled desde 500 Hz)
- **Duración**: 10 segundos
- **Normalización**: Z-score por canal

## 🛠️ Tecnologías Utilizadas

### Backend/ML
- **PyTorch**: Framework de deep learning
- **NumPy, Pandas**: Manipulación y análisis de datos
- **SciPy, WFDB**: Procesamiento de señales ECG
- **Scikit-learn**: Métricas, validación y evaluación
- **MLflow**: Tracking de experimentos y versionado de modelos
- **Prefect**: Orquestación de pipelines de datos

### Frontend
- **React 18**: Framework UI moderno
- **Vite**: Build tool rápido y dev server
- **JavaScript/JSX**: Lenguaje principal

### Despliegue
- **AWS SageMaker**: Servicio de ML para endpoints serverless
- **AWS Lambda**: Función serverless como proxy
- **API Gateway**: API HTTP para exponer el modelo
- **IAM**: Gestión de permisos y seguridad

## 📚 Documentación

### Documentación General
- `Books/DOCUMENTACION_GENERAL.md` - Visión general completa del proyecto
- `Books/README.md` - Guía del backend/ML
- `Books/README_NOTEBOOKS.md` - Descripción detallada de todos los notebooks

### Documentación de Datos
- `Books/Documentacion Datos Supervisados.md` - Pipeline completo de datos supervisados
- `Books/DOCUMENTACION_DATOS_NO_SUPERVISADOS_DOWNSAMPLING.md` - Datos no supervisados y downsampling

### Documentación de Entrenamiento
- `Books/DOCUMENTACION_ENTRENAMIENTO.md` - Proceso de entrenamiento, arquitecturas y MLflow

### Documentación de Despliegue
- `Books/DOCUMENTACION_DESPLIEGUE_SAGEMAKER.md` - Guía completa de despliegue en AWS
- `Frontend/README.md` - Documentación del frontend
- `Frontend/DOCUMENTACION_COMPLETA.md` - Documentación técnica completa del frontend

## 🔐 Seguridad

- ✅ **Sin credenciales expuestas**: Las credenciales AWS se manejan mediante IAM roles
- ✅ **API Gateway como proxy**: Todas las peticiones pasan por API Gateway
- ✅ **CORS configurado**: Control de acceso desde el frontend
- ✅ **Variables de entorno**: Configuración sensible en `.env` (no en repo)
- ✅ **Validación de entrada**: Validación de datos en Lambda antes de invocar SageMaker

## 📝 Notas Importantes

- Los **datasets originales** y **modelos entrenados** no están en el repositorio debido a su tamaño
- Los datos procesados se guardan en `data/` (no incluido en repo)
- Los artefactos de MLflow se guardan en `mlflow_artifacts/` y `mlflow.db` (no incluidos)
- Para usar GPU, asegúrate de tener drivers NVIDIA y CUDA instalados correctamente
- El frontend requiere configuración de API Gateway para funcionar (ver `Frontend/README.md`)
- Los datasets PTB-XL y MIMIC-IV-ECG requieren registro en PhysioNet

## 📊 Resultados y Métricas

Los modelos se evalúan con:
- **Métricas de clasificación**: Accuracy, Precision, Recall, F1-Score
- **Métricas de ranking**: ROC-AUC, PR-AUC
- **Visualizaciones**: Matrices de confusión, curvas ROC/PR
- **Análisis de costos**: Comparación de costos computacionales entre modelos

Ver `Books/computational_cost_comparison/` para comparaciones detalladas.

## 🎯 Modelos Recomendados

Para **clasificación supervisada**: `cnn1d_lstm_classification_supervised.ipynb` ⭐
- Mejor balance entre rendimiento y costo computacional
- Arquitectura CNN1D + LSTM

Para **detección de anomalías**: `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` ⭐
- Autoencoder híbrido CNN1D + LSTM
- Buen rendimiento en detección de anomalías

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras, por favor abre un issue en el repositorio.

## 📄 Licencia

Este proyecto utiliza datasets públicos (PTB-XL y MIMIC-IV-ECG) que tienen sus propias licencias. Consulta los archivos LICENSE en cada directorio de dataset.

## 👤 Autor

**Tomas V00805**

## 📧 Contacto

Para preguntas sobre el proyecto, abre un issue en GitHub.

---

**Nota**: Este proyecto requiere acceso a los datasets PTB-XL y MIMIC-IV-ECG, que deben descargarse por separado desde PhysioNet (requiere registro y aceptación de términos de uso).
