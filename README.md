# 🫀 Proyecto Final de Grado - Análisis de ECG con Deep Learning

Sistema completo para el análisis de señales ECG utilizando modelos de deep learning para la detección de anomalías. El proyecto incluye procesamiento de datos, entrenamiento de múltiples arquitecturas (CNN1D, LSTM, Transformer), despliegue en AWS SageMaker, y un frontend React para interactuar con el modelo en tiempo real.

## 📋 Características Principales

- **Procesamiento de señales ECG**: Filtrado, normalización, downsampling, selección de leads (I, II, III)
- **Múltiples arquitecturas de modelos**: CNN1D, CNN1D+LSTM, CNN1D+Transformer, Autoencoders
- **Datos supervisados y no supervisados**: Pipelines completos para ambos enfoques
- **Despliegue en producción**: AWS SageMaker Serverless + Lambda + API Gateway (arquitectura serverless completa)
- **Frontend interactivo**: Aplicación React + Vite para demo y pruebas en tiempo real con visualización de ECG
- **Arquitectura segura**: Sin exposición de credenciales AWS, todo manejado mediante IAM roles
- **Tracking de experimentos**: Integración con MLflow para seguimiento de entrenamientos
- **Análisis comparativo**: Comparación de costos computacionales entre modelos
- **Pipeline completo**: Desde datos crudos hasta modelo en producción con interfaz web

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                      │
│  - Selección de ECG de prueba                           │
│  - Visualización interactiva de señales                 │
│  - Envío de predicciones al modelo                      │
│  - Visualización de resultados                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ HTTP POST /predict
                     │ JSON: {"signals": [[[...]]]}
                     ▼
┌─────────────────────────────────────────────────────────┐
│              API Gateway (HTTP API)                      │
│  - Maneja CORS                                          │
│  - Enrutamiento                                         │
│  - Punto de entrada público                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Invoca función Lambda
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AWS Lambda (Python)                         │
│  - Recibe y valida requests                             │
│  - Invoca SageMaker usando IAM roles                    │
│  - Maneja errores y formatea respuestas                 │
│  - Sin credenciales expuestas                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ boto3.invoke_endpoint()
                     │ (Usando rol IAM)
                     ▼
┌─────────────────────────────────────────────────────────┐
│       SageMaker Endpoint (Serverless Inference)          │
│  - Modelo CNN1D+LSTM                                    │
│  - Procesa ECG en tiempo real                           │
│  - Retorna probabilidad de anomalía                     │
│  - Solo cobra por invocación (serverless)               │
└─────────────────────────────────────────────────────────┘
```

### Componentes de la Arquitectura

1. **Frontend (React + Vite)**
   - Interfaz de usuario moderna y responsive
   - Visualización de señales ECG con Canvas
   - Selección de ECG de prueba desde un conjunto predefinido
   - Envío de predicciones y visualización de resultados
   - Sin credenciales AWS (100% seguro)

2. **API Gateway (HTTP API)**
   - Expone endpoint público `/predict`
   - Maneja CORS para desarrollo local (`localhost:5173`)
   - Enrutamiento a Lambda
   - Tier gratuito hasta 1M requests/mes

3. **Lambda Function (Python)**
   - Función serverless que actúa como proxy seguro
   - Valida formato de entrada (forma `[1, 2000, 3]`)
   - Invoca SageMaker usando credenciales IAM (sin keys expuestas)
   - Maneja errores y formatea respuestas
   - Timeout configurable (recomendado: 30-60 segundos)

4. **SageMaker Endpoint (Serverless)**
   - Modelo de IA desplegado para inferencia
   - Arquitectura CNN1D + LSTM para detección de anomalías
   - Solo cobra por invocación (sin costo cuando está inactivo)
   - Cold start en primera invocación (~5-15 segundos)

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
│   │   ├── App.jsx                 # Componente principal de la aplicación
│   │   ├── App.css                 # Estilos del componente principal
│   │   ├── ECGVisualization.jsx    # Componente de visualización de señales ECG
│   │   ├── api/
│   │   │   └── client.js           # Cliente para comunicarse con API Gateway
│   │   ├── data/
│   │   │   └── ecg_samples.json    # Ejemplos de ECG para demo (generado)
│   │   ├── main.jsx                # Punto de entrada de React
│   │   └── index.css               # Estilos globales
│   ├── lambda_function.py          # Función Lambda para AWS (proxy seguro)
│   ├── generate_ecg_samples.py     # Script para generar ejemplos de ECG
│   ├── package.json                # Dependencias Node.js
│   ├── vite.config.js              # Configuración de Vite
│   ├── .env.example                # Plantilla de variables de entorno
│   ├── README.md                   # Guía básica del frontend
│   ├── DOCUMENTACION_COMPLETA.md   # Documentación técnica completa
│   ├── INICIO_RAPIDO.md            # Checklist rápido de configuración
│   ├── INSTRUCCIONES_AWS.md        # Guía paso a paso para AWS Lambda + API Gateway
│   ├── TROUBLESHOOTING.md          # Solución de problemas comunes
│   ├── VERIFICAR_INTEGRACION.md    # Guía para verificar la integración
│   └── SOLUCION_CORS_HTTP_API.md   # Guía específica para problemas CORS
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

## 🔧 Uso del Sistema Completo

### Flujo de Trabajo Completo

1. **Preparar Datos** → 2. **Entrenar Modelo** → 3. **Desplegar en AWS** → 4. **Configurar Frontend** → 5. **Usar Aplicación**

---

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

#### 4.1. Desplegar Modelo en SageMaker

Ver la guía completa en `Books/deploy_sagemaker_serverless.ipynb` o `Books/DOCUMENTACION_DESPLIEGUE_SAGEMAKER.md`

**Pasos principales:**
1. Preparar modelo para SageMaker (crear `.tar.gz` con código de inferencia)
2. Subir modelo a S3
3. Crear endpoint serverless en SageMaker
4. Probar endpoint localmente
5. Obtener nombre del endpoint (necesario para Lambda)

#### 4.2. Configurar Lambda + API Gateway

Ver guía completa en `Frontend/INSTRUCCIONES_AWS.md`

**Pasos principales:**
1. **Crear Rol IAM** para Lambda con permisos `sagemaker:InvokeEndpoint`
2. **Crear función Lambda** con código de `Frontend/lambda_function.py`
3. **Configurar variable de entorno** `SAGEMAKER_ENDPOINT` en Lambda
4. **Ajustar timeout** de Lambda (mínimo 30 segundos, recomendado 60)
5. **Crear API Gateway HTTP API** con ruta `POST /predict`
6. **Configurar integración** Lambda en API Gateway
7. **Habilitar CORS** en API Gateway para `http://localhost:5173`
8. **Obtener URL** de API Gateway (para configurar en frontend)

**Documentación detallada:**
- `Frontend/INSTRUCCIONES_AWS.md` - Guía paso a paso con capturas
- `Frontend/DOCUMENTACION_COMPLETA.md` - Explicación técnica completa
- `Frontend/TROUBLESHOOTING.md` - Solución de problemas comunes

### 5. Configurar AWS (Despliegue Completo)

Para desplegar el sistema completo, necesitas configurar AWS Lambda y API Gateway:

**Guía completa paso a paso:** Ver `Frontend/INSTRUCCIONES_AWS.md`

**Resumen rápido:**

1. **Crear Rol IAM** para Lambda con permisos `sagemaker:InvokeEndpoint`
2. **Crear función Lambda** (`Frontend/lambda_function.py`) con el rol creado
3. **Configurar variable de entorno** en Lambda: `SAGEMAKER_ENDPOINT=tu-endpoint-name`
4. **Crear API Gateway HTTP API** conectado a Lambda con ruta `POST /predict`
5. **Habilitar CORS** en API Gateway para `http://localhost:5173`
6. **Obtener URL** de API Gateway (ej: `https://xxxxx.execute-api.us-east-1.amazonaws.com`)

**Documentación detallada:**
- `Frontend/INSTRUCCIONES_AWS.md` - Guía paso a paso completa
- `Frontend/DOCUMENTACION_COMPLETA.md` - Documentación técnica completa
- `Books/DOCUMENTACION_DESPLIEGUE_SAGEMAKER.md` - Despliegue del modelo en SageMaker

### 6. Usar el Frontend

```bash
cd Frontend
npm install
```

**Configurar API Gateway:**
1. Crea `.env` desde `.env.example` (o créalo manualmente)
2. Agrega tu URL de API Gateway:
   ```env
   VITE_API_URL=https://tu-api.execute-api.us-east-1.amazonaws.com
   ```
   **Nota:** NO incluyas `/predict` al final, se agrega automáticamente.

3. **Ejecutar en desarrollo:**
   ```bash
   npm run dev
   ```

4. Abre `http://localhost:5173` en tu navegador

**Uso del Frontend:**
1. **Seleccionar ECG:** Haz click en una de las tarjetas de ECG para seleccionarla
   - Las tarjetas muestran si el ECG es NORMAL o ANÓMALO (etiqueta real)
   - Cada tarjeta muestra nombre, descripción y forma de datos
2. **Ver visualización:** El ECG seleccionado se muestra gráficamente con sus 3 canales
   - Visualización interactiva usando Canvas API
   - Tres gráficos superpuestos (uno por canal: I, II, III)
   - Responsive y adaptable al tamaño de ventana
3. **Enviar al modelo:** Click en "🚀 Enviar a Modelo"
   - El botón se deshabilita mientras procesa
   - Muestra indicador de carga durante la predicción
   - Primera invocación puede tardar más (cold start de SageMaker)
4. **Ver resultado:** 
   - **Resumen amigable**: Predicción (Normal/Anómalo), probabilidad y confianza (%)
   - **JSON raw**: Respuesta completa del modelo para depuración
   - **Comparación**: Si el ECG tenía etiqueta, compara predicción vs real (✅/❌)

**Características del Frontend:**
- ✅ Interfaz moderna y responsive
- ✅ Visualización en tiempo real de señales ECG
- ✅ Validación de configuración (verifica si API URL está configurada)
- ✅ Manejo robusto de errores con mensajes claros
- ✅ Indicadores de estado (loading, success, error)
- ✅ Diseño oscuro optimizado para visualización de señales
- ✅ Sin dependencias pesadas (React puro + Vite)

**Generar ECG de ejemplo (opcional):**
```bash
cd Frontend
python generate_ecg_samples.py
```
Este script regenera `src/data/ecg_samples.json` desde los datos de entrenamiento.

**Documentación del Frontend:**
- `Frontend/README.md` - Guía de uso y configuración
- `Frontend/DOCUMENTACION_COMPLETA.md` - Documentación técnica completa
- `Frontend/INICIO_RAPIDO.md` - Checklist rápido de configuración
- `Frontend/TROUBLESHOOTING.md` - Solución de problemas comunes

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
- **React 18**: Framework UI moderno con hooks
- **Vite**: Build tool rápido y dev server (HMR)
- **JavaScript/JSX**: Lenguaje principal
- **Canvas API**: Visualización de señales ECG
- **Fetch API**: Comunicación con API Gateway

### Despliegue
- **AWS SageMaker Serverless Inference**: Servicio de ML para endpoints serverless (sin costo cuando inactivo)
- **AWS Lambda (Python 3.11)**: Función serverless como proxy seguro
- **API Gateway HTTP API**: API HTTP para exponer el modelo públicamente
- **IAM Roles**: Gestión de permisos y seguridad (sin credenciales expuestas)
- **CloudWatch**: Logging y monitoreo

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
- `Books/DOCUMENTACION_DESPLIEGUE_SAGEMAKER.md` - Guía completa de despliegue del modelo en SageMaker
- `Frontend/README.md` - Guía básica del frontend
- `Frontend/DOCUMENTACION_COMPLETA.md` - Documentación técnica completa del sistema frontend + backend
- `Frontend/INICIO_RAPIDO.md` - Checklist rápido para poner en marcha el sistema
- `Frontend/INSTRUCCIONES_AWS.md` - Guía paso a paso detallada para configurar Lambda + API Gateway
- `Frontend/TROUBLESHOOTING.md` - Solución de problemas comunes
- `Frontend/VERIFICAR_INTEGRACION.md` - Guía para verificar que todo funciona correctamente
- `Frontend/SOLUCION_CORS_HTTP_API.md` - Guía específica para resolver problemas de CORS

## 🔐 Seguridad

- ✅ **Sin credenciales expuestas**: Las credenciales AWS se manejan mediante IAM roles en Lambda
- ✅ **Arquitectura segura**: Frontend → API Gateway → Lambda → SageMaker (credenciales solo en Lambda)
- ✅ **API Gateway como proxy**: Todas las peticiones pasan por API Gateway (punto de entrada controlado)
- ✅ **CORS configurado**: Control de acceso desde el frontend (configurable por dominio)
- ✅ **Variables de entorno**: Configuración sensible en `.env` (no incluido en repo, ver `.env.example`)
- ✅ **Validación de entrada**: Validación exhaustiva de formato en Lambda antes de invocar SageMaker
- ✅ **Manejo de errores**: Errores manejados sin exponer información sensible
- ✅ **IAM Roles**: Permisos granulares usando roles IAM (principio de menor privilegio)

**¿Por qué esta arquitectura es segura?**
- El frontend NO contiene credenciales AWS
- Lambda usa un rol IAM para autenticarse automáticamente con SageMaker
- API Gateway proporciona un punto de entrada controlado y configurable
- CORS limita qué dominios pueden hacer requests

## 📝 Notas Importantes

### Datos y Modelos
- Los **datasets originales** y **modelos entrenados** no están en el repositorio debido a su tamaño
- Los datos procesados se guardan en `data/` (no incluido en repo)
- Los artefactos de MLflow se guardan en `mlflow_artifacts/` y `mlflow.db` (no incluidos)
- Los modelos preparados para SageMaker están en `Books/sagemaker_models/` (no incluidos en repo)

### Requisitos Técnicos
- Para usar GPU, asegúrate de tener drivers NVIDIA y CUDA instalados correctamente
- El frontend requiere configuración de API Gateway para funcionar (ver `Frontend/INSTRUCCIONES_AWS.md`)
- Los datasets PTB-XL y MIMIC-IV-ECG requieren registro en PhysioNet
- Node.js 18+ es requerido para el frontend

### Configuración del Sistema
- **Primera vez configurando AWS**: Sigue la guía `Frontend/INSTRUCCIONES_AWS.md` paso a paso
- **Cold start**: La primera invocación del endpoint de SageMaker puede tardar 5-15 segundos
- **Timeout Lambda**: Configura mínimo 30 segundos (recomendado 60 segundos para evitar timeouts)
- **CORS**: Debe estar habilitado en API Gateway para que el frontend funcione desde `localhost:5173`

### Formato de Datos
- **Input esperado**: Forma `[1, 2000, 3]` = [batch_size, muestras_temporales, canales]
- **2000 muestras**: 10 segundos de señal a 200 Hz
- **3 canales**: Derivaciones I, II, III
- **Normalización**: Los datos deben estar normalizados (Z-score) antes de enviar al modelo

## 📊 Resultados y Métricas

Los modelos se evalúan con:
- **Métricas de clasificación**: Accuracy, Precision, Recall, F1-Score
- **Métricas de ranking**: ROC-AUC, PR-AUC
- **Visualizaciones**: Matrices de confusión, curvas ROC/PR
- **Análisis de costos**: Comparación de costos computacionales entre modelos

Ver `Books/computational_cost_comparison/` para comparaciones detalladas.

## 💰 Costos Estimados (AWS)

### SageMaker Serverless Inference
- **Por invocación**: ~$0.00022
- **Sin tráfico**: $0 (no hay costo cuando está inactivo)
- **Ejemplos mensuales**:
  - 1,000 invocaciones: $0.22
  - 10,000 invocaciones: $2.20
  - 100,000 invocaciones: $22.00

### Lambda
- **Primeros 1M requests/mes**: Gratis
- **Después**: $0.20 por 1M requests
- **Ejemplo**: 10,000 requests = $0.00 (dentro del tier gratuito)

### API Gateway (HTTP API)
- **Primeros 1M requests/mes**: Gratis
- **Después**: $1.00 por 1M requests
- **Ejemplo**: 10,000 requests = $0.00 (dentro del tier gratuito)

**Total estimado para demo/pruebas**: Prácticamente $0 (dentro de tier gratuito de AWS)

**Nota**: Los costos reales pueden variar según la región y uso. Consulta la [calculadora de AWS](https://calculator.aws/) para estimaciones precisas.

## 🎯 Modelos Recomendados

Para **clasificación supervisada**: `cnn1d_lstm_classification_supervised.ipynb` ⭐
- Mejor balance entre rendimiento y costo computacional
- Arquitectura CNN1D + LSTM

Para **detección de anomalías**: `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` ⭐
- Autoencoder híbrido CNN1D + LSTM
- Buen rendimiento en detección de anomalías

## 🤝 Contribuciones

Este es un proyecto académico. Para sugerencias o mejoras, por favor abre un issue en el repositorio.

## 🔄 Flujo Completo de una Predicción

### Ejemplo de Uso End-to-End

1. **Usuario selecciona ECG** en el frontend
   - Frontend carga datos del ECG seleccionado
   - Muestra visualización de los 3 canales

2. **Usuario envía al modelo**
   - Frontend prepara request: `{"signals": [[[...]]]}` con forma `[1, 2000, 3]`
   - Envía `POST` a `https://api-gateway-url/predict`

3. **API Gateway recibe request**
   - Valida CORS (si viene de `localhost:5173`)
   - Enruta a Lambda

4. **Lambda procesa**
   - Valida formato del JSON
   - Verifica forma `[1, 2000, 3]`
   - Prepara payload para SageMaker
   - Invoca endpoint usando `boto3` con credenciales IAM

5. **SageMaker procesa**
   - Modelo carga si no está cargado (cold start, primera vez)
   - Procesa ECG a través de CNN1D + LSTM
   - Retorna probabilidad de anomalía: `{"prediction": 0.85, "probability": 0.85}`

6. **Lambda formatea respuesta**
   - Agrega headers CORS
   - Retorna JSON al API Gateway

7. **Frontend muestra resultado**
   - Interpreta probabilidad (> 0.5 = Anómalo)
   - Muestra resumen amigable (predicción, confianza)
   - Compara con etiqueta real si está disponible
   - Muestra JSON raw de la respuesta

**Tiempo total**: 2-5 segundos (10-15 segundos en primera invocación por cold start)

## 📄 Licencia

Este proyecto utiliza datasets públicos (PTB-XL y MIMIC-IV-ECG) que tienen sus propias licencias. Consulta los archivos LICENSE en cada directorio de dataset.

## 👤 Autor

**Tomas V00805**

## 📧 Contacto

Para preguntas sobre el proyecto, abre un issue en GitHub.

---

**Nota**: Este proyecto requiere acceso a los datasets PTB-XL y MIMIC-IV-ECG, que deben descargarse por separado desde PhysioNet (requiere registro y aceptación de términos de uso).
