# 🫀 Proyecto de Análisis de ECG con Deep Learning

Sistema completo de análisis de señales ECG utilizando técnicas de aprendizaje profundo para clasificación supervisada y detección de anomalías.

## 📚 Documentación

- **[📖 Documentación General](DOCUMENTACION_GENERAL.md)** - Documentación completa del proyecto, arquitectura, pipelines y guías de uso
- **[🧠 Documentación de Entrenamiento](DOCUMENTACION_ENTRENAMIENTO.md)** - Guía detallada sobre notebooks de entrenamiento, arquitecturas, MLflow y Prefect
- **[📊 Datos No Supervisados y Downsampling](DOCUMENTACION_DATOS_NO_SUPERVISADOS_DOWNSAMPLING.md)** - Pipeline de datos no supervisados, downsampling y conversión a tensores
- **[📓 Resumen de Notebooks](README_NOTEBOOKS.md)** - Resúmenes rápidos de cada notebook para navegación rápida
- **[📊 Documentación Pipeline Supervisado](Documentacion%20Datos%20Supervisados.md)** - Documentación detallada del pipeline de datos supervisados

## 🚀 Inicio Rápido

### 1. Preparar Datos

```bash
# Construir dataset supervisado
jupyter notebook build_supervised_ecg_dataset.ipynb

# (Opcional) Reducir a 200Hz para entrenamiento más rápido
jupyter notebook downsample_supervised_data.ipynb

# Para detección de anomalías: preparar datos no supervisados
jupyter notebook build_unsupervised_ecg_dataset.ipynb
jupyter notebook downsample_unsupervised_data.ipynb
```

### 2. Entrenar Modelos

**Clasificación Supervisada** (recomendado):
```bash
jupyter notebook cnn1d_lstm_classification_supervised.ipynb
```

**Detección de Anomalías** (recomendado):
```bash
jupyter notebook cnn1d_lstm_autoencoder_anomaly_detection.ipynb
```

## 📋 Características Principales

- ✅ **Pipeline completo** desde datos crudos hasta modelos entrenados
- ✅ **Múltiples arquitecturas**: CNN, LSTM, Transformer, Autoencoders
- ✅ **Integración MLflow** para tracking de experimentos
- ✅ **Soporte GPU** (RTX 5080 compatible)
- ✅ **Procesamiento robusto** de señales ECG

## 🏗️ Estructura del Proyecto

```
Books/
├── 📄 DOCUMENTACION_GENERAL.md          # Documentación completa
├── 📄 DOCUMENTACION_ENTRENAMIENTO.md   # Guía detallada de entrenamiento
├── 📄 DOCUMENTACION_DATOS_NO_SUPERVISADOS_DOWNSAMPLING.md  # Datos no supervisados y downsampling
├── 📄 README_NOTEBOOKS.md                # Resúmenes de notebooks
├── 📄 Documentacion Datos Supervisados.md # Pipeline detallado
│
├── 🔧 Módulos Python
│   ├── supervised_ecg_pipeline.py       # Pipeline principal
│   ├── ecg_preprocessing.py             # Preprocesamiento
│   └── evaluation_threshold_tuning.py   # Evaluación
│
├── 📊 Notebooks de Datos
│   ├── build_supervised_ecg_dataset.ipynb
│   ├── build_unsupervised_ecg_dataset.ipynb
│   ├── downsample_supervised_data.ipynb
│   └── downsample_unsupervised_data.ipynb
│
├── 🧠 Notebooks de Clasificación
│   ├── cnn1d_classification_supervised.ipynb
│   ├── cnn1d_lstm_classification_supervised.ipynb ⭐
│   ├── cnn1d_transformer_classification_supervised.ipynb
│   └── lstm_classification_supervised.ipynb
│
└── 🔍 Notebooks de Anomalías
    ├── cnn1d_autoencoder_anomaly_detection.ipynb
    ├── cnn1d_lstm_autoencoder_anomaly_detection.ipynb ⭐
    └── lstm_autoencoder_pipeline.ipynb
```

## ⚙️ Requisitos

- **Python**: 3.11+
- **GPU**: Recomendada (RTX 5080, CUDA 12.8)
- **RAM**: 16GB mínimo, 32GB+ recomendado
- **Disco**: ~50-100GB libres

Los notebooks instalan automáticamente las dependencias necesarias.

## 📖 Guías

- [Documentación General](DOCUMENTACION_GENERAL.md) - Para entender el proyecto completo
- [Documentación de Entrenamiento](DOCUMENTACION_ENTRENAMIENTO.md) - Para entender cómo funcionan los notebooks de entrenamiento
- [Datos No Supervisados y Downsampling](DOCUMENTACION_DATOS_NO_SUPERVISADOS_DOWNSAMPLING.md) - Pipeline de datos no supervisados y downsampling
- [Resumen de Notebooks](README_NOTEBOOKS.md) - Para elegir qué notebook usar
- [Troubleshooting](DOCUMENTACION_GENERAL.md#troubleshooting) - Solución de problemas comunes

## 🎯 Modelos Recomendados

- **Clasificación**: `cnn1d_lstm_classification_supervised.ipynb` ⭐
- **Anomalías**: `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` ⭐

## 📝 Notas

- **Primera vez**: Ejecuta la celda "Setup CUDA" en cualquier notebook y reinicia el kernel
- **Rutas**: Configura `DATA_DIR` en cada notebook según tu estructura
- **GPU**: Los notebooks detectan automáticamente si hay GPU disponible

---

Para más información, consulta la [Documentación General](DOCUMENTACION_GENERAL.md).

