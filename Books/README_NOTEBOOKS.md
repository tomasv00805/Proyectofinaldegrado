# 📓 Resumen de Notebooks

Este documento proporciona un resumen rápido de cada notebook en el proyecto para facilitar la navegación y selección del notebook adecuado.

> 📖 Para documentación completa, ver [DOCUMENTACION_GENERAL.md](DOCUMENTACION_GENERAL.md)

---

## 📊 Construcción de Datos

### `build_supervised_ecg_dataset.ipynb`

**Propósito**: Crear dataset binario supervisado (NORMAL vs ANÓMALO) desde datos crudos.

**Input**: 
- PTB-XL (archivos `.hea`, `.dat`, CSV)
- MIMIC-IV-ECG (archivos `.hea`, `.dat`, CSV)

**Output**: 
- `data/Datos_supervisados/` con arrays numpy y tensores PyTorch
- Splits train/val/test balanceados (70/15/15)
- 10 folds estratificados para cross-validation

**Tiempo estimado**: 1-2 horas

**Requisitos**: 
- Datasets PTB-XL y MIMIC-IV-ECG descargados
- ~50GB espacio en disco

**Documentación detallada**: Ver `Documentacion Datos Supervisados.md`

---

### `build_unsupervised_ecg_dataset.ipynb`

**Propósito**: Crear dataset para entrenamiento de autoencoders (solo normales en train).

**Input**: 
- Datos ya procesados desde `Datos_supervisados/`

**Output**: 
- `data/Datos_no_supervisados/`
- Train: Solo normales (label=0)
- Val/Test: Mezcla de normales y anómalos (con labels)

**Tiempo estimado**: 10-20 minutos

**Requisitos**: 
- Dataset supervisado ya construido

---

## 🔽 Downsampling

### `downsample_supervised_data.ipynb`

**Propósito**: Reducir frecuencia de muestreo de datos supervisados de 500Hz → 200Hz.

**Input**: 
- `Datos_supervisados/numpy/` y `Datos_supervisados/tensors/` (500Hz)

**Output**: 
- `Datos_supervisados/numpy_200hz/` y `Datos_supervisados/tensors_200hz/` (200Hz)
- Reduce de 5000 → 2000 muestras por señal

**Tiempo estimado**: 30-60 minutos

**Ventajas**: 
- Reduce tamaño de archivos ~2.5x
- Acelera entrenamiento
- Mantiene preprocesado original

---

### `downsample_unsupervised_data.ipynb`

**Propósito**: Reducir frecuencia de muestreo de datos no supervisados de 500Hz → 200Hz y convertir a tensores.

**Input**: 
- `Datos_no_supervisados/numpy/` (500Hz)

**Output**: 
- `Datos_no_supervisados/numpy_200hz/` (200Hz)
- `Datos_no_supervisados/tensors_200hz/` (tensores PyTorch listos)

**Tiempo estimado**: 30-60 minutos

**Características**: 
- Guardado incremental constante (checkpoints cada 5 chunks)
- Genera tensores listos para entrenamiento

---

## 🧠 Clasificación Supervisada

### `cnn1d_classification_supervised.ipynb`

**Propósito**: Clasificación binaria con CNN1D puro.

**Arquitectura**: CNN1D para extracción de características locales.

**Input**: 
- `Datos_supervisados/tensors_200hz/` (archivos `.pt`)

**Output**: 
- Modelo entrenado guardado en `models/`
- Métricas en MLflow
- Gráficos de entrenamiento y evaluación

**Tiempo estimado**: 2-3 horas

**GPU**: Recomendada (RTX 5080 compatible)

**Recomendado para**: Baseline rápido, comparación con otros modelos

---

### `cnn1d_lstm_classification_supervised.ipynb` ⭐ **RECOMENDADO**

**Propósito**: Clasificación binaria con arquitectura híbrida CNN1D + LSTM.

**Arquitectura**: CNN1D para características locales + LSTM para dependencias temporales.

**Input**: 
- `Datos_supervisados/tensors_200hz/` (archivos `.pt`)

**Output**: 
- Modelo entrenado guardado en `models/`
- Métricas en MLflow
- Gráficos de entrenamiento y evaluación

**Tiempo estimado**: 2-4 horas

**GPU**: Recomendada (RTX 5080 compatible)

**Recomendado para**: **Uso general** - mejor balance rendimiento/complejidad

**Ventajas**: 
- Combina lo mejor de CNN y LSTM
- Buen rendimiento sin excesiva complejidad

---

### `cnn1d_transformer_classification_supervised.ipynb`

**Propósito**: Clasificación binaria con CNN1D + Transformer.

**Arquitectura**: CNN1D + Transformer con self-attention para relaciones globales.

**Input**: 
- `Datos_supervisados/tensors_200hz/` (archivos `.pt`)

**Output**: 
- Modelo entrenado guardado en `models/`
- Métricas en MLflow
- Gráficos de entrenamiento y evaluación

**Tiempo estimado**: 3-5 horas

**GPU**: Requerida (más lento que otros modelos)

**Recomendado para**: Máximo rendimiento cuando el tiempo no es limitante

**Ventajas**: 
- Mejor rendimiento potencial
- Captura dependencias complejas globales

---

### `lstm_classification_supervised.ipynb`

**Propósito**: Clasificación binaria con LSTM puro.

**Arquitectura**: Múltiples capas LSTM para secuencias temporales.

**Input**: 
- `Datos_supervisados/tensors_200hz/` (archivos `.pt`)

**Output**: 
- Modelo entrenado guardado en `models/`
- Métricas en MLflow
- Gráficos de entrenamiento y evaluación

**Tiempo estimado**: 2-4 horas

**GPU**: Recomendada

**Recomendado para**: Comparación con arquitecturas híbridas

---

## 🔍 Detección de Anomalías (Autoencoders)

### `cnn1d_autoencoder_anomaly_detection.ipynb`

**Propósito**: Detección de anomalías con autoencoder CNN1D puro.

**Arquitectura**: Encoder-decoder CNN1D.

**Entrenamiento**: Solo con ejemplos normales (no supervisado)

**Input**: 
- `Datos_no_supervisados/tensors_200hz/` (archivos `.pt`)
- Train: Solo normales
- Val/Test: Mezcla con labels

**Output**: 
- Modelo autoencoder entrenado
- Umbral óptimo seleccionado automáticamente
- Métricas de detección (precision, recall, F1)
- Gráficos de distribución de errores

**Tiempo estimado**: 2-3 horas

**GPU**: Recomendada

**Recomendado para**: Baseline rápido para detección de anomalías

---

### `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` ⭐ **RECOMENDADO**

**Propósito**: Detección de anomalías con autoencoder híbrido CNN1D + LSTM.

**Arquitectura**: Encoder-decoder híbrido (CNN1D + LSTM).

**Entrenamiento**: Solo con ejemplos normales (no supervisado)

**Input**: 
- `Datos_no_supervisados/tensors_200hz/` (archivos `.pt`)
- Train: Solo normales
- Val/Test: Mezcla con labels

**Output**: 
- Modelo autoencoder entrenado
- Umbral óptimo seleccionado automáticamente
- Métricas de detección (precision, recall, F1)
- Gráficos de distribución de errores

**Tiempo estimado**: 2-4 horas

**GPU**: Recomendada

**Recomendado para**: **Uso general** - mejor balance rendimiento/complejidad para detección de anomalías

**Ventajas**: 
- Mejor captura de patrones temporales complejos
- Reconstrucción más precisa que CNN puro

---

### `lstm_autoencoder_pipeline.ipynb`

**Propósito**: Detección de anomalías con autoencoder LSTM puro.

**Arquitectura**: Encoder-decoder LSTM.

**Entrenamiento**: Solo con ejemplos normales (no supervisado)

**Input**: 
- `Datos_no_supervisados/tensors_200hz/` (archivos `.pt`)
- Train: Solo normales
- Val/Test: Mezcla con labels

**Output**: 
- Modelo autoencoder entrenado
- Umbral óptimo seleccionado automáticamente
- Métricas de detección (precision, recall, F1)
- Gráficos de distribución de errores

**Tiempo estimado**: 2-4 horas

**GPU**: Recomendada

**Recomendado para**: Comparación con arquitecturas híbridas

---

## 🔄 Flujo de Trabajo Recomendado

### Para Clasificación Supervisada

1. ✅ `build_supervised_ecg_dataset.ipynb` - Construir datos
2. ✅ `downsample_supervised_data.ipynb` - (Opcional) Reducir a 200Hz
3. ✅ `cnn1d_lstm_classification_supervised.ipynb` - Entrenar modelo ⭐

### Para Detección de Anomalías

1. ✅ `build_supervised_ecg_dataset.ipynb` - Construir datos base
2. ✅ `build_unsupervised_ecg_dataset.ipynb` - Preparar datos no supervisados
3. ✅ `downsample_unsupervised_data.ipynb` - (Opcional) Reducir a 200Hz
4. ✅ `cnn1d_lstm_autoencoder_anomaly_detection.ipynb` - Entrenar modelo ⭐

---

## ⚙️ Configuración Común

Todos los notebooks de entrenamiento requieren:

1. **Setup CUDA (Windows)**: Ejecutar celda de setup antes de imports
2. **Configurar DATA_DIR**: Ajustar ruta a tus datos
3. **GPU**: Recomendada para entrenamiento rápido

Ver [DOCUMENTACION_GENERAL.md](DOCUMENTACION_GENERAL.md) para detalles completos.

---

## 📊 Comparación Rápida

| Notebook | Tipo | Arquitectura | Tiempo | Recomendado |
|----------|------|--------------|--------|-------------|
| `cnn1d_classification_supervised` | Clasificación | CNN1D | 2-3h | Baseline |
| `cnn1d_lstm_classification_supervised` | Clasificación | CNN1D+LSTM | 2-4h | ⭐ **Sí** |
| `cnn1d_transformer_classification_supervised` | Clasificación | CNN1D+Transformer | 3-5h | Máximo rendimiento |
| `lstm_classification_supervised` | Clasificación | LSTM | 2-4h | Comparación |
| `cnn1d_autoencoder_anomaly_detection` | Anomalías | CNN1D AE | 2-3h | Baseline |
| `cnn1d_lstm_autoencoder_anomaly_detection` | Anomalías | CNN1D+LSTM AE | 2-4h | ⭐ **Sí** |
| `lstm_autoencoder_pipeline` | Anomalías | LSTM AE | 2-4h | Comparación |

---

**Última actualización**: 2025-01-XX

