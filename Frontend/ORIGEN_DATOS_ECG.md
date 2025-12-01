# 📊 Origen de los Datos ECG para la Demo

## 📍 Ubicación de los Datos

Los ECG de prueba que se muestran en el frontend provienen de los datos de test del dataset supervisado:

```
data/Datos_supervisados/tensors_200hz/
├── X_test.pt    # Señales ECG (tensores PyTorch)
├── y_test.pt    # Etiquetas (0=normal, 1=anómalo)
├── X_train.pt
├── y_train.pt
├── X_val.pt
└── y_val.pt
```

## 🔄 Proceso de Generación

### 1. **Datos Originales**
Los datos provienen del dataset **MIMIC-IV ECG Diagnostic Electrocardiogram Matched Subset**, que se encuentra en:
```
mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/
```

### 2. **Pipeline de Procesamiento**
Los datos fueron procesados mediante scripts en `Books/`:
- `build_supervised_ecg_dataset.py` o `build_supervised_ecg_dataset.ipynb`
- Procesamiento de señales ECG crudas a formato tensorial
- Downsampling a 200 Hz (10 segundos = 2000 muestras)
- Extracción de 3 canales: I, II, III
- Normalización y preprocesamiento
- Creación de splits: train/val/test

### 3. **Formato de los Datos**
Cada ECG está en formato tensorial:
- **Forma**: `[2000, 3]`
  - 2000 muestras temporales (10 segundos a 200 Hz)
  - 3 canales (I, II, III)
- **Tipo**: `torch.Tensor` guardado como `.pt`
- **Etiquetas**: 
  - `0` = NORMAL
  - `1` = ANÓMALO

### 4. **Generación de Muestras para el Frontend**
El script `generate_ecg_samples.py` en `Frontend/`:
1. Carga `X_test.pt` y `y_test.pt`
2. Selecciona un número configurable de ECGs normales y anómalos
3. Convierte los tensores a listas JSON
4. Guarda en `Frontend/src/data/ecg_samples.json`

## 🚀 Cómo Generar Más Muestras

### Opción 1: Modificar el Script

Edita `generate_ecg_samples.py` y cambia estas variables:

```python
NUM_NORMAL = 10      # Aumentar para más ECGs normales
NUM_ANOMALO = 10     # Aumentar para más ECGs anómalos
MAX_SEARCH = 2000    # Buscar en más ECGs del dataset
```

Luego ejecuta:
```bash
cd Frontend
python generate_ecg_samples.py
```

### Opción 2: Ejecutar desde la Línea de Comandos con Parámetros

Puedes modificar el script para aceptar argumentos:

```bash
cd Frontend
python generate_ecg_samples.py --normal 15 --anomalo 15
```

## 📋 Estructura del Archivo `ecg_samples.json`

```json
{
  "samples": [
    {
      "id": "normal_0",
      "name": "ECG Normal #1",
      "signals": [[...], [...], ...],  // 2000 arrays de 3 valores
      "label": 0,
      "label_text": "NORMAL",
      "description": "...",
      "dataset_index": 42
    },
    ...
  ],
  "metadata": {
    "total_samples": 16,
    "normal_samples": 8,
    "anomalo_samples": 8,
    "generated_from": {...},
    "format": {...}
  }
}
```

## 🔍 Verificación de Datos

Para verificar que los datos existen:

```bash
# Desde la raíz del proyecto
cd Frontend
python -c "
import torch
from pathlib import Path
data_dir = Path('../data/Datos_supervisados/tensors_200hz')
X_test = torch.load(data_dir / 'X_test.pt', map_location='cpu')
y_test = torch.load(data_dir / 'y_test.pt', map_location='cpu')
print(f'Total ECGs: {len(X_test)}')
print(f'Forma: {X_test.shape}')
print(f'Normales: {(y_test == 0).sum()}')
print(f'Anómalos: {(y_test == 1).sum()}')
"
```

## 📝 Para la Demo

### Checklist Pre-Demo:

- [ ] ✅ Verificar que existen los archivos `.pt`:
  - `data/Datos_supervisados/tensors_200hz/X_test.pt`
  - `data/Datos_supervisados/tensors_200hz/y_test.pt`

- [ ] ✅ Generar suficientes muestras:
  ```bash
  cd Frontend
  python generate_ecg_samples.py
  ```

- [ ] ✅ Verificar que se generó `Frontend/src/data/ecg_samples.json`

- [ ] ✅ Verificar el tamaño del archivo (debe ser < 50 MB para cargar rápido)

- [ ] ✅ Reiniciar el servidor de desarrollo:
  ```bash
  npm run dev
  ```

### Recomendaciones:

1. **Cantidad de muestras**: Para una demo, 8-12 muestras de cada tipo es suficiente (total 16-24)
   - Más muestras = archivo JSON más grande = carga más lenta
   - Menos muestras = menos variedad en la demo

2. **Distribución**: Asegúrate de tener al menos 3-4 de cada tipo para demostrar ambos casos

3. **Tamaño del archivo**: Si el JSON es > 50 MB, considera reducir el número de muestras

## 🔄 Actualización de Muestras

Si necesitas actualizar las muestras después de cambios en el dataset:

```bash
# 1. Asegúrate de tener los datos actualizados en data/
# 2. Regenera las muestras
cd Frontend
python generate_ecg_samples.py

# 3. Reinicia el frontend
# (Ctrl+C para detener, luego npm run dev)
```

## 📚 Referencias

- **Dataset MIMIC-IV**: https://physionet.org/content/mimic-iv-ecg/
- **Documentación del Proyecto**: Ver `Books/DOCUMENTACION_GENERAL.md`
- **Pipeline de Datos**: Ver `Books/Documentacion Datos Supervisados.md`

