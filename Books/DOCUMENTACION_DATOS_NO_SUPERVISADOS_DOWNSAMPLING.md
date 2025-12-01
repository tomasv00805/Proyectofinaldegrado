# 📊 Documentación: Datos No Supervisados, Downsampling y Conversión a Tensores

Esta documentación explica en detalle cómo se construyen los datasets no supervisados, cómo se realiza el downsampling de datos y cómo se convierten a tensores PyTorch.

## 📋 Tabla de Contenidos

1. [Pipeline de Datos No Supervisados](#pipeline-de-datos-no-supervisados)
2. [Downsampling de Datos Supervisados](#downsampling-de-datos-supervisados)
3. [Downsampling y Conversión a Tensores (No Supervisados)](#downsampling-y-conversión-a-tensores-no-supervisados)
4. [Guía de Uso](#guía-de-uso)
5. [Troubleshooting](#troubleshooting)

---

## 🔍 Pipeline de Datos No Supervisados

**Archivo**: `build_unsupervised_ecg_dataset.ipynb`

**Propósito**: Crear un dataset específico para entrenamiento de autoencoders donde el train contiene solo ejemplos normales.

### Características Clave

1. **Train**: Solo ECG normales (label == 0) - para entrenar el autoencoder
2. **Val/Test**: Mezcla de normales y anómalos (con labels) - para evaluación
3. **Guardado en disco**: Todo se guarda directamente sin cargar arrays grandes en memoria
4. **Funciones de carga**: Funciones reutilizables para notebooks de entrenamiento

### Diferencias con el Dataset Supervisado

| Aspecto | Supervisado | No Supervisado |
|---------|-------------|----------------|
| **Train** | Normales + Anómalos (balanceado 50/50) | Solo normales |
| **Val/Test** | Normales + Anómalos (balanceado) | Normales + Anómalos (distribución natural) |
| **Balanceo** | Sí (downsampling a clase minoritaria) | No (mantiene distribución natural) |
| **Uso** | Clasificación supervisada | Detección de anomalías (autoencoders) |

### Flujo del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CARGAR DATOS DEL PIPELINE SUPERVISADO                    │
│    - X_train, y_train, X_val, y_val, X_test, y_test        │
│    - Metadata completa                                      │
│    - Usa memoria mapeada (mmap_mode='r') para eficiencia   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. SEPARAR DATOS PARA ENTRENAMIENTO NO SUPERVISADO          │
│    │                                                         │
│    ├─→ Extraer normales del train supervisado              │
│    │   - Separar en train (70%) y val/test pool (30%)      │
│    │                                                         │
│    ├─→ Combinar datos para VAL                              │
│    │   - val_sup completo (normales + anómalos)            │
│    │   - + 15% de normales del train_sup                   │
│    │                                                         │
│    ├─→ Combinar datos para TEST                             │
│    │   - test_sup completo (normales + anómalos)           │
│    │   - + 15% de normales del train_sup                   │
│    │                                                         │
│    └─→ Train no supervisado                                 │
│        - Solo normales (70% de normales del train_sup)      │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CREAR METADATOS                                           │
│    - Metadata para train (solo normales)                    │
│    - Metadata para val (combinada)                          │
│    - Metadata para test (combinada)                         │
│    - Metadata completa (concatenada)                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. GUARDAR DATASET EN DISCO                                  │
│    - Arrays numpy: X_train, y_train, X_val, y_val, etc.    │
│    - Metadata: CSV por split y completo                    │
│    - Resumen de estadísticas                                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FUNCIONES DE CARGA REUTILIZABLES                          │
│    - load_unsupervised_train_data()                         │
│    - load_unsupervised_val_test_data()                      │
│    - load_unsupervised_metadata()                           │
└─────────────────────────────────────────────────────────────┘
```

### Configuración

```python
# Directorios
PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data"
UNSUPERVISED_OUTPUT_DIR = DATA_DIR / "Datos_no_supervisados"
SUPERVISED_DATA_DIR = DATA_DIR / "Datos_supervisados"

# Parámetros de splits
VAL_RATIO = 0.15  # 15% para validación
TEST_RATIO = 0.15  # 15% para test
RANDOM_STATE = 42  # Para reproducibilidad

# Parámetros de etiquetado
LABEL_COL = "label"
NORMAL_LABEL_VALUE = 0
ANOMALY_LABEL_VALUE = 1
```

### Paso 1: Cargar Datos del Pipeline Supervisado

**Requisito previo**: El pipeline supervisado debe estar ejecutado primero.

```python
# Rutas de archivos
numpy_dir = SUPERVISED_DATA_DIR / "numpy"
metadata_dir = SUPERVISED_DATA_DIR / "metadata"

# Cargar arrays usando memoria mapeada (eficiente)
X_train_sup = np.load(numpy_dir / "X_train.npy", mmap_mode='r')
y_train_sup = np.load(numpy_dir / "y_train.npy")
X_val_sup = np.load(numpy_dir / "X_val.npy", mmap_mode='r')
y_val_sup = np.load(numpy_dir / "y_val.npy")
X_test_sup = np.load(numpy_dir / "X_test.npy", mmap_mode='r')
y_test_sup = np.load(numpy_dir / "y_test.npy")

# Cargar metadata
metadata_full = pd.read_csv(metadata_dir / "master_labels_full.csv")
```

**Ventajas de `mmap_mode='r'`**:
- No carga todo el array en memoria RAM
- Acceso eficiente a los datos desde disco
- Permite procesar datasets grandes sin problemas de memoria

### Paso 2: Separar Datos para Entrenamiento No Supervisado

#### Estrategia de Separación

1. **Extraer normales del train supervisado**:
   ```python
   train_normal_mask = (y_train_sup == NORMAL_LABEL_VALUE)
   train_normal_indices = np.where(train_normal_mask)[0]
   ```

2. **Separar normales en train (70%) y val/test pool (30%)**:
   ```python
   from sklearn.model_selection import StratifiedShuffleSplit
   
   sss_normales = StratifiedShuffleSplit(
       n_splits=1,
       test_size=0.30,  # 30% para val/test
       random_state=RANDOM_STATE
   )
   
   train_normal_idx_local, val_test_normal_idx_local = next(
       sss_normales.split(np.arange(len(train_normal_indices)), 
                         np.zeros(len(train_normal_indices), dtype=int))
   )
   ```

3. **Combinar datos para VAL**:
   ```python
   # Val = val_sup completo + normales adicionales del train_sup
   X_val_combined = np.concatenate([
       np.array(X_val_sup),  # val_sup completo
       np.array(X_train_sup[val_normal_from_train])  # Normales adicionales
   ], axis=0)
   ```

4. **Combinar datos para TEST**:
   ```python
   # Test = test_sup completo + normales adicionales del train_sup
   X_test_combined = np.concatenate([
       np.array(X_test_sup),  # test_sup completo
       np.array(X_train_sup[test_normal_from_train])  # Normales adicionales
   ], axis=0)
   ```

5. **Train no supervisado (solo normales)**:
   ```python
   X_train_unsup = np.array(X_train_sup[train_normal_idx])
   y_train_unsup = y_train_sup[train_normal_idx].copy()  # Todos son normales
   ```

### Paso 3: Crear Metadatos

```python
# Metadata para train (solo normales)
metadata_train_unsup = metadata_train_sup.iloc[train_normal_idx].copy()
metadata_train_unsup['split'] = 'train'
metadata_train_unsup['unsupervised_split'] = 'train_normal'

# Metadata para val (combinada)
metadata_val_unsup = pd.concat([
    metadata_val_sup.copy(),  # De val_sup
    metadata_train_sup.iloc[val_normal_from_train].copy()  # Normales adicionales
], ignore_index=True)

# Metadata para test (combinada)
metadata_test_unsup = pd.concat([
    metadata_test_sup.copy(),  # De test_sup
    metadata_train_sup.iloc[test_normal_from_train].copy()  # Normales adicionales
], ignore_index=True)
```

### Paso 4: Guardar Dataset en Disco

```python
numpy_dir = UNSUPERVISED_OUTPUT_DIR / "numpy"
metadata_dir = UNSUPERVISED_OUTPUT_DIR / "metadata"

# Guardar arrays numpy
np.save(numpy_dir / "X_train.npy", X_train_unsup)
np.save(numpy_dir / "y_train.npy", y_train_unsup)
np.save(numpy_dir / "X_val.npy", X_val_unsup)
np.save(numpy_dir / "y_val.npy", y_val_unsup)
np.save(numpy_dir / "X_test.npy", X_test_unsup)
np.save(numpy_dir / "y_test.npy", y_test_unsup)

# Guardar metadata
metadata_train_unsup.to_csv(metadata_dir / "metadata_train.csv", index=False)
metadata_val_unsup.to_csv(metadata_dir / "metadata_val.csv", index=False)
metadata_test_unsup.to_csv(metadata_dir / "metadata_test.csv", index=False)

# Metadata completa
metadata_full_unsup = pd.concat([
    metadata_train_unsup,
    metadata_val_unsup,
    metadata_test_unsup,
], ignore_index=True)
metadata_full_unsup.to_csv(metadata_dir / "metadata_full.csv", index=False)
```

### Paso 5: Funciones de Carga Reutilizables

#### `load_unsupervised_train_data()`

```python
def load_unsupervised_train_data(
    path_base: str | Path,
    mmap_mode: str = 'r'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga los datos de entrenamiento (solo normales).
    
    Args:
        path_base: Ruta base del dataset
        mmap_mode: Modo de mapeo de memoria ('r'=readonly, None=cargar completo)
    
    Returns:
        Tupla (X_train, y_train)
    """
    path_base = Path(path_base)
    numpy_dir = path_base / "numpy"
    
    X_train = np.load(numpy_dir / "X_train.npy", mmap_mode=mmap_mode)
    y_train = np.load(numpy_dir / "y_train.npy")
    
    return X_train, y_train
```

#### `load_unsupervised_val_test_data()`

```python
def load_unsupervised_val_test_data(
    path_base: str | Path,
    split: str = "both",  # 'val', 'test', o 'both'
    mmap_mode: str = 'r'
):
    """
    Carga datos de validación y/o test (normales y anómalos).
    
    Args:
        path_base: Ruta base del dataset
        split: Qué split cargar ('val', 'test', o 'both')
        mmap_mode: Modo de mapeo de memoria
    
    Returns:
        Si split='both': (X_val, y_val, X_test, y_test)
        Si split='val': (X_val, y_val)
        Si split='test': (X_test, y_test)
    """
    path_base = Path(path_base)
    numpy_dir = path_base / "numpy"
    
    if split == "both":
        X_val = np.load(numpy_dir / "X_val.npy", mmap_mode=mmap_mode)
        y_val = np.load(numpy_dir / "y_val.npy")
        X_test = np.load(numpy_dir / "X_test.npy", mmap_mode=mmap_mode)
        y_test = np.load(numpy_dir / "y_test.npy")
        return X_val, y_val, X_test, y_test
    elif split == "val":
        X_val = np.load(numpy_dir / "X_val.npy", mmap_mode=mmap_mode)
        y_val = np.load(numpy_dir / "y_val.npy")
        return X_val, y_val
    elif split == "test":
        X_test = np.load(numpy_dir / "X_test.npy", mmap_mode=mmap_mode)
        y_test = np.load(numpy_dir / "y_test.npy")
        return X_test, y_test
```

#### `load_unsupervised_metadata()`

```python
def load_unsupervised_metadata(
    path_base: str | Path,
    split: str = "all"  # 'train', 'val', 'test', o 'all'
) -> pd.DataFrame:
    """
    Carga metadatos del dataset no supervisado.
    
    Args:
        path_base: Ruta base del dataset
        split: Qué split cargar ('train', 'val', 'test', o 'all')
    
    Returns:
        DataFrame con metadatos
    """
    path_base = Path(path_base)
    metadata_dir = path_base / "metadata"
    
    if split == "all":
        return pd.read_csv(metadata_dir / "metadata_full.csv")
    elif split == "train":
        return pd.read_csv(metadata_dir / "metadata_train.csv")
    elif split == "val":
        return pd.read_csv(metadata_dir / "metadata_val.csv")
    elif split == "test":
        return pd.read_csv(metadata_dir / "metadata_test.csv")
```

### Estructura de Salida

```
data/Datos_no_supervisados/
├── numpy/
│   ├── X_train.npy  (solo normales)
│   ├── y_train.npy  (todos son 0, pero se guarda por consistencia)
│   ├── X_val.npy    (normales + anómalos)
│   ├── y_val.npy
│   ├── X_test.npy   (normales + anómalos)
│   └── y_test.npy
└── metadata/
    ├── metadata_train.csv
    ├── metadata_val.csv
    ├── metadata_test.csv
    ├── metadata_full.csv
    └── summary_stats.txt
```

### Ejemplo de Estadísticas

```
Train (solo normales):
  Total: 94,733 registros
  Shape: (94733, 5000, 3)
  Memoria: 5.29 GB

Val (normales + anómalos):
  Total: 78,301 registros
  Normales: 49,300
  Anómalos: 29,001
  Memoria: 4.38 GB

Test (normales + anómalos):
  Total: 78,302 registros
  Normales: 49,302
  Anómalos: 29,000
  Memoria: 4.38 GB
```

---

## 🔽 Downsampling de Datos Supervisados

**Archivo**: `downsample_supervised_data.ipynb`

**Propósito**: Reducir la frecuencia de muestreo de datos supervisados de 500Hz a 200Hz.

### ¿Qué Hace?

- Lee datos desde `Datos_supervisados/numpy/` y `Datos_supervisados/tensors/`
- Reduce cada señal de **5000 muestras** (10 seg @ 500Hz) a **2000 muestras** (10 seg @ 200Hz)
- Guarda los nuevos archivos en carpetas separadas (`numpy_200hz/`, `tensors_200hz/`)
- Mantiene la estructura original: `(N, 2000, 3)` donde 3 son los leads (II, V1, V5)

### Ventajas

- ✅ No necesitas reprocesar desde los datos originales
- ✅ Mantiene los splits train/val/test
- ✅ Conserva el preprocesado (filtrado, normalización)
- ✅ Reduce el tamaño de archivos ~2.5x (menos memoria)
- ✅ Acelera el entrenamiento

### Configuración

```python
# Ruta a la carpeta de datos supervisados
DATA_DIR = Path("../data/Datos_supervisados")

# Frecuencias de muestreo
ORIGINAL_FS = 500  # Hz original
TARGET_FS = 200    # Hz objetivo

# Duración de cada señal (segundos)
SIGNAL_DURATION = 10

# Calcular número de muestras
ORIGINAL_SAMPLES = int(SIGNAL_DURATION * ORIGINAL_FS)  # 5000
TARGET_SAMPLES = int(SIGNAL_DURATION * TARGET_FS)      # 2000

# Opciones de guardado
OVERWRITE_ORIGINAL = False  # No sobrescribir originales
USE_SEPARATE_FOLDER = True  # Guardar en carpetas separadas

# Procesar en chunks para ahorrar memoria
CHUNK_SIZE = 10000  # Número de muestras por chunk
```

### Método: `scipy.signal.resample`

El downsampling usa `scipy.signal.resample` que:
- Aplica filtrado anti-aliasing automáticamente
- Preserva la forma de la señal
- Mantiene la duración (10 segundos)

**Filtrado anti-aliasing**: Previene el aliasing (distorsión) que ocurriría al reducir la frecuencia de muestreo sin filtrar primero.

### Función de Downsampling con Guardado Incremental

```python
def downsample_and_save_incremental(
    X: np.ndarray, 
    target_samples: int, 
    output_file: Path,
    chunk_size: int = 10000
) -> tuple:
    """
    Reduce la frecuencia de muestreo y guarda incrementalmente en chunks.
    
    Args:
        X: Array de forma (N, T, C) con mmap_mode='r' o array normal
        target_samples: Número objetivo de timesteps (2000)
        output_file: Archivo donde guardar el resultado
        chunk_size: Tamaño de cada chunk a procesar
    
    Returns:
        Tuple con (shape_original, shape_final, tiempo_total)
    """
    if X.ndim != 3:
        raise ValueError(f"Se espera array 3D, pero se recibió {X.ndim}D")
    
    N, T, C = X.shape
    n_chunks = (N + chunk_size - 1) // chunk_size
    
    # Crear directorio si no existe
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Procesar y guardar chunk por chunk
    all_chunks = []
    checkpoint_file = output_file.parent / f".temp_{output_file.stem}_checkpoint.npy"
    
    for i, chunk_start in enumerate(range(0, N, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Leer chunk (desde mmap, muy eficiente)
        chunk = np.array(X[chunk_start:chunk_end])
        
        # Aplicar downsampling
        chunk_down = resample(chunk, target_samples, axis=1).astype(np.float32)
        del chunk  # Liberar inmediatamente
        
        # Acumular chunk
        all_chunks.append(chunk_down)
        
        # Cada 5 chunks o al final, guardar checkpoint
        if (i + 1) % 5 == 0 or chunk_end == N:
            new_chunks = np.concatenate(all_chunks, axis=0)
            
            # Cargar checkpoint anterior si existe
            if checkpoint_file.exists():
                previous = np.load(checkpoint_file)
                accumulated = np.concatenate([previous, new_chunks], axis=0)
                checkpoint_file.unlink()
            else:
                accumulated = new_chunks
            
            # Guardar nuevo checkpoint
            np.save(checkpoint_file, accumulated)
            all_chunks = []  # Limpiar lista
    
    # Cargar resultado final desde checkpoint
    accumulated = np.load(checkpoint_file)
    checkpoint_file.unlink()
    
    # Guardar archivo final
    np.save(output_file, accumulated)
    
    return (X.shape, accumulated.shape, elapsed)
```

**Características del guardado incremental**:
- **Procesa en chunks**: No carga todo en memoria
- **Checkpoints cada 5 chunks**: Permite reanudar si se interrumpe
- **Memoria eficiente**: Libera chunks después de procesarlos
- **Progreso visible**: Muestra porcentaje completado

### Procesamiento de Archivos NumPy

```python
numpy_dir = DATA_DIR / "numpy"
splits = ['train', 'val', 'test']

for split in splits:
    input_file = numpy_dir / f"X_{split}.npy"
    output_file = get_output_filename(input_file, OVERWRITE_ORIGINAL, USE_SEPARATE_FOLDER)
    
    # Cargar datos (usar mmap para ahorrar memoria)
    X = np.load(input_file, mmap_mode='r')
    
    # Aplicar downsampling y guardar incrementalmente
    input_shape, output_shape, elapsed = downsample_and_save_incremental(
        X, TARGET_SAMPLES, output_file, chunk_size=CHUNK_SIZE
    )
    
    del X  # Liberar memoria
```

### Procesamiento de Tensores PyTorch

```python
tensors_dir = DATA_DIR / "tensors"

if tensors_dir.exists():
    for split in splits:
        input_file = tensors_dir / f"X_{split}.pt"
        output_file = get_output_filename(input_file, OVERWRITE_ORIGINAL, USE_SEPARATE_FOLDER)
        
        # Cargar tensor
        X = torch.load(input_file)
        
        # Convertir a numpy para procesar
        X_np = X.numpy()
        
        # Aplicar downsampling
        X_down_np = resample(X_np, TARGET_SAMPLES, axis=1).astype(np.float32)
        
        # Convertir de vuelta a tensor
        X_down = torch.from_numpy(X_down_np)
        
        # Guardar
        torch.save(X_down, output_file)
```

### Copiar Archivos Y (Etiquetas)

Los archivos `y_*.npy` (etiquetas) no necesitan downsampling, solo se copian:

```python
for split in splits:
    y_file = numpy_dir / f"y_{split}.npy"
    y_output_file = get_output_filename(y_file, OVERWRITE_ORIGINAL, USE_SEPARATE_FOLDER)
    
    # Copiar archivo (no necesita procesamiento)
    shutil.copy2(y_file, y_output_file)
```

### Estructura de Salida

```
data/Datos_supervisados/
├── numpy/              # Originales (500Hz)
│   ├── X_train.npy
│   └── ...
├── numpy_200hz/        # Downsampled (200Hz)
│   ├── X_train.npy
│   ├── y_train.npy
│   └── ...
├── tensors/            # Originales (500Hz)
│   └── ...
└── tensors_200hz/      # Downsampled (200Hz)
    └── ...
```

### Ejemplo de Resultados

```
Procesando train...
  Shape original: (270668, 5000, 3)
  Tamaño total: 15.12 GB
  Aplicando downsampling 5000 → 2000...
  Shape nueva: (270668, 2000, 3)
  Tamaño nuevo: 6.05 GB
  Completado en 45.23 segundos
```

---

## 🔥 Downsampling y Conversión a Tensores (No Supervisados)

**Archivo**: `downsample_unsupervised_data.ipynb`

**Propósito**: Reducir frecuencia de muestreo de datos no supervisados de 500Hz a 200Hz y convertirlos a tensores PyTorch.

### ¿Qué Hace?

- Lee datos desde `Datos_no_supervisados/numpy/`
- Reduce cada señal de **5000 muestras** a **2000 muestras**
- Guarda datos downsampled en `numpy_200hz/` (guardado incremental constante)
- Convierte datos downsampled a tensores PyTorch y los guarda en `tensors_200hz/`
- Mantiene la estructura original: `(N, 2000, 3)`

### Ventajas

- ✅ Guardado incremental constante (checkpoints cada 5 chunks) para mayor velocidad
- ✅ No necesitas reprocesar desde los datos originales
- ✅ Mantiene los splits train/val/test
- ✅ Conserva el preprocesado (filtrado, normalización)
- ✅ Reduce el tamaño de archivos ~2.5x (menos memoria)
- ✅ Genera tensores listos para entrenamiento

### Configuración

```python
# Ruta a la carpeta de datos no supervisados
DATA_DIR = Path("../data/Datos_no_supervisados")

# Frecuencias de muestreo
ORIGINAL_FS = 500  # Hz original
TARGET_FS = 200    # Hz objetivo

# Calcular número de muestras
ORIGINAL_SAMPLES = int(SIGNAL_DURATION * ORIGINAL_FS)  # 5000
TARGET_SAMPLES = int(SIGNAL_DURATION * TARGET_FS)      # 2000

# Opciones
OVERWRITE_ORIGINAL = False
USE_SEPARATE_FOLDER = True
CHUNK_SIZE = 10000

# Convertir a tensores después del downsampling
CONVERT_TO_TENSORS = True
```

### Paso 1: Downsampling de Archivos NumPy

Similar al proceso de datos supervisados, pero con guardado constante:

```python
def downsample_and_save_incremental(
    X: np.ndarray, 
    target_samples: int, 
    output_file: Path,
    chunk_size: int = 10000
) -> tuple:
    """
    Reduce frecuencia de muestreo y guarda incrementalmente.
    Guarda constantemente (cada 5 chunks) para hacer el proceso más rápido.
    """
    # ... código similar al anterior ...
    
    # Cada 5 chunks o al final, guardar checkpoint
    # Esto hace el proceso más rápido al guardar constantemente
    if (i + 1) % 5 == 0 or chunk_end == N:
        # ... guardar checkpoint ...
```

### Paso 2: Conversión a Tensores PyTorch

```python
def convert_numpy_to_tensors(
    numpy_file: Path,
    tensor_file: Path,
    chunk_size: int = 10000
) -> tuple:
    """
    Convierte array numpy a tensores PyTorch y guarda incrementalmente.
    
    Args:
        numpy_file: Archivo .npy con datos downsampled
        tensor_file: Archivo .pt donde guardar tensores
        chunk_size: Tamaño de chunk para procesamiento
    
    Returns:
        Tuple con (shape, tiempo_total)
    """
    # Cargar datos numpy (usar mmap)
    X_np = np.load(numpy_file, mmap_mode='r')
    N, T, C = X_np.shape
    
    print(f"  Convirtiendo {N} señales a tensores PyTorch...")
    
    # Crear directorio si no existe
    tensor_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Procesar en chunks
    all_tensors = []
    checkpoint_file = tensor_file.parent / f".temp_{tensor_file.stem}_checkpoint.pt"
    
    for i, chunk_start in enumerate(range(0, N, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Leer chunk
        chunk_np = np.array(X_np[chunk_start:chunk_end])
        
        # Convertir a tensor
        chunk_tensor = torch.from_numpy(chunk_np).float()
        del chunk_np  # Liberar
        
        all_tensors.append(chunk_tensor)
        
        # Cada 5 chunks, guardar checkpoint
        if (i + 1) % 5 == 0 or chunk_end == N:
            new_tensors = torch.cat(all_tensors, dim=0)
            
            # Cargar checkpoint anterior si existe
            if checkpoint_file.exists():
                previous = torch.load(checkpoint_file)
                accumulated = torch.cat([previous, new_tensors], dim=0)
                checkpoint_file.unlink()
            else:
                accumulated = new_tensors
            
            # Guardar checkpoint
            torch.save(accumulated, checkpoint_file)
            all_tensors = []
            del new_tensors
    
    # Cargar resultado final
    accumulated = torch.load(checkpoint_file)
    checkpoint_file.unlink()
    
    # Guardar archivo final
    torch.save(accumulated, tensor_file)
    
    return (accumulated.shape, elapsed)
```

### Procesamiento Completo

```python
# 1. Downsampling de numpy
numpy_dir = DATA_DIR / "numpy"
numpy_200hz_dir = DATA_DIR / "numpy_200hz"

for split in ['train', 'val', 'test']:
    input_file = numpy_dir / f"X_{split}.npy"
    output_file = numpy_200hz_dir / f"X_{split}.npy"
    
    # Downsampling
    X = np.load(input_file, mmap_mode='r')
    downsample_and_save_incremental(X, TARGET_SAMPLES, output_file, CHUNK_SIZE)
    del X

# 2. Conversión a tensores (si está habilitado)
if CONVERT_TO_TENSORS:
    tensors_200hz_dir = DATA_DIR / "tensors_200hz"
    
    for split in ['train', 'val', 'test']:
        numpy_file = numpy_200hz_dir / f"X_{split}.npy"
        tensor_file = tensors_200hz_dir / f"X_{split}.pt"
        
        # Conversión
        convert_numpy_to_tensors(numpy_file, tensor_file, CHUNK_SIZE)
```

### Estructura de Salida

```
data/Datos_no_supervisados/
├── numpy/              # Originales (500Hz)
│   ├── X_train.npy
│   └── ...
├── numpy_200hz/        # Downsampled (200Hz)
│   ├── X_train.npy
│   ├── y_train.npy
│   └── ...
└── tensors_200hz/      # Tensores PyTorch (200Hz) ⭐
    ├── X_train.pt
    ├── y_train.pt
    ├── X_val.pt
    ├── y_val.pt
    ├── X_test.pt
    └── y_test.pt
```

### Ventajas de Tensores PyTorch

1. **Carga más rápida**: Los tensores se cargan más rápido que arrays numpy
2. **Listos para entrenamiento**: No necesitas conversión adicional
3. **Memoria eficiente**: PyTorch optimiza el almacenamiento
4. **Compatibilidad**: Compatible directamente con DataLoaders

### Ejemplo de Uso en Notebooks de Entrenamiento

```python
# Cargar tensores directamente
X_train = torch.load("data/Datos_no_supervisados/tensors_200hz/X_train.pt")
y_train = torch.load("data/Datos_no_supervisados/tensors_200hz/y_train.pt")

# Crear DataLoader directamente
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

---

## 🚀 Guía de Uso

### Flujo Completo Recomendado

#### 1. Construir Dataset Supervisado

```bash
jupyter notebook build_supervised_ecg_dataset.ipynb
```

**Tiempo estimado**: 1-2 horas

#### 2. Construir Dataset No Supervisado

```bash
jupyter notebook build_unsupervised_ecg_dataset.ipynb
```

**Tiempo estimado**: 10-20 minutos

**Requisito**: Dataset supervisado debe estar construido primero.

#### 3. (Opcional) Downsampling de Datos Supervisados

```bash
jupyter notebook downsample_supervised_data.ipynb
```

**Tiempo estimado**: 30-60 minutos

**Ventajas**:
- Reduce tamaño de archivos ~2.5x
- Acelera entrenamiento
- Mantiene preprocesado original

#### 4. (Opcional) Downsampling y Conversión a Tensores (No Supervisados)

```bash
jupyter notebook downsample_unsupervised_data.ipynb
```

**Tiempo estimado**: 30-60 minutos

**Ventajas**:
- Reduce tamaño de archivos ~2.5x
- Genera tensores listos para entrenamiento
- Carga más rápida en notebooks de entrenamiento

### Orden de Ejecución

```
1. build_supervised_ecg_dataset.ipynb
   ↓
2. build_unsupervised_ecg_dataset.ipynb
   ↓
3. (Opcional) downsample_supervised_data.ipynb
   ↓
4. (Opcional) downsample_unsupervised_data.ipynb
   ↓
5. Notebooks de entrenamiento (usan tensors_200hz/)
```

### Configuración de Rutas en Notebooks de Entrenamiento

**Para datos supervisados**:
```python
DATA_DIR = Path("../data/Datos_supervisados/tensors_200hz")
```

**Para datos no supervisados**:
```python
DATA_DIR = Path("../data/Datos_no_supervisados/tensors_200hz")
```

---

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Error: "Datos supervisados no encontrados"

**Síntoma**: Al ejecutar `build_unsupervised_ecg_dataset.ipynb`, aparece error de archivos faltantes.

**Solución**:
1. Ejecuta primero `build_supervised_ecg_dataset.ipynb`
2. Verifica que los archivos existan en `data/Datos_supervisados/numpy/`

#### 2. Out of Memory durante Downsampling

**Síntoma**: Error de memoria al procesar archivos grandes.

**Soluciones**:
- Reduce `CHUNK_SIZE` (ej: de 10000 a 5000)
- Usa `mmap_mode='r'` al cargar (ya está implementado)
- Cierra otros programas que usen memoria

#### 3. Checkpoints Temporales No Eliminados

**Síntoma**: Archivos `.temp_*_checkpoint.npy` o `.temp_*_checkpoint.pt` quedan en el directorio.

**Solución**:
```python
# Limpiar manualmente
import glob
from pathlib import Path

data_dir = Path("../data/Datos_no_supervisados")
for temp_file in data_dir.rglob(".temp_*"):
    temp_file.unlink()
    print(f"Eliminado: {temp_file}")
```

#### 4. Archivos Ya Existen

**Síntoma**: El notebook pregunta si sobrescribir archivos existentes.

**Opciones**:
- **Sí (s)**: Sobrescribe archivos existentes
- **No (n)**: Salta el procesamiento de ese split

**Para forzar sobrescritura**:
```python
OVERWRITE_ORIGINAL = True  # En configuración
```

#### 5. Verificación de Resultados

**Verificar shapes**:
```python
import numpy as np
from pathlib import Path

# Verificar datos downsampled
data_dir = Path("../data/Datos_supervisados/numpy_200hz")
X_train = np.load(data_dir / "X_train.npy", mmap_mode='r')
print(f"Shape: {X_train.shape}")  # Debe ser (N, 2000, 3)
print(f"Tamaño: {X_train.nbytes / 1024**3:.2f} GB")
```

**Verificar tensores**:
```python
import torch

# Verificar tensores
tensor_dir = Path("../data/Datos_no_supervisados/tensors_200hz")
X_train = torch.load(tensor_dir / "X_train.pt")
print(f"Shape: {X_train.shape}")  # Debe ser torch.Size([N, 2000, 3])
print(f"Dtype: {X_train.dtype}")  # Debe ser torch.float32
```

### Optimización de Rendimiento

#### Ajustar CHUNK_SIZE

- **CHUNK_SIZE pequeño** (5000): Menos memoria, más lento
- **CHUNK_SIZE grande** (20000): Más memoria, más rápido
- **Recomendado**: 10000 (balance)

#### Usar SSD en lugar de HDD

Los checkpoints frecuentes son más rápidos en SSD.

#### Procesar en Paralelo (Futuro)

Actualmente el procesamiento es secuencial. Para datasets muy grandes, se podría paralelizar por split.

---

## 📝 Resumen

Esta documentación cubre:

✅ **Pipeline de datos no supervisados**: Cómo se construyen datasets para autoencoders  
✅ **Downsampling de datos supervisados**: Reducción de 500Hz a 200Hz  
✅ **Downsampling y conversión a tensores**: Proceso completo para datos no supervisados  
✅ **Funciones de carga**: Cómo usar los datos en notebooks de entrenamiento  
✅ **Guías de uso**: Orden de ejecución y configuración  
✅ **Troubleshooting**: Solución de problemas comunes  

Para más información sobre el pipeline supervisado, ver [Documentacion Datos Supervisados.md](Documentacion%20Datos%20Supervisados.md).

---

**Última actualización**: 2025-01-XX

