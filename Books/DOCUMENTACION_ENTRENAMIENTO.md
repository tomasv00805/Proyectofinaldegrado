# 🧠 Documentación Detallada: Notebooks de Entrenamiento

Esta documentación explica en detalle cómo están estructurados los notebooks de entrenamiento, cómo se construyen los modelos, y cómo funcionan las integraciones con MLflow y Prefect.

## 📋 Tabla de Contenidos

1. [Estructura General de los Notebooks](#estructura-general-de-los-notebooks)
2. [Arquitecturas de Modelos](#arquitecturas-de-modelos)
3. [Integración con MLflow](#integración-con-mlflow)
4. [Orquestación con Prefect](#orquestación-con-prefect)
5. [Flujo Completo de Entrenamiento](#flujo-completo-de-entrenamiento)
6. [Guía de Personalización](#guía-de-personalización)

---

## 📐 Estructura General de los Notebooks

Todos los notebooks de entrenamiento siguen una estructura estándar de **10 secciones principales**:

### 1. Setup CUDA y Dependencias

**Propósito**: Configurar el entorno, instalar dependencias y configurar CUDA para Windows.

**Contenido**:
- Configuración de rutas DLL de CUDA (necesario en Windows)
- Instalación automática de dependencias base (MLflow, Prefect, scikit-learn, etc.)
- Instalación de PyTorch con CUDA 12.8 (compatible con RTX 5080)
- Verificación de GPU disponible

**Ejemplo de código**:
```python
# Rutas candidatas para DLLs de CUDA
CUDA_CANDIDATES = [
    os.environ.get("CUDA_PATH"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    # ...
]

# Añadir rutas DLL en Windows
if hasattr(os, "add_dll_directory"):
    for candidate in CUDA_CANDIDATES:
        path = Path(candidate)
        if path.is_dir():
            os.add_dll_directory(str(path))
```

**⚠️ Importante**: Después de ejecutar esta celda, **debes reiniciar el kernel de Jupyter** si PyTorch fue reinstalado.

---

### 2. Configuración General

**Propósito**: Definir todos los hiperparámetros, rutas y configuraciones del experimento.

**Componentes principales**:

#### Rutas y Nombres
```python
DATA_DIR = Path("../data/Datos_supervisados/tensors_200hz")
EXPERIMENT_NAME = "CNN1D_LSTM_ECG_Supervisado_v1"  # Nombre del experimento en MLflow
RUN_NAME = "cnn1d_lstm_ecg_v1"  # Nombre del run en MLflow
OUTPUT_DIR = Path("./outputs")  # Directorio para guardar artefactos
```

#### Parámetros de Datos
```python
N_CHANNELS = 3      # Derivaciones de ECG (II, V1, V5)
SEQ_LEN = 2000      # Timesteps por ejemplo (10 seg @ 200Hz)
```

#### Parámetros de Arquitectura
- **Para modelos CNN1D**: `out_channels_list`, `kernel_sizes`, `pool_sizes`
- **Para modelos LSTM**: `HIDDEN_SIZE`, `NUM_LAYERS`, `DROPOUT`
- **Para modelos Transformer**: `d_model`, `nhead`, `num_encoder_layers`
- **Para Autoencoders**: `enc_out_channels`, `dec_out_channels`, `latent_channels`

#### Parámetros de Entrenamiento
```python
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-5
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
CLIP_GRAD_NORM = 1.0
```

#### Configuración de Semillas y GPU
```python
SEED = 42
USE_CUDA = True
ENABLE_CUDNN_BENCHMARK = True
```

**Función de configuración de semillas**:
```python
def set_seed_everywhere(seed: int = 42, enable_cudnn_benchmark: bool = True):
    """Fija semillas para reproducibilidad y optimiza GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark
```

---

### 3. Carga y Preparación de Datos

**Propósito**: Cargar datos desde archivos `.pt` y crear DataLoaders de PyTorch.

**Funciones principales**:

#### `load_tensor_data(data_dir: Path)`
Carga tensores desde disco:
```python
def load_tensor_data(data_dir: Path):
    """Carga tensores X_train, y_train, X_val, y_val, X_test, y_test."""
    X_train = torch.load(data_dir / "X_train.pt")
    y_train = torch.load(data_dir / "y_train.pt")
    # ... similar para val y test
    return X_train, y_train, X_val, y_val, X_test, y_test
```

#### `create_dataloaders_from_tensors(...)`
Crea DataLoaders de PyTorch:
```python
def create_dataloaders_from_tensors(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 128,
    shuffle_train: bool = True,
):
    """Crea DataLoaders para train, val y test."""
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,  # 0 en Windows, >0 en Linux
        pin_memory=True if torch.cuda.is_available() else False,
    )
    # ... similar para val y test
    return train_loader, val_loader, test_loader
```

**Nota para Autoencoders**: En modelos de detección de anomalías, el train loader solo contiene ejemplos normales (filtrados antes de crear el DataLoader).

---

### 4. Definición del Modelo

**Propósito**: Definir la arquitectura del modelo como una clase que hereda de `nn.Module`.

Ver sección [Arquitecturas de Modelos](#arquitecturas-de-modelos) para detalles completos.

**Función de creación**:
```python
def create_model(config: Dict) -> ModelClass:
    """Crea e instancia el modelo."""
    model = ModelClass(
        n_channels=config["N_CHANNELS"],
        seq_len=config["SEQ_LEN"],
        # ... otros parámetros
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")
    
    return model
```

---

### 5. Funciones de Entrenamiento y Evaluación

**Propósito**: Implementar los loops de entrenamiento y evaluación.

#### Función de Entrenamiento por Época

```python
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Entrena el modelo por una época.
    
    Returns:
        Tupla con (loss_promedio, accuracy_promedio)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True).float()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Normalizar pérdida si usas gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Calcular accuracy
        predictions = (outputs > 0.5).long()
        correct += (predictions == batch_y.long()).sum().item()
        total += batch_y.size(0)
        total_loss += loss.item() * gradient_accumulation_steps
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc
```

#### Función de Evaluación

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en un conjunto de datos.
    
    Returns:
        Tupla con (loss, accuracy, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).float()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            predictions = (outputs > 0.5).long()
            correct += (predictions == batch_y.long()).sum().item()
            total += batch_y.size(0)
            total_loss += loss.item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.long().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds)
```

#### Función de Cálculo de Métricas

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calcula métricas completas de clasificación."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Por clase
    precision_normal = precision_score(y_true, y_pred, pos_label=0)
    recall_normal = recall_score(y_true, y_pred, pos_label=0)
    f1_normal = f1_score(y_true, y_pred, pos_label=0)
    
    # ... similar para clase anómala
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_normal": precision_normal,
        "recall_normal": recall_normal,
        "f1_normal": f1_normal,
        # ... más métricas
        "confusion_matrix": cm,
    }
```

---

### 6. Integración con MLflow

**Propósito**: Tracking completo de experimentos, hiperparámetros, métricas y artefactos.

Ver sección [Integración con MLflow](#integración-con-mlflow) para detalles completos.

---

### 7. Orquestación con Prefect

**Propósito**: Orquestar el flujo completo de entrenamiento con manejo de errores y logging.

Ver sección [Orquestación con Prefect](#orquestación-con-prefect) para detalles completos.

---

### 8. Ejecución del Flujo Completo

**Propósito**: Celda final que ejecuta todo el pipeline.

```python
if __name__ == "__main__":
    results = training_flow(CONFIG)
    print("\n✓ Proceso finalizado exitosamente")
```

---

### 9. Guardado de Modelo

**Propósito**: Exportar modelos en diferentes formatos.

#### Guardado Estándar
```python
def save_model_standard(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    metadata: Dict,
):
    """Guarda modelo en formato estándar PyTorch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar state_dict
    torch.save(model.state_dict(), output_dir / f"{model_name}_state_dict.pt")
    
    # Guardar modelo completo
    torch.save(model, output_dir / f"{model_name}.pt")
    
    # Guardar metadatos
    with open(output_dir / f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

#### Exportación para AWS SageMaker
```python
def export_model_for_sagemaker(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    config: Dict,
):
    """Exporta modelo en formato compatible con SageMaker."""
    # Crear estructura de directorios
    sagemaker_dir = output_dir / f"sagemaker_{model_name}"
    code_dir = sagemaker_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    torch.save(model.state_dict(), sagemaker_dir / "model.pth")
    
    # Crear inference.py
    # ... código de inference para SageMaker
    
    # Crear requirements.txt
    # ... dependencias
    
    # Crear config.json
    # ... configuración
```

---

### 10. Prueba de Predicciones

**Propósito**: Visualizar predicciones en ejemplos específicos.

```python
def test_model_with_examples_interactive(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_examples: int = 5,
):
    """Muestra predicciones en ejemplos aleatorios del test set."""
    model.eval()
    
    # Obtener ejemplos aleatorios
    indices = np.random.choice(len(test_loader.dataset), n_examples, replace=False)
    
    for idx in indices:
        x, y_true = test_loader.dataset[idx]
        x = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            y_pred_proba = model(x)
            y_pred = (y_pred_proba > 0.5).long().item()
        
        # Visualizar señal y predicción
        # ...
```

---

## 🏗️ Arquitecturas de Modelos

### 1. CNN1D Clasificador

**Arquitectura**: CNN1D pura para extracción de características locales.

**Estructura**:
```
Input: (batch, channels, seq_len)
    ↓
[Conv1d → BatchNorm → ReLU → MaxPool] × N capas
    ↓
Flatten
    ↓
[Linear → ReLU → Dropout] × M capas
    ↓
Linear (1) → Sigmoid
    ↓
Output: (batch,) probabilidades
```

**Implementación**:
```python
class CNN1D_Classifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        out_channels_list: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[Optional[int]],
        use_batchnorm: bool = True,
        cnn_activation: str = "relu",
        cnn_dropout: float = 0.1,
        fc_units: List[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Bloque CNN
        cnn_layers = []
        in_channels = n_channels
        
        for i in range(len(out_channels_list)):
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels_list[i],
                kernel_size=kernel_sizes[i],
                padding=(kernel_sizes[i] - 1) // 2,  # 'same' padding
            ))
            
            if use_batchnorm:
                cnn_layers.append(nn.BatchNorm1d(out_channels_list[i]))
            
            cnn_layers.append(nn.ReLU())
            
            if cnn_dropout > 0:
                cnn_layers.append(nn.Dropout(cnn_dropout))
            
            if pool_sizes[i] is not None:
                cnn_layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i]))
            
            in_channels = out_channels_list[i]
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calcular tamaño de salida de CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            cnn_output = self.cnn(dummy_input)
            flattened_size = cnn_output.numel()
        
        # Capas Fully Connected
        fc_layers = []
        in_features = flattened_size
        
        for fc_size in fc_units:
            fc_layers.append(nn.Linear(in_features, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            in_features = fc_size
        
        self.fc = nn.Sequential(*fc_layers)
        self.fc_out = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Asegurar formato (batch, channels, seq_len)
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        # CNN
        cnn_out = self.cnn(x)  # (batch, out_channels, seq_len_reduced)
        
        # Flatten
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # (batch, flattened_size)
        
        # Fully Connected
        fc_out = self.fc(cnn_out)
        out = self.fc_out(fc_out)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)  # (batch,)
```

---

### 2. CNN1D + LSTM Clasificador

**Arquitectura**: Híbrida - CNN1D para características locales + LSTM para dependencias temporales.

**Estructura**:
```
Input: (batch, channels, seq_len)
    ↓
[Conv1d → BatchNorm → ReLU → MaxPool] × N capas CNN
    ↓
Transpose: (batch, seq_len_reduced, features)
    ↓
LSTM (bidireccional, M capas)
    ↓
Last hidden state: (batch, hidden_size)
    ↓
[Linear → ReLU → Dropout]
    ↓
Linear (1) → Sigmoid
    ↓
Output: (batch,) probabilidades
```

**Implementación**:
```python
class CNN1D_LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        out_channels_list: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[Optional[int]],
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        fc_units: int = 64,
    ):
        super().__init__()
        
        # Bloque CNN (igual que CNN1D puro)
        # ... código similar al anterior
        
        # Calcular tamaño de salida de CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_seq_len = cnn_output.shape[2]
            self.cnn_output_channels = cnn_output.shape[1]
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        
        # Fully Connected
        self.fc1 = nn.Linear(hidden_size * 2, fc_units)  # *2 por bidireccional
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        cnn_out = self.cnn(x)  # (batch, channels, seq_len_reduced)
        
        # Transponer para LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len_reduced, channels)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_out)
        # Usar último hidden state de última capa
        last_hidden = hidden[-1]  # (batch, hidden_size) - solo forward
        # Para bidireccional, concatenar forward y backward
        if self.lstm.bidirectional:
            forward_hidden = hidden[-2]  # Forward
            backward_hidden = hidden[-1]  # Backward
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Fully Connected
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)
```

**Características clave**:
- **CNN extrae características locales**: Detecta patrones como ondas P, QRS, T
- **LSTM captura dependencias temporales**: Relaciones entre eventos a lo largo del tiempo
- **Bidireccional**: Procesa la señal en ambas direcciones para mejor contexto

---

### 3. CNN1D + Transformer Clasificador

**Arquitectura**: CNN1D + Transformer Encoder con self-attention.

**Estructura**:
```
Input: (batch, channels, seq_len)
    ↓
[Conv1d → BatchNorm → ReLU → MaxPool] × N capas CNN
    ↓
Transpose: (batch, seq_len_reduced, features)
    ↓
Positional Encoding
    ↓
Transformer Encoder (M capas)
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Norm
    ↓
Global Average Pooling o CLS token
    ↓
[Linear → ReLU → Dropout]
    ↓
Linear (1) → Sigmoid
    ↓
Output: (batch,) probabilidades
```

**Implementación**:
```python
class CNN1D_TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        out_channels_list: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[Optional[int]],
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        fc_units: int = 64,
    ):
        super().__init__()
        
        # Bloque CNN
        # ... código similar
        
        # Calcular tamaño de salida de CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_seq_len = cnn_output.shape[2]
            self.cnn_output_channels = cnn_output.shape[1]
        
        # Proyección a d_model
        self.cnn_to_transformer = nn.Linear(self.cnn_output_channels, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.cnn_output_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Fully Connected
        self.fc1 = nn.Linear(d_model, fc_units)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        cnn_out = self.cnn(x)  # (batch, channels, seq_len_reduced)
        
        # Transponer y proyectar
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len_reduced, channels)
        transformer_input = self.cnn_to_transformer(cnn_out)  # (batch, seq_len_reduced, d_model)
        
        # Positional Encoding
        transformer_input = self.pos_encoder(transformer_input)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(transformer_input)  # (batch, seq_len_reduced, d_model)
        
        # Global Average Pooling
        pooled = transformer_out.mean(dim=1)  # (batch, d_model)
        
        # Fully Connected
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)
```

**Positional Encoding**:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**Características clave**:
- **Self-Attention**: Captura relaciones entre cualquier par de timesteps
- **Multi-Head**: Múltiples representaciones de atención en paralelo
- **Global Context**: Puede relacionar eventos distantes en el tiempo

---

### 4. CNN1D Autoencoder

**Arquitectura**: Encoder-decoder CNN1D para detección de anomalías.

**Estructura**:
```
Input: (batch, channels, seq_len)
    ↓
ENCODER:
[Conv1d → BatchNorm → ReLU → MaxPool] × N capas
    ↓
Latent Representation: (batch, latent_channels, encoded_seq_len)
    ↓
DECODER:
[ConvTranspose1d/Upsample → Conv1d → BatchNorm → ReLU] × M capas
    ↓
Output: (batch, channels, seq_len) - Reconstrucción
```

**Implementación**:
```python
class CNN1D_Autoencoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        enc_out_channels: List[int],
        enc_kernel_sizes: List[int],
        enc_pool_sizes: List[Optional[int]],
        dec_out_channels: List[int],
        dec_kernel_sizes: List[int],
        dec_upsample_sizes: List[Optional[int]],
        latent_channels: int,
        use_batchnorm: bool = True,
        cnn_activation: str = "relu",
    ):
        super().__init__()
        
        # ENCODER
        encoder_layers = []
        in_channels = n_channels
        
        for i in range(len(enc_out_channels)):
            encoder_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=enc_out_channels[i],
                kernel_size=enc_kernel_sizes[i],
                padding=(enc_kernel_sizes[i] - 1) // 2,
            ))
            
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(enc_out_channels[i]))
            
            encoder_layers.append(nn.ReLU())
            
            if enc_pool_sizes[i] is not None:
                encoder_layers.append(nn.MaxPool1d(kernel_size=enc_pool_sizes[i]))
            
            in_channels = enc_out_channels[i]
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calcular tamaño de salida del encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            encoder_output = self.encoder(dummy_input)
            self.encoded_seq_len = encoder_output.shape[2]
            self.encoded_channels = encoder_output.shape[1]
        
        # Proyección a latent_channels
        if self.encoded_channels != latent_channels:
            self.latent_projection = nn.Conv1d(
                in_channels=self.encoded_channels,
                out_channels=latent_channels,
                kernel_size=1,
            )
            self.encoded_channels = latent_channels
        else:
            self.latent_projection = nn.Identity()
        
        # DECODER
        decoder_layers = []
        decoder_in_channels = latent_channels
        
        for i in range(len(dec_out_channels)):
            # Upsampling
            if dec_upsample_sizes[i] is not None:
                decoder_layers.append(nn.ConvTranspose1d(
                    in_channels=decoder_in_channels,
                    out_channels=decoder_in_channels,
                    kernel_size=dec_upsample_sizes[i],
                    stride=dec_upsample_sizes[i],
                ))
            
            # Conv1d
            decoder_layers.append(nn.Conv1d(
                in_channels=decoder_in_channels,
                out_channels=dec_out_channels[i],
                kernel_size=dec_kernel_sizes[i],
                padding=(dec_kernel_sizes[i] - 1) // 2,
            ))
            
            if use_batchnorm and i < len(dec_out_channels) - 1:
                decoder_layers.append(nn.BatchNorm1d(dec_out_channels[i]))
            
            # Activación (tanh en última capa)
            if i == len(dec_out_channels) - 1:
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.ReLU())
            
            decoder_in_channels = dec_out_channels[i]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Ajuste final de tamaño
        with torch.no_grad():
            dummy_latent = torch.zeros(1, latent_channels, self.encoded_seq_len)
            decoder_output = self.decoder(dummy_latent)
            if decoder_output.shape[2] != seq_len:
                self.final_adjust = nn.Upsample(size=seq_len, mode='linear')
            else:
                self.final_adjust = nn.Identity()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: x -> representación latente."""
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        encoded = self.encoder(x)
        encoded = self.latent_projection(encoded)
        return encoded
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decoder: representación latente -> reconstrucción."""
        decoded = self.decoder(latent)
        decoded = self.final_adjust(decoded)
        return decoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo: encode -> decode."""
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Transponer de vuelta
        reconstructed = reconstructed.transpose(1, 2)
        return reconstructed
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula error de reconstrucción (MSE) por muestra."""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse
```

**Características clave**:
- **Encoder comprime**: Reduce la señal a una representación latente compacta
- **Decoder reconstruye**: Intenta recuperar la señal original desde el latente
- **Error de reconstrucción**: ECG normales tienen error bajo, anómalos tienen error alto

---

### 5. CNN1D + LSTM Autoencoder

**Arquitectura**: Híbrida - CNN1D + LSTM en encoder y decoder.

**Estructura**:
```
Input: (batch, channels, seq_len)
    ↓
ENCODER:
CNN1D → LSTM Encoder
    ↓
Latent: (batch, latent_size)
    ↓
DECODER:
LSTM Decoder → CNN1D
    ↓
Output: (batch, channels, seq_len)
```

**Implementación**:
```python
class CNN1D_LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        out_channels_list: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[Optional[int]],
        enc_hidden_size: int,
        enc_num_layers: int,
        enc_dropout: float,
        dec_hidden_size: int,
        dec_num_layers: int,
        dec_dropout: float,
        latent_dim: int,
        bidirectional: bool = True,
        use_batchnorm: bool = True,
        cnn_activation: str = "relu",
    ):
        super().__init__()
        
        # ENCODER: CNN1D
        cnn_layers = []
        in_channels = n_channels
        
        for i in range(len(out_channels_list)):
            cnn_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels_list[i],
                kernel_size=kernel_sizes[i],
                padding=(kernel_sizes[i] - 1) // 2,
            ))
            
            if use_batchnorm:
                cnn_layers.append(nn.BatchNorm1d(out_channels_list[i]))
            
            cnn_layers.append(nn.ReLU())
            
            if pool_sizes[i] is not None:
                cnn_layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i]))
            
            in_channels = out_channels_list[i]
        
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        
        # Calcular tamaño de salida de CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            cnn_output = self.cnn_encoder(dummy_input)
            self.cnn_output_seq_len = cnn_output.shape[2]
            self.cnn_output_channels = cnn_output.shape[1]
        
        # ENCODER: LSTM
        self.lstm_encoder = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=enc_hidden_size,
            num_layers=enc_num_layers,
            batch_first=True,
            dropout=enc_dropout if enc_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Proyección al espacio latente
        lstm_output_size = enc_hidden_size * 2 if bidirectional else enc_hidden_size
        self.latent_projection = nn.Linear(lstm_output_size, latent_dim)
        
        # DECODER: LSTM
        self.latent_to_hidden = nn.Linear(latent_dim, dec_hidden_size * dec_num_layers)
        
        self.lstm_decoder = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=dec_hidden_size,
            num_layers=dec_num_layers,
            batch_first=True,
            dropout=dec_dropout if dec_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        
        # DECODER: Proyección a canales originales
        self.hidden_to_cnn = nn.Linear(dec_hidden_size, self.cnn_output_channels)
        self.output_projection = nn.Sequential(
            nn.Linear(self.cnn_output_channels, n_channels),
            nn.Tanh()
        )
        
        # Upsampling para revertir pooling
        upsample_layers = []
        for i in range(len(pool_sizes) - 1, -1, -1):
            if pool_sizes[i] is not None:
                upsample_layers.append(
                    nn.Upsample(scale_factor=pool_sizes[i], mode='linear')
                )
        self.upsample = nn.Sequential(*upsample_layers) if upsample_layers else nn.Identity()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: CNN1D → LSTM → Latent."""
        if x.shape[1] == self.seq_len:
            x = x.transpose(1, 2)
        
        # CNN
        cnn_out = self.cnn_encoder(x)  # (batch, channels, seq_len_reduced)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len_reduced, channels)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm_encoder(cnn_out)
        
        # Último hidden state
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Latent
        latent = self.latent_projection(last_hidden)
        return latent
    
    def decode(self, latent: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        """Decoder: Latent → LSTM → CNN1D → Reconstrucción."""
        batch_size = latent.shape[0]
        
        # Inicializar hidden state desde latent
        hidden_init = self.latent_to_hidden(latent)
        hidden_init = hidden_init.view(batch_size, self.lstm_decoder.num_layers, self.dec_hidden_size)
        hidden_init = hidden_init.transpose(0, 1)
        
        # Input inicial (ceros)
        decoder_input = torch.zeros(
            batch_size, target_seq_len, self.cnn_output_channels,
            device=latent.device
        )
        
        # LSTM Decoder
        lstm_out, _ = self.lstm_decoder(decoder_input, (hidden_init, torch.zeros_like(hidden_init)))
        
        # Proyección
        projected = self.hidden_to_cnn(lstm_out)  # (batch, seq_len, cnn_channels)
        projected = self.output_projection(projected)  # (batch, seq_len, n_channels)
        
        # Upsampling
        projected = projected.transpose(1, 2)  # (batch, n_channels, seq_len_reduced)
        upsampled = self.upsample(projected)  # (batch, n_channels, seq_len)
        reconstructed = upsampled.transpose(1, 2)  # (batch, seq_len, n_channels)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward completo."""
        latent = self.encode(x)
        reconstructed = self.decode(latent, self.cnn_output_seq_len)
        return reconstructed
```

**Características clave**:
- **Encoder CNN1D**: Extrae características locales
- **Encoder LSTM**: Captura dependencias temporales en las características
- **Decoder LSTM**: Genera secuencia desde el latente
- **Decoder CNN1D**: Reconstruye la señal original
- **Mejor que CNN1D puro**: Captura mejor patrones temporales complejos

---

### 6. LSTM Clasificador

**Arquitectura**: LSTM puro para clasificación (sin CNN).

**Estructura**:
```
Input: (batch, seq_len, input_size)
    ↓
LSTM (bidireccional, M capas)
    ↓
Last hidden state: (batch, hidden_size * 2)
    ↓
[Linear → ReLU → Dropout]
    ↓
Linear (1) → Sigmoid
    ↓
Output: (batch,) probabilidades
```

**Implementación**:
```python
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True,
        fc_units: int = 64,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Fully Connected
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, fc_units)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de forma (batch_size, seq_len, input_size)
        
        Returns:
            Tensor de forma (batch_size,) con probabilidades
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Usar último hidden state
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            forward_hidden = hidden[-2]  # Forward
            backward_hidden = hidden[-1]  # Backward
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            last_hidden = hidden[-1]
        
        # Fully Connected
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze(-1)
```

**Características clave**:
- **Sin CNN**: Procesa directamente la secuencia temporal
- **Bidireccional**: Procesa en ambas direcciones para mejor contexto
- **Múltiples capas**: Captura dependencias a diferentes niveles
- **Más simple**: Menos parámetros que arquitecturas híbridas

---

### 7. LSTM Autoencoder

**Arquitectura**: LSTM puro en encoder y decoder para detección de anomalías.

**Estructura**:
```
Input: (batch, seq_len, input_size)
    ↓
ENCODER:
LSTM (M capas) → Latent Projection
    ↓
Latent: (batch, latent_dim)
    ↓
DECODER:
Latent Expansion → LSTM (M capas)
    ↓
Output: (batch, seq_len, input_size) - Reconstrucción
```

**Implementación**:
```python
class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes_encoder: List[int],
        hidden_sizes_decoder: List[int],
        latent_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # ENCODER: Múltiples capas LSTM
        encoder_layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_sizes_encoder[i - 1]
            out_size = hidden_sizes_encoder[i]
            encoder_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=out_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0 if i == num_layers - 1 else dropout,
                )
            )
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Proyección al espacio latente
        self.latent_projection = nn.Linear(
            hidden_sizes_encoder[-1],
            latent_dim
        )
        
        # DECODER: Expansión del latente
        self.latent_expansion = nn.Linear(
            latent_dim,
            hidden_sizes_decoder[0]
        )
        
        # DECODER: Múltiples capas LSTM
        decoder_layers = []
        for i in range(num_layers):
            in_size = hidden_sizes_decoder[i]
            out_size = hidden_sizes_decoder[i + 1] if i < num_layers - 1 else input_size
            decoder_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=out_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0 if i == num_layers - 1 else dropout,
                )
            )
        self.decoder = nn.ModuleList(decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder: x -> latent vector
        
        Args:
            x: Tensor de forma (batch_size, seq_len, input_size)
        
        Returns:
            Tensor de forma (batch_size, latent_dim)
        """
        h = x
        for lstm_layer in self.encoder:
            h, (hidden, cell) = lstm_layer(h)
            # Usar último hidden state
            h = hidden[-1]  # (batch_size, hidden_size)
            h = h.unsqueeze(1)  # (batch_size, 1, hidden_size) para siguiente capa
        
        # Último hidden state de última capa
        last_hidden = h.squeeze(1)  # (batch_size, hidden_size)
        
        # Proyección al espacio latente
        latent = self.latent_projection(last_hidden)  # (batch_size, latent_dim)
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decoder: latent -> reconstrucción
        
        Args:
            latent: Tensor de forma (batch_size, latent_dim)
            seq_len: Longitud de secuencia a reconstruir
        
        Returns:
            Tensor de forma (batch_size, seq_len, input_size)
        """
        # Expandir espacio latente
        h = self.latent_expansion(latent)  # (batch_size, hidden_size_decoder[0])
        h = h.unsqueeze(1)  # (batch_size, 1, hidden_size_decoder[0])
        
        # Repetir para toda la secuencia
        h = h.repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size_decoder[0])
        
        # Decodificar capa por capa
        for lstm_layer in self.decoder:
            h, (hidden, cell) = lstm_layer(h)
        
        return h  # (batch_size, seq_len, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward completo: encode -> decode
        
        Args:
            x: Tensor de forma (batch_size, seq_len, input_size)
        
        Returns:
            Tensor reconstruido de forma (batch_size, seq_len, input_size)
        """
        latent = self.encode(x)
        seq_len = x.size(1)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula error de reconstrucción (MSE) por muestra."""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse
```

**Características clave**:
- **Encoder LSTM**: Comprime secuencia a vector latente
- **Decoder LSTM**: Reconstruye secuencia desde latente
- **Múltiples capas**: Mejor capacidad de compresión/reconstrucción
- **Enfocado en temporal**: Sin convoluciones, solo dependencias temporales

---

## 📊 Integración con MLflow

MLflow se integra en múltiples puntos del flujo de entrenamiento para tracking completo.

### Configuración Inicial

```python
def setup_mlflow(config: Dict) -> str:
    """
    Configura MLflow y crea/obtiene el experimento.
    
    Returns:
        experiment_id: ID del experimento
    """
    experiment_name = config["EXPERIMENT_NAME"]
    
    # Configurar tracking URI
    if config.get("MLFLOW_TRACKING_URI") is not None:
        mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    else:
        # Usar SQLite local por defecto
        PARENT_DIR = Path(__file__).parent
        TRACKING_DB = (PARENT_DIR / "mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
        print(f"✓ MLflow tracking URI: sqlite:///{TRACKING_DB.as_posix()}")
    
    # Crear o obtener experimento
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            ARTIFACT_ROOT = (PARENT_DIR / "mlflow_artifacts").resolve()
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=ARTIFACT_ROOT.as_uri()
            )
            print(f"✓ Experimento MLflow creado: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Experimento MLflow existente: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"⚠ Error al configurar MLflow: {e}")
        experiment_id = mlflow.set_experiment(experiment_name)
    
    return experiment_id
```

### Logging de Hiperparámetros

```python
with mlflow.start_run(experiment_id=experiment_id, run_name=config["RUN_NAME"]):
    # Log hiperparámetros
    mlflow.log_params({
        "n_channels": config["N_CHANNELS"],
        "seq_len": config["SEQ_LEN"],
        "cnn_out_channels": str(config["out_channels_list"]),
        "cnn_kernel_sizes": str(config["kernel_sizes"]),
        "hidden_size": config["HIDDEN_SIZE"],
        "num_layers": config["NUM_LAYERS"],
        "batch_size": config["BATCH_SIZE"],
        "learning_rate": config["LEARNING_RATE"],
        "num_epochs": config["NUM_EPOCHS"],
        "weight_decay": config["WEIGHT_DECAY"],
        # ... más hiperparámetros
    })
```

### Logging de Métricas Durante Entrenamiento

```python
# Dentro del loop de entrenamiento
for epoch in range(1, NUM_EPOCHS + 1):
    # ... entrenar y validar ...
    
    # Log métricas por época
    mlflow.log_metrics({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_macro": val_f1,
        "val_f1_normal": val_metrics["f1_normal"],
        "val_f1_anom": val_metrics["f1_anom"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "learning_rate": current_lr,
    }, step=epoch)  # step permite ver evolución en el tiempo
```

### Logging de Artefactos

```python
# Guardar curvas de entrenamiento
curves_path = save_training_curves(
    train_losses, train_accuracies, val_losses, val_f1_scores, config["OUTPUT_DIR"]
)
mlflow.log_artifact(str(curves_path))  # Guarda el archivo PNG

# Guardar matriz de confusión
cm_val_path, _ = save_confusion_matrix(
    val_metrics["confusion_matrix"], config["OUTPUT_DIR"], "val"
)
mlflow.log_artifact(str(cm_val_path))

# Guardar modelo completo
mlflow.pytorch.log_model(model, "model")  # Guarda modelo en formato MLflow
```

### Logging de Métricas Finales (Test)

```python
# Después del entrenamiento, evaluar en test
with mlflow.start_run(experiment_id=experiment_id, run_name=config["RUN_NAME"]):
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_normal": test_metrics["f1_normal"],
        "test_f1_anom": test_metrics["f1_anom"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
    })
    
    # Guardar matriz de confusión de test
    cm_test_path, _ = save_confusion_matrix(
        test_metrics["confusion_matrix"], config["OUTPUT_DIR"], "test"
    )
    mlflow.log_artifact(str(cm_test_path))
```

### Visualización en MLflow UI

Para ver los resultados:

```bash
# En terminal
cd Books
mlflow ui

# Abrir en navegador: http://localhost:5000
```

**Interfaz de MLflow**:
- **Experiments**: Lista de experimentos
- **Runs**: Comparación de runs dentro de un experimento
- **Métricas**: Gráficos de evolución de métricas por época
- **Parámetros**: Tabla de hiperparámetros
- **Artefactos**: Descarga de modelos y gráficos guardados

---

## 🪄 Orquestación con Prefect

Prefect 2.x se usa para orquestar el flujo completo con manejo de errores, logging y caching.

### Conceptos Básicos

- **Flow**: Función decorada con `@flow` que orquesta el proceso completo
- **Task**: Función decorada con `@task` que realiza una operación específica
- **Logging**: Prefect captura automáticamente prints y logs
- **Caching**: Puede cachear resultados de tasks (deshabilitado en estos notebooks con `NO_CACHE`)

### Definición de Tasks

```python
from prefect import task, flow
from prefect.tasks import NO_CACHE

@task(name="load_data", log_prints=True, cache_policy=NO_CACHE)
def task_load_data(config: Dict):
    """
    Tarea Prefect para cargar datos.
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Tupla con (train_loader, val_loader, test_loader, y_val, y_test)
    """
    print("📂 Cargando datos...")
    
    # Cargar datos
    X_train, y_train, X_val, y_val, X_test, y_test = load_tensor_data(
        config["DATA_DIR"]
    )
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_from_tensors(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config["BATCH_SIZE"],
        shuffle_train=True,
    )
    
    print("✓ Datos cargados y preparados")
    return train_loader, val_loader, test_loader, y_val.numpy(), y_test.numpy()
```

**Características del decorador `@task`**:
- `name`: Nombre identificable en la UI de Prefect
- `log_prints=True`: Captura todos los `print()` como logs
- `cache_policy=NO_CACHE`: No cachea resultados (siempre ejecuta)

### Definición del Flow Principal

```python
@flow(name="cnn1d_lstm_classification_training_flow", log_prints=True)
def cnn1d_lstm_classification_training_flow(config: Dict = None):
    """
    Flujo principal de Prefect que orquesta todo el proceso:
    1. Carga y preparación de datos
    2. Creación del modelo
    3. Entrenamiento
    4. Evaluación en test
    """
    if config is None:
        config = CONFIG
    
    print("🚀 Iniciando flujo de entrenamiento...")
    print(f"Experimento MLflow: {config['EXPERIMENT_NAME']}")
    
    # 1. Configurar MLflow
    experiment_id = setup_mlflow(config)
    
    # 2. Cargar datos (Task)
    dataloaders = task_load_data(config)
    train_loader, val_loader, test_loader, y_val, y_test = dataloaders
    
    # 3. Crear modelo
    print("🧠 Creando modelo...")
    model = create_model(config)
    
    # 4. Entrenar (Task)
    model, train_losses, train_accs, val_losses, val_f1_scores, best_f1, learning_rates = task_train_model(
        model, train_loader, val_loader, y_val, config, DEVICE, experiment_id
    )
    
    # 5. Evaluar en test (Task)
    test_metrics = task_evaluate_test(
        model, test_loader, y_test, DEVICE, config, experiment_id
    )
    
    print("\n" + "="*60)
    print("✅ FLUJO COMPLETADO")
    print("="*60)
    print(f"Mejor F1 en validación: {best_f1:.4f}")
    print(f"F1 en test: {test_metrics['f1_macro']:.4f}")
    
    return {
        "model": model,
        "test_metrics": test_metrics,
        "best_f1": best_f1,
    }
```

**Características del decorador `@flow`**:
- `name`: Nombre del flow en la UI de Prefect
- `log_prints=True`: Captura todos los prints

### Task de Entrenamiento

```python
@task(name="train_model", log_prints=True, cache_policy=NO_CACHE)
def task_train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    config: Dict,
    device: torch.device,
    experiment_id: str,
):
    """Tarea Prefect para entrenar el modelo."""
    print("🏋️ Iniciando entrenamiento...")
    
    # Mover modelo a dispositivo
    model = model.to(device)
    
    # Inicializar optimizador y criterio
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    criterion = nn.BCELoss()
    
    # Learning Rate Scheduler
    scheduler = None
    if config.get("USE_SCHEDULER", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("SCHEDULER_MODE", "max"),
            factor=config.get("SCHEDULER_FACTOR", 0.5),
            patience=config.get("SCHEDULER_PATIENCE", 5),
        )
    
    # Iniciar run de MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name=config["RUN_NAME"]):
        # Log hiperparámetros
        mlflow.log_params({...})
        
        # Loop de entrenamiento
        for epoch in range(1, config["NUM_EPOCHS"] + 1):
            # ... entrenar y validar ...
            
            # Log métricas
            mlflow.log_metrics({...}, step=epoch)
        
        # Guardar artefactos
        mlflow.log_artifact(...)
        mlflow.pytorch.log_model(model, "model")
    
    return model, train_losses, train_accs, val_losses, val_f1_scores, best_f1, learning_rates
```

### Ventajas de Prefect

1. **Logging Automático**: Todos los prints se capturan automáticamente
2. **Manejo de Errores**: Prefect maneja errores y permite reintentos
3. **UI de Monitoreo**: Puedes ver el progreso en tiempo real (si configuras Prefect Cloud o servidor local)
4. **Reproducibilidad**: Cada ejecución queda registrada con timestamps y estados
5. **Paralelización**: Puedes ejecutar múltiples flows en paralelo

### Ejecución del Flow

```python
# En la última celda del notebook
if __name__ == "__main__":
    results = cnn1d_lstm_classification_training_flow(CONFIG)
    print("\n✓ Proceso finalizado exitosamente")
```

**Salida de Prefect**:
```
2025-11-24 16:46:00 INFO  [prefect.flow_runs] Beginning flow run 'uppish-rabbit' for flow 'cnn1d_lstm_classification_training_flow'
2025-11-24 16:46:00 INFO  [prefect.flow_runs] 🚀 Iniciando flujo de entrenamiento...
2025-11-24 16:46:00 INFO  [prefect.task_runs] 📂 Cargando datos...
2025-11-24 16:46:07 INFO  [prefect.task_runs] ✓ Datos cargados y preparados
2025-11-24 16:46:08 INFO  [prefect.flow_runs] 🧠 Creando modelo...
2025-11-24 16:46:08 INFO  [prefect.task_runs] 🏋️ Iniciando entrenamiento...
...
```

---

## 🔄 Flujo Completo de Entrenamiento

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SETUP CUDA Y DEPENDENCIAS                               │
│    - Configurar DLLs CUDA                                  │
│    - Instalar dependencias                                 │
│    - Verificar GPU                                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CONFIGURACIÓN GENERAL                                    │
│    - Definir hiperparámetros                                │
│    - Configurar rutas                                      │
│    - Fijar semillas                                         │
│    - Configurar dispositivo                                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CARGA Y PREPARACIÓN DE DATOS                             │
│    - Cargar tensores desde disco                           │
│    - Crear DataLoaders                                      │
│    - (Para autoencoders: filtrar normales en train)        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. DEFINICIÓN DEL MODELO                                    │
│    - Crear instancia del modelo                             │
│    - Contar parámetros                                     │
│    - Verificar arquitectura                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. CONFIGURACIÓN MLFLOW                                     │
│    - Crear/obtener experimento                             │
│    - Configurar tracking URI                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. FLUJO PREFECT (ORQUESTACIÓN)                             │
│    │                                                         │
│    ├─→ Task: Cargar datos                                   │
│    │                                                         │
│    ├─→ Crear modelo                                         │
│    │                                                         │
│    ├─→ Task: Entrenar modelo                                │
│    │   │                                                     │
│    │   ├─→ Iniciar run MLflow                               │
│    │   │                                                     │
│    │   ├─→ Log hiperparámetros                              │
│    │   │                                                     │
│    │   ├─→ Loop de entrenamiento (N épocas)                 │
│    │   │   │                                                 │
│    │   │   ├─→ Entrenar una época                          │
│    │   │   │   - Forward pass                               │
│    │   │   │   - Backward pass                              │
│    │   │   │   - Optimizer step                             │
│    │   │   │                                                 │
│    │   │   ├─→ Validar                                      │
│    │   │   │   - Forward pass (eval mode)                  │
│    │   │   │   - Calcular métricas                         │
│    │   │   │                                                 │
│    │   │   ├─→ Log métricas en MLflow                      │
│    │   │   │                                                 │
│    │   │   └─→ Actualizar scheduler                        │
│    │   │                                                     │
│    │   ├─→ Guardar mejor modelo                            │
│    │   │                                                     │
│    │   ├─→ Guardar curvas de entrenamiento                │
│    │   │                                                     │
│    │   ├─→ Guardar matriz de confusión                    │
│    │   │                                                     │
│    │   └─→ Log modelo en MLflow                            │
│    │                                                         │
│    └─→ Task: Evaluar en test                               │
│        │                                                     │
│        ├─→ Evaluar modelo                                   │
│        │                                                     │
│        ├─→ Calcular métricas completas                      │
│        │                                                     │
│        └─→ Log métricas y artefactos en MLflow             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. GUARDADO DE MODELO                                       │
│    - Guardar en formato estándar PyTorch                   │
│    - Exportar para AWS SageMaker (opcional)                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. PRUEBA DE PREDICCIONES                                   │
│    - Visualizar ejemplos del test set                      │
│    - Mostrar predicciones vs etiquetas reales              │
└─────────────────────────────────────────────────────────────┘
```

### Secuencia Temporal

1. **Setup** (1-2 minutos): Configuración inicial
2. **Carga de datos** (5-10 minutos): Cargar tensores desde disco
3. **Creación de modelo** (<1 minuto): Instanciar modelo
4. **Entrenamiento** (2-4 horas): Loop principal
   - Por época: ~5-10 minutos
   - Total: Depende de `NUM_EPOCHS`
5. **Evaluación en test** (5-10 minutos): Evaluar modelo final
6. **Guardado** (<1 minuto): Exportar modelos

---

## 🎨 Guía de Personalización

### Modificar Arquitectura del Modelo

1. **Agregar capas CNN**:
```python
# En CONFIG
out_channels_list = [32, 64, 128, 256]  # Agregar más capas
kernel_sizes = [7, 5, 3, 3]
pool_sizes = [2, 2, 2, 2]
```

2. **Cambiar tamaño de LSTM**:
```python
HIDDEN_SIZE = 256  # Aumentar de 128 a 256
NUM_LAYERS = 3     # Aumentar de 2 a 3
```

3. **Modificar capas Fully Connected**:
```python
FC_UNITS = 128  # Aumentar de 64 a 128
# O agregar múltiples capas FC en la clase del modelo
```

### Agregar Nuevas Métricas

1. **En función `compute_metrics`**:
```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    # ... métricas existentes ...
    
    # Agregar nueva métrica
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, y_pred)
    
    return {
        # ... métricas existentes ...
        "roc_auc": roc_auc,
    }
```

2. **Log en MLflow**:
```python
mlflow.log_metrics({
    # ... métricas existentes ...
    "val_roc_auc": val_metrics["roc_auc"],
}, step=epoch)
```

### Cambiar Optimizador

```python
# En función de entrenamiento
# De Adam a SGD con momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=config["LEARNING_RATE"],
    momentum=0.9,
    weight_decay=config["WEIGHT_DECAY"],
)
```

### Agregar Early Stopping

```python
# En loop de entrenamiento
best_f1 = 0.0
patience = 5
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ... entrenar y validar ...
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        # Guardar mejor modelo
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping en época {epoch}")
            break
```

### Personalizar Loss Function

```python
# Para problemas con clases desbalanceadas
class_weight = torch.tensor([1.0, 2.0])  # Pesar más la clase anómala
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight[1] / class_weight[0])

# O usar Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=1, gamma=2)
```

---

## 📝 Resumen

Esta documentación cubre:

✅ **Estructura completa** de los notebooks de entrenamiento  
✅ **Arquitecturas detalladas** de todos los modelos  
✅ **Integración MLflow** paso a paso  
✅ **Orquestación Prefect** con ejemplos  
✅ **Flujo completo** de entrenamiento  
✅ **Guías de personalización**

Para más información sobre el proyecto completo, ver [DOCUMENTACION_GENERAL.md](DOCUMENTACION_GENERAL.md).

---

**Última actualización**: 2025-01-XX

