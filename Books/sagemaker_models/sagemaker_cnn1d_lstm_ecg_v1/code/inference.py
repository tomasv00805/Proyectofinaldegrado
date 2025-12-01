"""
Script de inferencia para AWS SageMaker.
SageMaker llama a estas funciones automáticamente.
"""
import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# Definición del modelo
# ========================================
class CNN1D_LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        out_channels_list: list,
        kernel_sizes: list,
        pool_sizes: list,
        use_batchnorm: bool = True,
        cnn_activation: str = "relu",
        cnn_dropout: float = 0.0,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_units: int = 32,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        # CNN1D Blocks
        cnn_layers = []
        in_channels = n_channels
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(out_channels_list, kernel_sizes, pool_sizes)
        ):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            if use_batchnorm:
                cnn_layers.append(nn.BatchNorm1d(out_channels))
            if cnn_activation == "relu":
                cnn_layers.append(nn.ReLU())
            if cnn_dropout > 0:
                cnn_layers.append(nn.Dropout(cnn_dropout))
            if pool_size is not None and pool_size > 1:
                cnn_layers.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calcular tamaño de salida de CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, seq_len)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_channels = cnn_output.shape[1]
            self.cnn_output_seq_len = cnn_output.shape[2]

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
        self.fc1 = nn.Linear(hidden_size, fc_units)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            print(f"[forward] Input shape: {x.shape}, dtype: {x.dtype}")
            
            # Verificar y ajustar formato de entrada
            if len(x.shape) == 3 and x.shape[1] == self.seq_len and x.shape[2] == self.n_channels:
                print(f"[forward] Transponiendo de [batch, seq_len, channels] a [batch, channels, seq_len]")
                x = x.transpose(1, 2)
                print(f"[forward] Después de transpose: {x.shape}")

            # CNN
            print(f"[forward] Ejecutando CNN...")
            cnn_out = self.cnn(x)
            print(f"[forward] CNN output shape: {cnn_out.shape}")
            
            # Transponer para LSTM: [batch, channels, seq] -> [batch, seq, channels]
            cnn_out = cnn_out.transpose(1, 2)
            print(f"[forward] CNN output después de transpose: {cnn_out.shape}")

            # LSTM
            print(f"[forward] Ejecutando LSTM...")
            lstm_out, (hidden, cell) = self.lstm(cnn_out)
            print(f"[forward] LSTM output shape: {lstm_out.shape}")
            print(f"[forward] Hidden shape: {hidden.shape}, Cell shape: {cell.shape}")
            
            # Obtener último hidden state
            # Para LSTM bidireccional con num_layers=2:
            # hidden tiene forma (4, batch_size, hidden_size)
            # hidden[0] = forward capa 0, hidden[1] = backward capa 0
            # hidden[2] = forward capa 1, hidden[3] = backward capa 1
            last_hidden = hidden[-1]
            print(f"[forward] Last hidden shape: {last_hidden.shape}")

            # Fully Connected
            print(f"[forward] Ejecutando FC1...")
            out = self.fc1(last_hidden)
            print(f"[forward] FC1 output shape: {out.shape}")
            
            out = self.relu(out)
            out = self.dropout_fc(out)
            
            print(f"[forward] Ejecutando FC2...")
            out = self.fc2(out)
            print(f"[forward] FC2 output shape: {out.shape}")
            
            out = self.sigmoid(out)
            print(f"[forward] Sigmoid output shape: {out.shape}")
            
            result = out.squeeze(-1)
            print(f"[forward] Final output shape: {result.shape}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error en forward: {e}"
            print(f"[forward] ERROR: {error_msg}")
            print(f"[forward] Input shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
            print(f"[forward] Input dtype: {x.dtype if hasattr(x, 'dtype') else 'N/A'}")
            import traceback
            print(traceback.format_exc())
            raise


# ========================================
# Funciones requeridas por SageMaker
# ========================================
def model_fn(model_dir):
    """Carga el modelo desde el directorio."""
    import logging
    import time
    
    # Configurar logging para que se vea en CloudWatch
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # También usar print para asegurar que se vea en los logs
    print("=" * 70)
    print("INICIANDO CARGA DEL MODELO")
    print("=" * 70)
    print(f"Cargando modelo desde {model_dir}")
    
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")
    logger.info(f"Usando dispositivo: {device}")

    try:
        # Cargar configuración
        config_path = os.path.join(model_dir, "config.json")
        print(f"Leyendo config desde {config_path}")
        logger.info(f"Leyendo config desde {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        model_config = config['model_config']
        print(f"Configuración del modelo cargada")
        logger.info(f"Configuración del modelo: {model_config}")

        # Crear modelo
        print("Creando arquitectura del modelo...")
        logger.info("Creando modelo...")
        model = CNN1D_LSTMClassifier(
            n_channels=model_config['n_channels'],
            seq_len=model_config['seq_len'],
            out_channels_list=model_config['out_channels_list'],
            kernel_sizes=model_config['kernel_sizes'],
            pool_sizes=model_config['pool_sizes'],
            use_batchnorm=model_config['use_batchnorm'],
            cnn_activation=model_config['cnn_activation'],
            cnn_dropout=model_config.get('cnn_dropout', 0.0),
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            fc_units=model_config['fc_units'],
        )
        print("Arquitectura del modelo creada")

        # Cargar pesos
        model_path = os.path.join(model_dir, "model.pth")
        print(f"Cargando pesos desde {model_path}")
        logger.info(f"Cargando pesos desde {model_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
        
        # Optimizar carga: usar map_location directamente y weights_only=False si es necesario
        print("Cargando checkpoint...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Checkpoint cargado, keys: {list(checkpoint.keys())}")
        
        # Verificar que tiene model_state_dict
        if 'model_state_dict' not in checkpoint:
            print(f"⚠️ ADVERTENCIA: 'model_state_dict' no encontrado en checkpoint")
            print(f"   Keys disponibles: {list(checkpoint.keys())}")
            # Intentar usar 'state_dict' como alternativa
            if 'state_dict' in checkpoint:
                print("   Usando 'state_dict' como alternativa...")
                checkpoint = {'model_state_dict': checkpoint['state_dict']}
            else:
                # Si no hay ninguno, asumir que el checkpoint ES el state_dict
                print("   Asumiendo que el checkpoint completo es el state_dict...")
                checkpoint = {'model_state_dict': checkpoint}
        
        print("Cargando state_dict en el modelo...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Pesos cargados exitosamente")

        # Mover a dispositivo y poner en modo evaluación
        print(f"Moviendo modelo a {device}...")
        model.to(device)
        model.eval()
        
        # Hacer una inferencia de prueba para "calentar" el modelo
        print("Realizando inferencia de prueba (warm-up)...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, model_config['seq_len'], model_config['n_channels']).to(device)
            _ = model(dummy_input)
        print("Inferencia de prueba completada")
        
        elapsed_time = time.time() - start_time
        print(f"Modelo cargado exitosamente en {elapsed_time:.2f} segundos")
        print("=" * 70)
        logger.info("Modelo cargado y listo para inferencia")

        return model
    except Exception as e:
        error_msg = f"Error cargando modelo: {e}"
        print(f"ERROR: {error_msg}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        logger.error(error_msg, exc_info=True)
        raise


def input_fn(request_body, request_content_type):
    """Preprocesa la entrada."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        print(f"[input_fn] Content type: {request_content_type}")
        print(f"[input_fn] Request body type: {type(request_body)}")
        print(f"[input_fn] Request body size: {len(request_body) if isinstance(request_body, (str, bytes)) else 'N/A'} bytes")
        
        if request_content_type == 'application/json':
            print("[input_fn] Procesando JSON...")
            
            # Manejar tanto bytes como string
            if isinstance(request_body, bytes):
                print("[input_fn] Request body es bytes, decodificando a string...")
                try:
                    request_body = request_body.decode('utf-8')
                    print("[input_fn] Decodificado exitosamente a UTF-8")
                except UnicodeDecodeError as e:
                    print(f"[input_fn] Error decodificando UTF-8: {e}")
                    # Intentar con latin-1 como fallback
                    request_body = request_body.decode('latin-1')
                    print("[input_fn] Decodificado con latin-1 como fallback")
            
            # Parsear JSON
            print(f"[input_fn] Parseando JSON (tipo: {type(request_body)})...")
            data = json.loads(request_body)
            print(f"[input_fn] JSON parseado exitosamente, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            if 'signals' in data:
                signals = np.array(data['signals'], dtype=np.float32)
                print(f"[input_fn] Encontrado 'signals' en data")
            elif 'signal' in data:
                signals = np.array([data['signal']], dtype=np.float32)
                print(f"[input_fn] Encontrado 'signal' en data")
            else:
                signals = np.array(data, dtype=np.float32)
                if len(signals.shape) == 2:
                    signals = signals[np.newaxis, :, :]
                print(f"[input_fn] Usando data directamente")

            print(f"[input_fn] Signals shape: {signals.shape}")
            print(f"[input_fn] Signals dtype: {signals.dtype}")
            print(f"[input_fn] Signals range: [{signals.min():.4f}, {signals.max():.4f}]")
            
            # Validar forma esperada: [batch, seq_len, n_channels] = [1, 2000, 3]
            if len(signals.shape) != 3:
                raise ValueError(f"Se espera forma 3D [batch, seq_len, channels], pero se recibió: {signals.shape}")
            
            batch_size, seq_len, n_channels = signals.shape
            print(f"[input_fn] Batch size: {batch_size}, Seq len: {seq_len}, Channels: {n_channels}")
            
            # Validar dimensiones esperadas
            if seq_len != 2000:
                print(f"[input_fn] ADVERTENCIA: seq_len={seq_len}, se esperaba 2000")
            if n_channels != 3:
                print(f"[input_fn] ADVERTENCIA: n_channels={n_channels}, se esperaba 3")
            
            logger.info(f"Input shape: {signals.shape}")
            
            # Convertir a tensor
            tensor = torch.from_numpy(signals)
            print(f"[input_fn] Tensor creado, shape: {tensor.shape}, dtype: {tensor.dtype}")
            
            # Asegurar que sea float32
            if tensor.dtype != torch.float32:
                print(f"[input_fn] Convirtiendo dtype de {tensor.dtype} a float32")
                tensor = tensor.float()
            
            return tensor

        elif request_content_type == 'application/x-npy':
            print("[input_fn] Procesando NPY...")
            import io
            array = np.load(io.BytesIO(request_body))
            if len(array.shape) == 2:
                array = array[np.newaxis, :, :]
            print(f"[input_fn] Array shape: {array.shape}")
            logger.info(f"Input shape: {array.shape}")
            return torch.from_numpy(array.astype(np.float32))

        else:
            error_msg = f"Content type {request_content_type} no soportado"
            print(f"[input_fn] ERROR: {error_msg}")
            raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error en input_fn: {e}"
        print(f"[input_fn] ERROR: {error_msg}")
        import traceback
        print(traceback.format_exc())
        logger.error(error_msg, exc_info=True)
        raise


def predict_fn(input_data, model):
    """Hace la predicción."""
    import logging
    import time
    
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[predict_fn] Dispositivo: {device}")
        logger.info(f"Dispositivo para predicción: {device}")
        
        print(f"[predict_fn] Input data type: {type(input_data)}")
        if hasattr(input_data, 'shape'):
            print(f"[predict_fn] Input shape: {input_data.shape}")
        logger.info(f"Input data type: {type(input_data)}, shape: {input_data.shape if hasattr(input_data, 'shape') else 'N/A'}")
        
        model.eval()

        with torch.no_grad():
            # Convertir a tensor si es necesario
            if not isinstance(input_data, torch.Tensor):
                print("[predict_fn] Convirtiendo input a tensor...")
                input_data = torch.tensor(input_data, dtype=torch.float32)

            # Mover a dispositivo
            print(f"[predict_fn] Moviendo input a {device}...")
            input_data = input_data.to(device)
            print(f"[predict_fn] Input en dispositivo, shape: {input_data.shape}")
            logger.info(f"Input data en dispositivo, shape: {input_data.shape}")
            
            # Ejecutar inferencia
            print("[predict_fn] Ejecutando forward pass...")
            logger.info("Ejecutando forward pass...")
            inference_start = time.time()
            probabilities = model(input_data)
            inference_time = time.time() - inference_start
            print(f"[predict_fn] Forward pass completado en {inference_time:.3f} segundos")
            
            if hasattr(probabilities, 'shape'):
                print(f"[predict_fn] Predicción shape: {probabilities.shape}")
            logger.info(f"Predicción completada, shape: {probabilities.shape if hasattr(probabilities, 'shape') else 'N/A'}")

            # Convertir a numpy y asegurar formato correcto
            if isinstance(probabilities, torch.Tensor):
                print("[predict_fn] Convirtiendo tensor a numpy...")
                probabilities = probabilities.cpu().numpy()
                print(f"[predict_fn] Convertido a numpy, shape: {probabilities.shape}, dtype: {probabilities.dtype}")
            
            # Asegurar que sea un array numpy
            if not isinstance(probabilities, np.ndarray):
                print(f"[predict_fn] Convirtiendo a numpy array (tipo actual: {type(probabilities)})")
                probabilities = np.array(probabilities)
            
            # Asegurar que sea float32
            if probabilities.dtype != np.float32:
                print(f"[predict_fn] Convirtiendo dtype de {probabilities.dtype} a float32")
                probabilities = probabilities.astype(np.float32)
            
            # Aplanar si es necesario (asegurar que sea 1D)
            if len(probabilities.shape) > 1:
                print(f"[predict_fn] Aplanando array de shape {probabilities.shape}")
                probabilities = probabilities.flatten()
            
            print(f"[predict_fn] Probabilidades finales: shape={probabilities.shape}, dtype={probabilities.dtype}, valores={probabilities}")
            logger.info(f"Convertido a numpy, shape: {probabilities.shape}, dtype: {probabilities.dtype}")

        total_time = time.time() - start_time
        print(f"[predict_fn] Predicción total completada en {total_time:.3f} segundos")
        
        # Asegurar que retornamos un numpy array
        return probabilities
    except Exception as e:
        error_msg = f"Error en predict_fn: {e}"
        print(f"[predict_fn] ERROR: {error_msg}")
        import traceback
        print(traceback.format_exc())
        logger.error(error_msg, exc_info=True)
        raise


def output_fn(prediction, accept):
    """Formatea la salida."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        print(f"[output_fn] Iniciando formateo de salida")
        print(f"[output_fn] Prediction type: {type(prediction)}")
        print(f"[output_fn] Accept header: {accept}")
        
        # Convertir prediction a un formato manejable
        if isinstance(prediction, torch.Tensor):
            print("[output_fn] Convirtiendo tensor a numpy...")
            prediction = prediction.cpu().numpy()
        
        # Manejar diferentes formatos de accept
        if accept == 'application/json' or accept is None:
            print("[output_fn] Formateando como JSON...")
            
            if isinstance(prediction, np.ndarray):
                # Si es un array, convertir a lista
                pred_list = prediction.tolist()
                print(f"[output_fn] Array convertido a lista, longitud: {len(pred_list)}")
                
                # Si es un array 1D con un solo elemento, extraerlo
                if len(pred_list) == 1:
                    pred_value = pred_list[0]
                    result = {
                        'prediction': float(pred_value),
                        'probability': float(pred_value)
                    }
                else:
                    result = {
                        'predictions': pred_list,
                        'probabilities': pred_list
                    }
            elif isinstance(prediction, (list, tuple)):
                print(f"[output_fn] Es lista/tupla, longitud: {len(prediction)}")
                if len(prediction) == 1:
                    pred_value = float(prediction[0])
                    result = {
                        'prediction': pred_value,
                        'probability': pred_value
                    }
                else:
                    result = {
                        'predictions': [float(p) for p in prediction],
                        'probabilities': [float(p) for p in prediction]
                    }
            else:
                # Es un escalar
                print(f"[output_fn] Es escalar: {prediction}")
                pred_value = float(prediction)
                result = {
                    'prediction': pred_value,
                    'probability': pred_value
                }
            
            print(f"[output_fn] Resultado formateado: {result}")
            json_result = json.dumps(result)
            print(f"[output_fn] JSON creado, longitud: {len(json_result)}")
            return json_result

        elif accept == 'text/csv':
            print("[output_fn] Formateando como CSV...")
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            elif isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy().tolist()
            csv_result = ','.join([str(float(p)) for p in prediction])
            print(f"[output_fn] CSV creado: {csv_result}")
            return csv_result

        else:
            print(f"[output_fn] Accept no reconocido ({accept}), usando JSON por defecto...")
            # Fallback a JSON
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            elif isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy().tolist()
            
            if isinstance(prediction, list) and len(prediction) == 1:
                result = {'predictions': [float(prediction[0])]}
            else:
                result = {'predictions': [float(p) for p in prediction] if isinstance(prediction, list) else [float(prediction)]}
            
            return json.dumps(result)
            
    except Exception as e:
        error_msg = f"Error en output_fn: {e}"
        print(f"[output_fn] ERROR: {error_msg}")
        import traceback
        print(traceback.format_exc())
        logger.error(error_msg, exc_info=True)
        
        # Retornar un error JSON en lugar de crashear
        error_result = {
            'error': str(e),
            'error_type': type(e).__name__,
            'prediction_type': str(type(prediction)),
            'accept': str(accept)
        }
        return json.dumps(error_result)
