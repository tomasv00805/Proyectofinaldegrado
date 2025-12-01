# 📚 Documentación: Despliegue de Modelo en SageMaker Serverless

## 📑 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Objetivo](#-objetivo)
3. [Requisitos Previos](#-requisitos-previos)
4. [Costos](#-costos)
5. [Estructura del Notebook](#-estructura-del-notebook)
6. [Guía de Uso](#-guía-de-uso)
7. [Configuración Detallada](#-configuración-detallada)
8. [Solución de Problemas](#-solución-de-problemas)
9. [Monitoreo y Logs](#-monitoreo-y-logs)
10. [Uso del Endpoint desde Código](#-uso-del-endpoint-desde-código)
11. [Eliminación del Endpoint](#️-eliminación-del-endpoint)
12. [Recursos Adicionales](#-recursos-adicionales)
13. [Checklist de Despliegue](#-checklist-de-despliegue)
14. [Soporte](#-soporte)

---

## 📋 Descripción General

Este notebook (`deploy_sagemaker_serverless.ipynb`) proporciona una guía paso a paso para desplegar un modelo de clasificación de ECG (CNN1D-LSTM) en AWS SageMaker usando **Serverless Inference**. El modelo está diseñado para detectar anomalías en señales de electrocardiograma.

### Características Principales

- ✅ Despliegue automatizado en AWS SageMaker
- ✅ Configuración de Serverless Inference (pago por uso)
- ✅ Verificación de credenciales y configuración
- ✅ Pruebas del endpoint con datos reales
- ✅ Diagnóstico y solución de problemas
- ✅ Monitoreo de logs en CloudWatch

---

## 🎯 Objetivo

Desplegar un modelo de deep learning entrenado (CNN1D-LSTM) como endpoint de inferencia en AWS SageMaker con las siguientes características:

- **Modo Serverless**: Solo se cobra por invocación, sin costo cuando está inactivo
- **Alta disponibilidad**: El endpoint está disponible 24/7
- **Escalabilidad automática**: SageMaker maneja el escalado automáticamente
- **Fácil integración**: API REST para invocaciones desde cualquier aplicación

---

## 📦 Requisitos Previos

### 1. Cuenta AWS

- Cuenta AWS activa con permisos de facturación
- Acceso a los servicios: SageMaker, S3, IAM, CloudWatch

### 2. Credenciales AWS

Necesitas obtener:
- **Access Key ID**: Identificador de acceso
- **Secret Access Key**: Clave secreta de acceso

**Cómo obtenerlas:**
1. Ve a https://console.aws.amazon.com
2. IAM → Usuarios → Tu usuario → Security credentials
3. Create access key
4. Descarga el CSV (solo se muestra una vez)

### 3. Rol IAM

Un rol IAM con los siguientes permisos:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (o permisos específicos en tu bucket)

**Cómo crearlo:**
1. IAM → Roles → Create role
2. Selecciona: "SageMaker"
3. Adjunta las políticas mencionadas
4. Copia el ARN del rol (formato: `arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME`)

### 4. Archivo del Modelo

El archivo comprimido del modelo debe estar en:
```
sagemaker_models/cnn1d_lstm_ecg_v1_sagemaker.tar.gz
```

**Estructura del archivo .tar.gz:**
```
cnn1d_lstm_ecg_v1_sagemaker.tar.gz
├── model.pth                    # Modelo entrenado
├── config.json                  # Configuración del modelo
└── code/
    ├── inference.py            # Código de inferencia
    └── requirements.txt        # Dependencias Python
```

### 5. Dependencias Python

El notebook instalará automáticamente:
- `boto3` (SDK de AWS para Python)
- `sagemaker` (versión 2.x, compatible con este notebook)

---

## 💰 Costos

### Serverless Inference

- **Por inferencia:** ~$0.00022 USD
- **Sin tráfico:** $0 USD (no hay costo cuando está inactivo)
- **Memoria configurada:** 3072 MB (3 GB)

### Ejemplos de Costo Mensual

| Inferencias/mes | Costo Aproximado |
|----------------|------------------|
| 1,000          | $0.22            |
| 10,000         | $2.20            |
| 100,000        | $22.00           |
| 1,000,000      | $220.00          |

**Nota:** Los costos pueden variar según la región y el tiempo de procesamiento real.

---

## 📝 Estructura del Notebook

El notebook está organizado en los siguientes pasos:

### Paso 0: Instalación de Dependencias
- Verifica e instala `boto3` y `sagemaker`
- Detecta y corrige problemas de versión (especialmente sagemaker 3.x → 2.x)
- Valida que las dependencias funcionen correctamente

### Paso 1: Configuración
**⚠️ IMPORTANTE: Edita esta celda con tus valores**

Configura:
- `AWS_ACCESS_KEY_ID`: Tu Access Key ID
- `AWS_SECRET_ACCESS_KEY`: Tu Secret Access Key
- `SAGEMAKER_ROLE_ARN`: ARN del rol IAM
- `AWS_REGION`: Región AWS (ej: "us-east-1")
- `ENDPOINT_NAME`: Nombre único para el endpoint
- `SERVERLESS_MEMORY_MB`: Memoria en MB (default: 3072)
- `SERVERLESS_MAX_CONCURRENCY`: Máximo de invocaciones simultáneas (default: 10)

### Paso 1.5: Diagnóstico Rápido (Opcional)
- Diagnostica problemas con sagemaker
- Reinstala automáticamente si es necesario
- Útil si encuentras errores de importación

### Paso 2: Verificación de Archivos
- Verifica que el archivo del modelo exista
- Muestra el tamaño del archivo
- Valida la ruta

### Paso 3: Configuración y Verificación de Credenciales AWS
- Configura las credenciales en el entorno
- Verifica que las credenciales sean válidas
- Muestra información de la cuenta AWS

### Paso 4: Configuración del Rol de SageMaker
- Obtiene o configura el rol IAM
- Valida que el rol tenga los permisos necesarios

### Paso 5: Configuración de Sesión de SageMaker
- Crea la sesión de SageMaker
- Configura el bucket S3 (usa el bucket por defecto si no se especifica)

### Paso 5.5: Recrear Modelo .tar.gz (Opcional)
- Recrea el archivo .tar.gz si actualizaste el código de inferencia
- Valida la estructura del archivo

### Paso 5.6: Sobrescribir Modelo en S3 (Opcional)
- Sube una versión actualizada del modelo a S3
- Útil si corregiste problemas en el código de inferencia

### Paso 6: Subir Modelo a S3
- Sube el archivo .tar.gz a S3
- Verifica si ya existe (no sobrescribe por defecto)
- Muestra la URI del modelo en S3

### Paso 7: Crear Modelo en SageMaker
- Registra el modelo en SageMaker
- Configura el framework (PyTorch 2.0.0, Python 3.10)
- Especifica el punto de entrada (`inference.py`)

### Paso 8: Configurar Serverless Inference
- Configura la memoria y concurrencia máxima
- Muestra información de costos

### Paso 9: Desplegar Endpoint
**⏱️ Este paso puede tardar 5-10 minutos**

- Elimina endpoints/configuraciones existentes (si aplica)
- Crea el endpoint serverless
- Espera a que esté en estado "InService"

### Paso 10: Probar el Endpoint
- Carga un ECG real desde los datos de prueba
- Envía una petición al endpoint
- Muestra la respuesta y la interpretación

### Paso 11: Probar Endpoint Específico
- Permite probar cualquier endpoint especificando su nombre
- Útil para probar endpoints existentes o en diferentes regiones

### Paso 12: Verificar y Probar Endpoint desde URL
- Extrae información de una URL de endpoint
- Verifica el estado del endpoint
- Prueba el endpoint usando la URL completa

### Paso 13: Ver Logs de CloudWatch
- Muestra los logs recientes del endpoint
- Útil para diagnosticar problemas

### Resumen Final
- Muestra información completa del endpoint desplegado
- Instrucciones de uso
- Enlaces a monitoreo

### Eliminar Endpoint (Opcional)
- Código para eliminar el endpoint y el modelo
- Deshabilitado por defecto por seguridad

---

## 🚀 Guía de Uso

### Ejecución Paso a Paso

1. **Abre el notebook** en Jupyter o VS Code
2. **Ejecuta el Paso 0** para instalar dependencias
3. **Edita el Paso 1** con tus credenciales AWS
4. **Ejecuta los pasos en orden** (Shift + Enter en cada celda)
5. **Espera** durante el Paso 9 (despliegue, 5-10 minutos)
6. **Prueba** el endpoint en el Paso 10

### Ejecución Rápida

Si ya tienes todo configurado y solo quieres probar un endpoint existente:
- Ejecuta el Paso 11 o Paso 12 directamente

---

## 🔧 Configuración Detallada

### Variables de Configuración (Paso 1)

```python
# Credenciales AWS
AWS_ACCESS_KEY_ID = "TU_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "TU_SECRET_ACCESS_KEY"

# Configuración SageMaker
SAGEMAKER_ROLE_ARN = "arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME"
AWS_REGION = "us-east-1"  # Cambia según tu región
ENDPOINT_NAME = "cnn1d-lstm-ecg-v1-serverless"

# Configuración Serverless
SERVERLESS_MEMORY_MB = 3072  # Memoria en MB (3GB)
SERVERLESS_MAX_CONCURRENCY = 10  # Máximo de invocaciones simultáneas
```

### Formatos de Datos

El modelo espera datos en el siguiente formato:

**Entrada:**
```json
{
  "signals": [
    [
      [valor1_canal1, valor1_canal2, valor1_canal3],
      [valor2_canal1, valor2_canal2, valor2_canal3],
      ...
    ]
  ]
}
```

- **Forma:** `[1, 2000, 3]` (1 muestra, 2000 puntos de tiempo, 3 canales)
- **Tipo:** `float32`
- **Rango:** Normalizado (típicamente [0, 1] o [-1, 1])

**Salida:**
```json
{
  "prediction": 0.95,
  "probability": 0.95
}
```

- `prediction`: Probabilidad de anomalía (0-1)
- `probability`: Misma probabilidad (redundante)
- **Threshold:** > 0.5 = Anómalo, ≤ 0.5 = Normal

---

## 🐛 Solución de Problemas

### Error: "sagemaker.Session no disponible"

**Causa:** Versión incorrecta de sagemaker (3.x en lugar de 2.x)

**Solución:**
1. Ejecuta el Paso 1.5 (Diagnóstico Rápido)
2. O manualmente:
   ```bash
   pip uninstall sagemaker sagemaker-core sagemaker-mlops sagemaker-serve sagemaker-train -y
   pip install 'sagemaker<3.0'
   ```
3. Reinicia el kernel y vuelve a ejecutar desde el Paso 0

### Error: "No se pueden verificar las credenciales AWS"

**Causa:** Credenciales incorrectas o no configuradas

**Solución:**
1. Verifica que las credenciales en el Paso 1 sean correctas
2. Asegúrate de que no tengan espacios extra
3. Verifica que el usuario IAM tenga permisos necesarios

### Error: "Endpoint no encontrado"

**Causa:** El endpoint no existe o está en otra región

**Solución:**
1. Verifica el nombre del endpoint en la consola AWS
2. Verifica que estés usando la región correcta
3. Asegúrate de que el endpoint esté en estado "InService"

### Error: "Modelo no encontrado"

**Causa:** El archivo .tar.gz no existe en la ruta especificada

**Solución:**
1. Verifica que el archivo exista en `sagemaker_models/`
2. Verifica el nombre del archivo (debe ser exacto)
3. Si actualizaste el código, ejecuta el Paso 5.5 para recrear el .tar.gz

### Error: "Timeout en la inferencia"

**Causa:** Memoria insuficiente o modelo muy grande

**Solución:**
1. Aumenta `SERVERLESS_MEMORY_MB` en el Paso 1 (ej: 4096 o 6144)
2. Recrea el endpoint con la nueva configuración

### Error: "Estructura incorrecta del tar.gz"

**Causa:** El archivo .tar.gz no tiene la estructura esperada

**Solución:**
1. Ejecuta el Paso 5.5 para recrear el archivo
2. Verifica que tenga:
   - `model.pth` en la raíz
   - `config.json` en la raíz
   - `code/inference.py`
   - `code/requirements.txt`

---

## 📊 Monitoreo y Logs

### CloudWatch Logs

Los logs del endpoint están disponibles en:
```
/aws/sagemaker/Endpoints/{ENDPOINT_NAME}
```

**Acceso:**
- Consola AWS → CloudWatch → Log groups
- O usa el Paso 13 del notebook

### Métricas

Métricas disponibles en:
- SageMaker → Endpoints → {ENDPOINT_NAME} → Monitoring
- CloudWatch → Metrics → AWS/SageMaker

**Métricas importantes:**
- `Invocations`: Número de invocaciones
- `ModelLatency`: Latencia del modelo
- `Invocation4XXErrors`: Errores 4xx
- `Invocation5XXErrors`: Errores 5xx

---

## 💻 Uso del Endpoint desde Código

### Python (boto3)

```python
import boto3
import json
import numpy as np

# Crear cliente
runtime = boto3.client(
    'sagemaker-runtime',
    region_name='us-east-1',
    aws_access_key_id='TU_ACCESS_KEY',
    aws_secret_access_key='TU_SECRET_KEY'
)

# Preparar datos
ecg_data = np.random.randn(1, 2000, 3).astype(np.float32)
data = {
    "signals": ecg_data.tolist()
}

# Invocar endpoint
response = runtime.invoke_endpoint(
    EndpointName='cnn1d-lstm-ecg-v1-serverless',
    ContentType='application/json',
    Body=json.dumps(data)
)

# Leer respuesta
result = json.loads(response['Body'].read())
print(f"Probabilidad de anomalía: {result['prediction']:.4f}")
```

### Python (SageMaker SDK)

```python
from sagemaker.predictor import Predictor
import json
import numpy as np

# Crear predictor
predictor = Predictor(
    endpoint_name='cnn1d-lstm-ecg-v1-serverless',
    serializer=json.dumps,
    deserializer=json.loads
)

# Preparar datos
ecg_data = np.random.randn(1, 2000, 3).astype(np.float32)
data = {"signals": ecg_data.tolist()}

# Invocar
result = predictor.predict(data)
print(result)
```

### cURL

```bash
curl -X POST \
  https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/cnn1d-lstm-ecg-v1-serverless/invocations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: AWS4-HMAC-SHA256 ...' \
  -d '{
    "signals": [[[0.1, 0.2, 0.3], ...]]
  }'
```

**Nota:** Necesitas firmar la petición con AWS Signature Version 4. Usa AWS CLI o boto3 para generar la firma.

---

## 🗑️ Eliminación del Endpoint

### Desde el Notebook

Ejecuta la última celda (descomenta las líneas):
```python
predictor.delete_endpoint()
predictor.delete_model()
```

### Desde la Consola AWS

1. SageMaker → Endpoints
2. Selecciona el endpoint
3. Actions → Delete
4. Confirma la eliminación

### Desde AWS CLI

```bash
aws sagemaker delete-endpoint --endpoint-name cnn1d-lstm-ecg-v1-serverless
aws sagemaker delete-endpoint-config --endpoint-config-name cnn1d-lstm-ecg-v1-serverless
aws sagemaker delete-model --model-name <model-name>
```

**⚠️ Importante:** Eliminar el endpoint evita costos cuando no lo estés usando.

---

## 📚 Recursos Adicionales

### Documentación Oficial

- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [SageMaker PyTorch Model](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)
- [boto3 SageMaker Runtime](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html)

### Enlaces Útiles

- Consola SageMaker: https://console.aws.amazon.com/sagemaker/
- CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/
- Calculadora de Costos AWS: https://calculator.aws/

---

## ✅ Checklist de Despliegue

Antes de comenzar, verifica:

- [ ] Cuenta AWS activa
- [ ] Credenciales AWS obtenidas (Access Key ID y Secret Access Key)
- [ ] Rol IAM creado con permisos necesarios
- [ ] Archivo del modelo en `sagemaker_models/cnn1d_lstm_ecg_v1_sagemaker.tar.gz`
- [ ] Python 3.8+ instalado
- [ ] Conexión a internet estable
- [ ] Límites de cuenta AWS verificados (número de endpoints)

Durante el despliegue:

- [ ] Paso 0 ejecutado sin errores
- [ ] Paso 1 configurado con tus credenciales
- [ ] Paso 3 verifica credenciales correctamente
- [ ] Paso 6 sube el modelo a S3
- [ ] Paso 9 completa el despliegue (5-10 minutos)
- [ ] Paso 10 prueba el endpoint exitosamente

Después del despliegue:

- [ ] Endpoint en estado "InService"
- [ ] Prueba exitosa con datos reales
- [ ] Logs disponibles en CloudWatch
- [ ] Documentas la URL del endpoint para uso futuro

---

## 📞 Soporte

Si encuentras problemas no cubiertos en esta documentación:

1. Revisa los logs en CloudWatch (Paso 13)
2. Verifica la documentación oficial de AWS SageMaker
3. Consulta los issues en el repositorio del proyecto
4. Revisa el código de inferencia (`inference.py`) para errores

---

## 📝 Notas Finales

- **Tiempo de despliegue:** 7-12 minutos típicamente
- **Tiempo de cold start:** 5-30 segundos (primera invocación después de inactividad)
- **Tiempo de inferencia:** 1-3 segundos por ECG
- **Regiones soportadas:** Verifica en la documentación de AWS qué regiones soportan Serverless Inference

---

**Última actualización:** Noviembre 2024  
**Versión del notebook:** 1.0  
**Versión de SageMaker SDK:** 2.x

