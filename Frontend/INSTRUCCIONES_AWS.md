# 📋 Instrucciones Paso a Paso: Configurar Lambda + API Gateway

Esta guía te llevará paso a paso para configurar la arquitectura completa: Lambda → API Gateway → SageMaker.

---

## 🔐 PASO 1: Crear Rol IAM para Lambda

**Objetivo:** Crear un rol que permita a Lambda invocar tu endpoint de SageMaker.

### 1.1. Ir a IAM Console
1. Ve a [AWS Console](https://console.aws.amazon.com)
2. Busca "IAM" en el buscador superior
3. Click en **"Roles"** en el menú izquierdo
4. Click en **"Create role"**

### 1.2. Configurar Rol
1. **"Trusted entity type":** Selecciona **"AWS service"**
2. **"Use case":** Selecciona **"Lambda"**
3. Click en **"Next"**

### 1.3. Agregar Políticas
Agrega estas dos políticas:

**a) Política básica de ejecución:**
- Busca: `AWSLambdaBasicExecutionRole`
- Selecciónala (permite escribir logs en CloudWatch)

**b) Permiso para invocar SageMaker:**
1. Click en **"Create policy"** (se abrirá en otra pestaña)
2. En la nueva pestaña, click en **"JSON"**
3. Pega este JSON (reemplaza `YOUR_ENDPOINT_NAME` con tu endpoint real, ej: `cnn1d-lstm-ecg-v1-serverless`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:us-east-1:*:endpoint/YOUR_ENDPOINT_NAME"
        }
    ]
}
```

4. Click en **"Next"**
5. Nombre: `LambdaInvokeSageMakerPolicy`
6. Click en **"Create policy"**
7. Vuelve a la pestaña anterior y recarga (F5)
8. Busca `LambdaInvokeSageMakerPolicy` y selecciónala

### 1.4. Finalizar Rol
1. Click en **"Next"**
2. **Role name:** `LambdaSageMakerECGRole` (o el nombre que prefieras)
3. **Description:** "Rol para Lambda que invoca endpoint de SageMaker para ECG"
4. Click en **"Create role"**

✅ **Checklist:**
- [ ] Rol creado
- [ ] Tiene `AWSLambdaBasicExecutionRole`
- [ ] Tiene política personalizada para `sagemaker:InvokeEndpoint`

---

## 🚀 PASO 2: Crear Función Lambda

### 2.1. Ir a Lambda Console
1. En AWS Console, busca **"Lambda"**
2. Click en **"Create function"**

### 2.2. Configurar Función
1. **"Author from scratch"** (ya seleccionado)
2. **Function name:** `ecg-sagemaker-proxy` (o el nombre que prefieras)
3. **Runtime:** Selecciona **"Python 3.11"** (o la versión más reciente disponible)
4. **Architecture:** `x86_64`
5. **Permissions:** 
   - **"Use an existing role"**
   - Selecciona el rol que creaste: `LambdaSageMakerECGRole`
6. Click en **"Create function"**

### 2.3. Cargar Código
1. En la página de la función, baja hasta **"Code source"**
2. Abre el archivo `lambda_function.py` (está en `Frontend/lambda_function.py`)
3. Copia TODO el contenido
4. En Lambda Console, elimina el código de ejemplo
5. Pega tu código
6. Click en **"Deploy"** (botón naranja arriba a la derecha)

### 2.4. Configurar Variables de Entorno
1. En la página de la función, busca **"Configuration"** (menú izquierdo)
2. Click en **"Environment variables"**
3. Click en **"Edit"**
4. Agrega esta variable:

| Key | Value |
|-----|-------|
| `SAGEMAKER_ENDPOINT` | `cnn1d-lstm-ecg-v1-serverless` (o el nombre de tu endpoint) |

**Nota importante:** NO configures `AWS_REGION` como variable de entorno. Es una variable reservada que Lambda proporciona automáticamente. El código de Lambda la detectará automáticamente.

5. Click en **"Save"**

### 2.5. Ajustar Timeout
1. En **"Configuration"**, click en **"General configuration"**
2. Click en **"Edit"**
3. **Timeout:** Cambia a `30 seconds` (o más si tu modelo tarda más)
4. **Memory:** `512 MB` es suficiente (puedes subirlo si hay problemas)
5. Click en **"Save"**

✅ **Checklist:**
- [ ] Función Lambda creada
- [ ] Código cargado
- [ ] Variables de entorno configuradas
- [ ] Timeout ajustado a 30s

---

## 🧪 PASO 3: Probar Lambda (Opcional pero Recomendado)

### 3.1. Crear Test Event
1. En Lambda Console, en la parte superior, click en **"Test"**
2. Click en **"Create new test event"**
3. **Event name:** `test-ecg`
4. Reemplaza el JSON con este ejemplo:

**Opción 1: Test mínimo (solo válida la Lambda, no el modelo)**
```json
{
  "body": "{\"signals\": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]}"
}
```

**⚠️ Advertencia:** Este test solo valida que Lambda funcione, pero fallará al invocar SageMaker porque el modelo espera `[1, 2000, 3]`.

**Opción 2: Test real con forma correcta (recomendado)**

Genera un ECG de prueba válido con Python en tu máquina local:

```python
import json
import numpy as np

# Generar ECG de prueba con forma [1, 2000, 3]
ecg_test = np.random.randn(1, 2000, 3).astype(float).tolist()

test_event = {
    "body": json.dumps({
        "signals": ecg_test
    })
}

print(json.dumps(test_event, indent=2))
```

Copia la salida y úsala como test event en Lambda.

**Forma esperada:** `[1, 2000, 3]`
- `1`: batch size (un ECG a la vez)
- `2000`: número de muestras temporales
- `3`: número de canales

5. Click en **"Create"**

### 3.2. Ejecutar Test
1. Click en **"Test"** nuevamente
2. Espera unos segundos
3. Deberías ver:
   - **Status:** `Succeeded`
   - **Response:** JSON con `prediction` y `probability`

Si hay errores:
- Verifica que `SAGEMAKER_ENDPOINT` esté correcto
- Verifica que el endpoint de SageMaker esté en estado `InService`
- Revisa los logs en CloudWatch

✅ **Checklist:**
- [ ] Test ejecutado exitosamente
- [ ] Respuesta contiene `prediction` y `probability`

---

## 🌐 PASO 4: Crear API Gateway HTTP API

### 4.1. Ir a API Gateway Console
1. En AWS Console, busca **"API Gateway"**
2. Click en **"Create API"**
3. Selecciona **"HTTP API"** (NO REST API)
4. Click en **"Build"**

### 4.1.1. Nombre de la API (Opcional pero Recomendado)
Cuando se te pida un nombre para la API, puedes usar cualquiera de estos:

**Opciones sugeridas:**
- `ecg-model-api` (simple y descriptivo)
- `ecg-sagemaker-api` (indica que usa SageMaker)
- `ecg-anomaly-detection-api` (descriptivo del propósito)
- `ecg-demo-api` (si es solo para demo)

**Nota:** El nombre es solo para organización en la consola. No afecta la funcionalidad. Si no quieres poner un nombre, puedes dejarlo vacío.

### 4.2. Configurar API
1. **"Integrations"**:
   - Click en **"Add integration"**
   
2. **Configurar la integración Lambda:**
   - **Integration type:** Selecciona `Lambda`
   - **Lambda function:** Selecciona `ecg-sagemaker-proxy` de la lista desplegable
   - **Version or alias:** `$LATEST` (o déjalo como está si ya dice $LATEST)
   - **Destino de integración / Integration destination:** Esto debería llenarse automáticamente cuando seleccionas la función Lambda. Debería mostrar algo como:
     ```
     ecg-sagemaker-proxy (arn:aws:lambda:us-east-1:...)
     ```
     Si no se llena automáticamente, no te preocupes, sigue adelante.
   - Deja marcado **"Use default timeout"** (o el timeout por defecto)
   
3. Click en **"Next"**

### 4.3. Configurar Rutas
1. **"Configure routes"**:
   - **Method:** `POST`
   - **Resource path:** `/predict`
   - Click en **"Next"**

### 4.4. Configurar Stages
1. **"Configure stages"**:
   - **Stage name:** `prod` (o `dev` para pruebas)
   - Click en **"Next"**

### 4.5. Revisar y Crear
1. Revisa la configuración
2. Click en **"Create"**

### 4.6. Obtener URL de la API
1. Después de crear, verás una página con la URL de tu API
2. **IMPORTANTE - Dos casos posibles:**

   **Caso A: Si NO creaste un stage (o usas `$default`):**
   ```
   https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com
   ```
   Úsala completa en `.env` (sin `/predict` al final)
   
   **Caso B: Si creaste un stage (ej: `dev`, `prod`):**
   ```
   https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/dev
   ```
   Úsala completa en `.env` (incluye `/dev`, pero sin `/predict` al final)

3. **Guarda esta URL completa** - la necesitarás en el frontend

4. **¿Cómo saber cuál usar?**
   - Ve a API Gateway → Tu API → **"Stages"**
   - Si ves un stage llamado `dev`, `prod`, etc., usa la URL **CON** ese stage
   - Si solo ves `$default` o no hay stages, usa la URL **SIN** stage

5. El código del frontend agregará `/predict` automáticamente:
   - Si pusiste: `https://xxx.execute-api.us-east-1.amazonaws.com`
     → Endpoint final: `https://xxx.execute-api.us-east-1.amazonaws.com/predict`
   - Si pusiste: `https://xxx.execute-api.us-east-1.amazonaws.com/dev`
     → Endpoint final: `https://xxx.execute-api.us-east-1.amazonaws.com/dev/predict`

### 4.7. Habilitar CORS (Importante)

**⚠️ CORS es CRÍTICO - Sin esto, el frontend no funcionará**

1. En la página de tu API, en el menú izquierdo, click en **"CORS"**

2. **Configura CORS:**

   **a) Access-Control-Allow-Origin:**
   - Para desarrollo local, agrega: `http://localhost:5173`
   - O puedes usar `*` para permitir cualquier origen (solo para desarrollo/demo)
   - Si quieres ambos, puedes agregar múltiples orígenes (algunas versiones de API Gateway lo permiten)
   
   **b) Access-Control-Allow-Methods:**
   - Debe incluir: `POST` y `OPTIONS`
   - Opcionalmente: `GET` (aunque no lo uses)
   
   **c) Access-Control-Allow-Headers:**
   - Debe incluir: `Content-Type`
   - Opcionalmente: `Authorization` (si planeas usarlo después)

3. **IMPORTANTE - Después de configurar:**
   - Click en **"Save"**
   - **Espera 10-30 segundos** para que los cambios se propaguen
   - Prueba nuevamente desde el frontend

4. **Si aún no funciona, verifica:**
   - Que hayas guardado los cambios
   - Que estés accediendo desde `http://localhost:5173` (exactamente, sin `https`)
   - Que el método sea `POST` (el preflight OPTIONS debe estar permitido automáticamente)

✅ **Checklist:**
- [ ] API Gateway HTTP API creada
- [ ] Ruta `/predict` configurada con método POST
- [ ] Integración con Lambda configurada
- [ ] CORS habilitado
- [ ] URL de la API guardada

---

## 🔍 PASO 5: Verificar que Todo Funciona

### 5.1. Probar con cURL (Desde Terminal)

Reemplaza `YOUR_API_URL` con la URL de tu API Gateway:

```bash
curl -X POST https://YOUR_API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"signals": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...]]}'
```

**Nota:** En un test real, necesitas enviar `signals` con forma `[1, 2000, 3]`.

### 5.2. Verificar Respuesta
Deberías recibir algo como:

```json
{
  "prediction": 0.1234,
  "probability": 0.1234
}
```

### 5.3. Ver Logs si Hay Problemas
1. Ve a **CloudWatch** en AWS Console
2. **Log groups** → `/aws/lambda/ecg-sagemaker-proxy`
3. Revisa los logs más recientes

---

## 🎯 Resumen de URLs y Nombres

Guarda esta información:

```
Rol IAM: LambdaSageMakerECGRole
Función Lambda: ecg-sagemaker-proxy
Endpoint SageMaker: cnn1d-lstm-ecg-v1-serverless
API Gateway URL: https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com
Endpoint completo: https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/predict
```

---

## ⚠️ Troubleshooting

### Error: "SAGEMAKER_ENDPOINT no está configurado"
- **Solución:** Verifica que la variable de entorno esté configurada en Lambda

### Error: "AccessDenied" al invocar SageMaker
- **Solución:** Verifica que el rol de Lambda tenga permisos para `sagemaker:InvokeEndpoint`

### Error: CORS desde el frontend
- **Solución:** Asegúrate de haber configurado CORS en API Gateway con tu dominio localhost

### Error: Timeout
- **Solución:** Aumenta el timeout de Lambda a 60 segundos si tu modelo tarda más

---

## 🎉 ¡Listo!

Ahora tienes:
- ✅ Lambda configurada
- ✅ API Gateway exponiendo tu Lambda
- ✅ CORS habilitado
- ✅ Sin credenciales expuestas en el frontend

Continúa con la configuración del frontend React.

