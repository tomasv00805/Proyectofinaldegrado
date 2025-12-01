# 📚 Documentación Completa del Sistema

## 🎯 Objetivo del Proyecto

Este sistema demuestra la integración de un modelo de detección de anomalías en ECG (electrocardiogramas) desplegado en AWS SageMaker con un frontend web, siguiendo las mejores prácticas de seguridad y arquitectura en la nube.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │
       │ HTTP POST /predict
       │ JSON: {"signals": [[[...]]]}
       ▼
┌─────────────────────────────────────┐
│        API Gateway (HTTP API)        │
│  • Maneja CORS                       │
│  • Enrutamiento                      │
│  • Sin autenticación (demo)          │
└──────────────┬──────────────────────┘
               │
               │ Invoca función Lambda
               ▼
┌─────────────────────────────────────┐
│         AWS Lambda (Python)          │
│  • Recibe request                    │
│  • Valida formato                    │
│  • Invoca SageMaker usando IAM       │
│  • Retorna respuesta                 │
└──────────────┬──────────────────────┘
               │
               │ boto3.invoke_endpoint()
               │ (Usando rol IAM)
               ▼
┌─────────────────────────────────────┐
│   SageMaker Endpoint (Serverless)    │
│  • Modelo CNN1D+LSTM                 │
│  • Procesa ECG                       │
│  • Retorna probabilidad de anomalía  │
└─────────────────────────────────────┘
```

### Componentes Principales

1. **Frontend (React + Vite)**
   - Interfaz de usuario para seleccionar ECG y ver resultados
   - Se ejecuta en `localhost:5173`
   - NO contiene credenciales AWS

2. **API Gateway (HTTP API)**
   - Expone un endpoint público `/predict`
   - Maneja CORS para permitir requests desde el frontend
   - Enruta requests a Lambda

3. **Lambda Function (Python)**
   - Función serverless que actúa como proxy
   - Invoca el endpoint de SageMaker usando credenciales IAM
   - Maneja errores y formatea respuestas

4. **SageMaker Endpoint (Serverless Inference)**
   - Modelo de IA desplegado para inferencia
   - Procesa ECGs y retorna predicciones
   - Solo cobra por invocación (sin costo cuando está inactivo)

---

## 🔐 Seguridad: Por Qué NO Exponer Credenciales en el Frontend


**arquitectura:**
- Frontend → API Gateway → Lambda → SageMaker
- Credenciales AWS solo en Lambda (usando rol IAM)
- Frontend solo tiene la URL pública de API Gateway

**Ventajas:**
1. **Seguridad:** Credenciales nunca salen del backend (Lambda)
2. **Control:** Puedes agregar autenticación, rate limiting, logging
3. **Escalabilidad:** Lambda escala automáticamente
4. **Costos:** Solo pagas por invocaciones (serverless)

---

## 📥 Formato de Entrada y Salida

### Request del Frontend a API Gateway

**URL:** `POST https://tu-api-gateway-url.execute-api.us-east-1.amazonaws.com/predict`

**Headers:**
```http
Content-Type: application/json
```

**Body:**
```json
{
  "signals": [
    [
      [0.1, 0.2, 0.3],  // Muestra 1: [canal1, canal2, canal3]
      [0.4, 0.5, 0.6],  // Muestra 2: [canal1, canal2, canal3]
      ...
      [0.7, 0.8, 0.9]   // Muestra 2000: [canal1, canal2, canal3]
    ]
  ]
}
```

**Forma esperada:** `[1, 2000, 3]`
- `1`: batch size (un ECG a la vez)
- `2000`: número de muestras temporales (10 segundos a 200 Hz)
- `3`: número de canales (típicamente I, II, III o derivaciones similares)

### Respuesta del Modelo

**Status:** `200 OK`

**Body:**
```json
{
  "prediction": 0.9999,
  "probability": 0.9999
}
```

**Interpretación:**
- `prediction` y `probability`: Probabilidad de que el ECG sea anómalo (0-1)
- **> 0.5**: ECG anómalo
- **≤ 0.5**: ECG normal

### Ejemplo de Uso Completo

```javascript
// 1. Frontend prepara datos
const ecgData = {
  signals: [selectedECG.signals]  // Forma: [1, 2000, 3]
}

// 2. Frontend envía a API Gateway
const response = await fetch(API_URL + '/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(ecgData)
})

// 3. Recibe respuesta
const result = await response.json()
// result = { prediction: 0.9999, probability: 0.9999 }

// 4. Interpreta resultado
const isAnomaly = result.prediction > 0.5
const confidence = (isAnomaly ? result.prediction : (1 - result.prediction)) * 100
```

---

## 🔄 Flujo Completo de una Predicción

### Paso 1: Usuario selecciona ECG
- Usuario hace click en una tarjeta de ECG en el frontend
- El frontend carga los datos del ECG seleccionado

### Paso 2: Usuario envía al modelo
- Usuario hace click en "Enviar a Modelo"
- Frontend prepara el request:
  ```javascript
  {
    signals: [ecgSample.signals]  // Envolver en array para batch=1
  }
  ```

### Paso 3: Request a API Gateway
- Frontend hace `POST` a `https://api-gateway-url/predict`
- API Gateway recibe el request y lo enruta a Lambda

### Paso 4: Lambda procesa
- Lambda recibe `event["body"]` (JSON string)
- Parsea y valida el JSON
- Prepara payload para SageMaker:
  ```python
  payload = {
      "signals": request_data["signals"]
  }
  ```

### Paso 5: Lambda invoca SageMaker
- Lambda usa `boto3.client('sagemaker-runtime')`
- Credenciales se obtienen automáticamente del rol IAM
- Invoca endpoint:
  ```python
  response = sagemaker_runtime.invoke_endpoint(
      EndpointName=endpoint_name,  # Desde variable de entorno
      ContentType='application/json',
      Body=json.dumps(payload).encode('utf-8')
  )
  ```

### Paso 6: SageMaker procesa
- Modelo carga si no está cargado (cold start)
- Procesa el ECG a través de CNN1D + LSTM
- Retorna probabilidad de anomalía

### Paso 7: Lambda formatea respuesta
- Lambda lee la respuesta de SageMaker
- Agrega headers CORS
- Retorna al API Gateway:
  ```python
  return {
      'statusCode': 200,
      'headers': cors_headers,
      'body': json.dumps(model_response)
  }
  ```

### Paso 8: Frontend recibe y muestra
- Frontend recibe JSON con `prediction` y `probability`
- Calcula si es normal o anómalo (threshold 0.5)
- Muestra:
  - Resumen amigable (predicción, confianza)
  - JSON raw
  - Comparación con etiqueta real (si está disponible)

---

## 📊 Modelo de IA

### Tipo de Modelo
- **Arquitectura:** CNN1D + LSTM (Bidireccional)
- **Propósito:** Detección de anomalías en ECG
- **Salida:** Clasificación binaria (Normal/Anómalo)

### Características Técnicas
- **Input shape:** `[batch, 2000, 3]`
  - 2000 muestras temporales
  - 3 canales (derivaciones)
- **Frecuencia de muestreo:** 200 Hz
- **Duración:** 10 segundos (2000 muestras / 200 Hz)
- **Output:** Probabilidad (0-1) de anomalía

### Procesamiento
1. **CNN1D:** Extrae características locales de las señales
2. **LSTM:** Captura dependencias temporales a largo plazo
3. **Fully Connected:** Clasifica en normal/anómalo
4. **Sigmoid:** Normaliza a probabilidad [0, 1]

---

## 🚀 Cómo Usar en una Demo

### Preparación (Una vez)

1. **Configurar AWS:**
   - Crear rol IAM para Lambda
   - Crear función Lambda
   - Configurar API Gateway
   - Habilitar CORS
   - Guardar URL de API Gateway

2. **Configurar Frontend:**
   - Instalar dependencias: `npm install`
   - Crear `.env` con `VITE_API_URL`
   - (Opcional) Regenerar ECG samples

### Demo (Cada vez)

**Paso 1: Levantar Frontend**
```bash
cd Frontend
npm run dev
```
- Frontend se abre en `http://localhost:5173`

**Paso 2: Seleccionar ECG**
- Click en una de las tarjetas de ECG
- Ver información del ECG seleccionado
- ECGs están etiquetados como "NORMAL" o "ANÓMALO" (para comparar)

**Paso 3: Enviar al Modelo**
- Click en botón "🚀 Enviar a Modelo"
- Esperar respuesta (puede tardar 5-15 segundos en primera invocación por cold start)

**Paso 4: Ver Resultado**
- **Resumen:** Predicción (Normal/Anómalo), probabilidad, confianza
- **JSON Raw:** Respuesta completa del modelo
- **Comparación:** Si el ECG tenía etiqueta, compara predicción vs real

### Puntos Clave para la Demo

1. **Mostrar Arquitectura:**
   - "El frontend no tiene credenciales AWS"
   - "Todo pasa por API Gateway"
   - "Lambda usa IAM roles para seguridad"

2. **Mostrar Resultados:**
   - Predicción correcta/incorrecta vs etiqueta real
   - Probabilidad de confianza
   - Tiempo de respuesta

3. **Explicar Seguridad:**
   - Por qué no exponer credenciales
   - Cómo funciona IAM
   - CORS y permisos

---

## 💰 Costos Estimados

### SageMaker Serverless Inference
- **Por invocación:** ~$0.00022
- **Sin tráfico:** $0 (no hay costo cuando está inactivo)
- **Ejemplos mensuales:**
  - 1,000 invocaciones: $0.22
  - 10,000 invocaciones: $2.20
  - 100,000 invocaciones: $22.00

### Lambda
- **Primeros 1M requests/mes:** Gratis
- **Después:** $0.20 por 1M requests
- **Ejemplo:** 10,000 requests = $0.00 (dentro del tier gratuito)

### API Gateway (HTTP API)
- **Primeros 1M requests/mes:** Gratis
- **Después:** $1.00 por 1M requests
- **Ejemplo:** 10,000 requests = $0.00 (dentro del tier gratuito)

**Total estimado para demo:** Prácticamente $0 (dentro de tier gratuito)

---

## 🔍 Troubleshooting Avanzado

### Problema: Lambda timeout
**Síntoma:** Lambda retorna error 500 después de ~30 segundos

**Solución:**
1. Aumentar timeout de Lambda a 60 segundos
2. Verificar que el endpoint de SageMaker responda rápido
3. Revisar logs de CloudWatch para identificar cuellos de botella

### Problema: Cold start lento
**Síntoma:** Primera invocación tarda mucho (30-60 segundos)

**Explicación:** Normal en serverless. El modelo se carga en memoria en la primera invocación.

**Solución (si necesario):**
- Usar provisioned concurrency (tiene costo)
- O aceptar el cold start (solo afecta primera invocación)

### Problema: CORS desde navegador
**Síntoma:** Error en consola del navegador sobre CORS

**Solución:**
1. Verificar que CORS esté habilitado en API Gateway
2. Agregar `http://localhost:5173` a orígenes permitidos
3. Verificar headers en respuesta de Lambda

### Problema: Error 403 desde Lambda
**Síntoma:** Lambda retorna error 403

**Solución:**
1. Verificar que el rol de Lambda tenga permisos `sagemaker:InvokeEndpoint`
2. Verificar que el ARN del endpoint en la política IAM sea correcto
3. Verificar que el endpoint de SageMaker exista y esté en estado `InService`

---

## 📝 Notas Finales

### Para Producción

Si esto fuera para producción, considerarías:

1. **Autenticación:**
   - API Keys en API Gateway
   - Cognito para usuarios
   - JWT tokens

2. **Rate Limiting:**
   - Throttling en API Gateway
   - Limitar requests por usuario/IP

3. **Logging y Monitoreo:**
   - CloudWatch Logs más detallados
   - Métricas de uso
   - Alertas de errores

4. **Optimización:**
   - Caching de respuestas similares
   - Batch processing
   - Optimización del modelo

5. **Seguridad:**
   - HTTPS obligatorio
   - Validación más estricta de inputs
   - Rate limiting por IP

### Para el Proyecto Final

Este sistema demuestra:

✅ **Arquitectura serverless moderna**
✅ **Seguridad (sin exponer credenciales)**
✅ **Integración de ML con web**
✅ **Buenas prácticas de AWS**
✅ **Frontend moderno (React + Vite)**

---

## 📚 Referencias

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [API Gateway HTTP API](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api.html)
- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

---

**Última actualización:** Noviembre 2024

