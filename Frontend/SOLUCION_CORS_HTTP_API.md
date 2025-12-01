# 🔧 Solución Error CORS en API Gateway HTTP API

## ❌ Error que estás viendo:

```
Access to fetch at 'https://...' from origin 'http://localhost:5173' 
has been blocked by CORS policy: Response to preflight request doesn't pass 
access control check: No 'Access-Control-Allow-Origin' header is present
```

## 📖 ¿Qué significa este error?

1. **Tu frontend está intentando conectarse** ✅ (esto funciona)
2. **El navegador envía un "preflight request" OPTIONS** antes del POST real
3. **API Gateway NO está respondiendo con headers CORS** ❌
4. **El navegador bloquea la petición** por seguridad

---

## ✅ Solución: Configurar CORS en API Gateway HTTP API

En **API Gateway HTTP API**, CORS se configura de forma diferente a REST API.

### Paso 1: Ir a la Configuración de CORS

1. Ve a **AWS Console** → **API Gateway**
2. Selecciona tu API HTTP (`ecg-model-api` o el nombre que usaste)
3. En el menú izquierdo, busca **"CORS"** o **"Develop" → "CORS"**

**Si no ves "CORS" en el menú:**
- Ve a **"Develop"** → **"CORS"**
- O ve a **"Routes"** → Selecciona `POST /predict` → Busca sección de CORS
- O busca **"Authorization"** o **"Integration"** y luego CORS

### Paso 2: Configurar CORS (Método 1: Desde CORS)

Si ves la opción "CORS":

1. Click en **"CORS"**
2. Click en **"Configure"** o **"Edit"**
3. Configura estos valores:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: Content-Type
Access-Control-Allow-Methods: POST, OPTIONS
```

4. Click en **"Save"**

### Paso 3: Configurar CORS (Método 2: Si no ves opción CORS directa)

En API Gateway HTTP API, CORS a veces se configura desde la ruta:

1. Ve a **"Routes"** → Click en `POST /predict`
2. Busca una sección de **"CORS"** o **"Authorization"**
3. O ve a **"Integrations"** → Click en tu integración → Busca CORS
4. Configura los mismos valores que arriba

### Paso 4: Verificar que OPTIONS esté Habilitado

**IMPORTANTE:** API Gateway HTTP API puede necesitar una ruta OPTIONS explícita:

1. Ve a **"Routes"**
2. Verifica si existe `OPTIONS /predict`
3. Si NO existe:
   - Click en **"Create"** o **"Add route"**
   - **Method:** `OPTIONS`
   - **Resource path:** `/predict`
   - **Integration:** La misma Lambda (`ecg-sagemaker-proxy`)
   - O déjala sin integración (API Gateway manejará el OPTIONS automáticamente si CORS está configurado)

---

## 🎯 Solución Alternativa: Usar la Integración de CORS Automática

En algunas versiones de API Gateway HTTP API, puedes habilitar CORS automáticamente:

1. Ve a tu ruta `POST /predict`
2. Click en **"Configure"** en la sección de Integración
3. Busca una opción tipo:
   - ✅ **"Enable CORS"**
   - ✅ **"Use CORS"**
   - ✅ **"CORS enabled"**
4. Márcala como habilitada
5. Guarda

---

## 🔍 Verificación Rápida

Después de configurar:

1. **Espera 30-60 segundos** (los cambios pueden tardar)
2. Abre la consola del navegador (F12)
3. Ejecuta este test:

```javascript
// Probar preflight OPTIONS
fetch('https://n1mek8nsrc.execute-api.us-east-1.amazonaws.com/dev/predict', {
  method: 'OPTIONS',
  headers: {
    'Origin': 'http://localhost:5173',
    'Access-Control-Request-Method': 'POST',
    'Access-Control-Request-Headers': 'Content-Type'
  }
})
.then(res => {
  console.log('✅ OPTIONS Status:', res.status);
  console.log('✅ CORS Headers:', {
    origin: res.headers.get('access-control-allow-origin'),
    methods: res.headers.get('access-control-allow-methods'),
    headers: res.headers.get('access-control-allow-headers')
  });
})
.catch(err => console.error('❌ Error:', err));
```

**Resultado esperado:**
- Status: `200` o `204`
- Headers con `access-control-allow-origin: *`

---

## 🚨 Si Nada Funciona: Solución Manual en Lambda

Si API Gateway HTTP API no maneja CORS automáticamente, la Lambda ya está preparada:

**Tu Lambda YA maneja OPTIONS requests** (líneas 49-54 de `lambda_function.py`):

```python
# Manejar preflight OPTIONS request
if event.get('httpMethod') == 'OPTIONS' or event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
    return {
        'statusCode': 200,
        'headers': cors_headers,
        'body': json.dumps({'message': 'OK'})
    }
```

**Pero puede que necesites crear la ruta OPTIONS manualmente:**

1. Ve a **"Routes"** → **"Create"** o **"Add route"**
2. **Method:** `OPTIONS`
3. **Resource path:** `/predict`
4. **Integration:** Lambda function → `ecg-sagemaker-proxy`
5. Guarda

---

## ✅ Checklist Final

- [ ] CORS configurado en API Gateway (método 1 o 2)
- [ ] Ruta `OPTIONS /predict` existe (o CORS automático habilitado)
- [ ] Esperaste 30-60 segundos después de guardar
- [ ] Probaste limpiar cache del navegador
- [ ] Lambda tiene el código actualizado con manejo de OPTIONS

---

## 📝 Nota Importante

En **API Gateway HTTP API**, a diferencia de REST API:
- CORS se puede configurar a nivel de API
- Pero a veces necesitas crear rutas OPTIONS explícitas
- O habilitar "CORS" en la configuración de la integración

¿Puedes ver la opción "CORS" en tu API Gateway? ¿Qué opciones ves en el menú izquierdo de tu API?

