# 🔍 Troubleshooting - Error de Conexión

Guía paso a paso para diagnosticar y solucionar el error de conexión con API Gateway.

---

## ❌ Error: "Error de conexión. Verifica que: 1. La URL de la API esté correcta en .env..."

Este error indica que el frontend no puede conectarse con tu API Gateway.

---

## 🔍 Diagnóstico Paso a Paso

### Paso 1: Verificar archivo `.env`

1. **Ubicación:** El archivo `.env` debe estar en `Frontend/.env` (misma carpeta que `package.json`)

2. **Verificar que existe:**
   ```bash
   cd Frontend
   ls .env  # En Windows: dir .env
   ```

3. **Contenido correcto:**
   ```env
   VITE_API_URL=https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com
   ```
   
   **⚠️ IMPORTANTE:**
   - NO debe terminar con `/`
   - NO debe incluir `/predict` al final
   - Debe empezar con `https://`
   - Ejemplo correcto: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`
   - Ejemplo incorrecto: `https://abc123xyz.execute-api.us-east-1.amazonaws.com/predict`

4. **Si el archivo no existe:**
   ```bash
   cd Frontend
   copy .env.example .env  # Windows
   # o
   cp .env.example .env    # Linux/Mac
   ```
   Luego edítalo y agrega tu URL real.

---

### Paso 2: Verificar que la URL sea correcta

1. **Obtener la URL de tu API Gateway:**
   - Ve a AWS Console → API Gateway
   - Selecciona tu API (`ecg-model-api` o el nombre que usaste)
   - En el panel izquierdo, click en **"Stages"**
   - Click en tu stage (ej: `$default` o `prod`)
   - Verás la **"Invoke URL"** (algo como `https://abc123xyz.execute-api.us-east-1.amazonaws.com`)
   - **Copia SOLO la parte base** (sin rutas adicionales)

2. **Verificar formato:**
   - ✅ Correcto: `https://abc123xyz.execute-api.us-east-1.amazonaws.com`
   - ❌ Incorrecto: `https://abc123xyz.execute-api.us-east-1.amazonaws.com/predict`
   - ❌ Incorrecto: `https://abc123xyz.execute-api.us-east-1.amazonaws.com/`
   - ❌ Incorrecto: `abc123xyz.execute-api.us-east-1.amazonaws.com` (falta https://)

---

### Paso 3: Reiniciar el servidor de desarrollo

**Después de crear o modificar `.env`, SIEMPRE debes reiniciar Vite:**

1. **Detén el servidor:**
   - Presiona `Ctrl+C` en la terminal donde está corriendo `npm run dev`

2. **Reinicia:**
   ```bash
   npm run dev
   ```

3. **Verifica en la consola del navegador:**
   - Abre DevTools (F12)
   - Ve a la pestaña "Console"
   - No deberías ver el mensaje "VITE_API_URL no está configurada"

---

### Paso 4: Verificar que API Gateway esté desplegada

1. **En AWS Console:**
   - Ve a API Gateway
   - Selecciona tu API
   - Click en **"Stages"** en el menú izquierdo
   - Deberías ver al menos un stage (ej: `$default` o `prod`)
   - Si no hay stages, la API no está desplegada → Ve al Paso 5

2. **Verificar el endpoint:**
   - Click en el stage
   - Deberías ver rutas como `/predict`
   - Si no ves rutas, necesitas configurarlas

---

### Paso 5: Verificar que la ruta `/predict` exista

1. **En API Gateway:**
   - Selecciona tu API
   - Click en **"Routes"** en el menú izquierdo
   - Deberías ver: `POST /predict`
   - Si no está, necesitas crearla (ver `INSTRUCCIONES_AWS.md` - Paso 4.3)

---

### Paso 6: Verificar CORS

1. **En API Gateway:**
   - Selecciona tu API
   - Click en **"CORS"** en el menú izquierdo
   - Verifica que esté configurado:
     - **Access-Control-Allow-Origin:** `http://localhost:5173` (o `*` para desarrollo)
     - **Access-Control-Allow-Methods:** `POST, OPTIONS`
     - **Access-Control-Allow-Headers:** `Content-Type`

2. **Si no está configurado:**
   - Click en **"Configure"**
   - Agrega los valores arriba
   - Click en **"Save"**

---

### Paso 7: Probar la API directamente

Abre una nueva pestaña en tu navegador y ejecuta esto en la consola (F12):

```javascript
// Reemplaza con tu URL real
const apiUrl = 'https://TU-API-GATEWAY-URL.execute-api.us-east-1.amazonaws.com/predict';

fetch(apiUrl, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ signals: [[[0.1, 0.2, 0.3]]] })
})
.then(res => res.json())
.then(data => console.log('✅ Respuesta:', data))
.catch(err => console.error('❌ Error:', err));
```

**Resultados esperados:**
- ✅ Si funciona: Verás la respuesta del modelo o un error de validación (esto es bueno, significa que la API está respondiendo)
- ❌ Si no funciona: Verás un error de CORS o de conexión

---

## 🔧 Soluciones Comunes

### Problema 1: "VITE_API_URL no está configurada"

**Solución:**
1. Crea el archivo `.env` en `Frontend/`
2. Agrega: `VITE_API_URL=https://tu-url-aqui.execute-api.us-east-1.amazonaws.com`
3. Reinicia `npm run dev`

### Problema 2: CORS Error en la consola del navegador

**Síntoma:** Error en consola tipo "CORS policy" o "Access-Control-Allow-Origin"

**Solución:**
1. Ve a API Gateway → Tu API → CORS
2. Configura:
   - **Access-Control-Allow-Origin:** `http://localhost:5173`
   - **Access-Control-Allow-Methods:** `POST, OPTIONS`
   - Guarda y espera unos segundos

### Problema 3: 404 Not Found

**Síntoma:** Error 404 al hacer la petición

**Solución:**
- Verifica que la ruta `/predict` esté configurada en API Gateway
- Verifica que la URL en `.env` NO termine con `/predict` (se agrega automáticamente)

### Problema 4: Network Error / Failed to fetch

**Síntoma:** Error de red, no se puede conectar

**Solución:**
- Verifica que la URL en `.env` sea correcta
- Verifica que la API Gateway esté desplegada (tiene un stage)
- Verifica tu conexión a internet
- Verifica que no haya firewall bloqueando

### Problema 5: Cambié .env pero no funciona

**Solución:**
- **SIEMPRE** reinicia `npm run dev` después de cambiar `.env`
- Vite solo lee `.env` al iniciar

---

## 📋 Checklist de Verificación

Usa este checklist para verificar todo:

- [ ] Archivo `.env` existe en `Frontend/`
- [ ] `VITE_API_URL` está configurada en `.env`
- [ ] URL NO termina con `/` ni `/predict`
- [ ] URL empieza con `https://`
- [ ] Servidor de desarrollo fue reiniciado después de crear/modificar `.env`
- [ ] API Gateway tiene al menos un stage desplegado
- [ ] La ruta `POST /predict` existe en API Gateway
- [ ] CORS está configurado en API Gateway
- [ ] Lambda function está configurada correctamente

---

## 🆘 Si Nada Funciona

1. **Verifica en CloudWatch Logs:**
   - Ve a CloudWatch → Log groups
   - Busca `/aws/lambda/ecg-sagemaker-proxy`
   - Revisa los logs más recientes

2. **Verifica el test de Lambda:**
   - Ve a Lambda → `ecg-sagemaker-proxy` → Test
   - Ejecuta un test para verificar que Lambda funciona

3. **Prueba con cURL (desde terminal):**
   ```bash
   curl -X POST https://TU-API-URL.execute-api.us-east-1.amazonaws.com/predict \
     -H "Content-Type: application/json" \
     -d '{"signals": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]}'
   ```

---

## 📞 Información Útil para Debugging

Si necesitas ayuda adicional, proporciona:

1. URL completa que está en `.env` (puedes ocultar parte con `xxx`)
2. Mensaje de error completo de la consola del navegador
3. Si ves algún error en CloudWatch Logs de Lambda
4. Resultado del test de Lambda

