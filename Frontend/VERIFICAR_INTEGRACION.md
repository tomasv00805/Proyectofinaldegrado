# ✅ Verificar Configuración de Integración en API Gateway

## 🔍 Lo que Debes Verificar

### 1. Click en "Configurar" en la sección "Integración"

Cuando haces click en **"Configurar"**, deberías ver:

#### ✅ Configuración Correcta:

- **Integration type:** `Lambda function`
- **Lambda function:** `ecg-sagemaker-proxy` (debe aparecer el nombre completo)
- **Use default timeout:** ✅ Marcado (o timeout configurado)
- **Payload version:** `2.0` (por defecto, está bien)

#### ❌ Si ves algo diferente:

- Si no ves el nombre de tu Lambda, la integración no está bien configurada
- Si dice "AWS Service" o algo diferente a "Lambda function", está mal

---

## 🔧 Cómo Corregir si Está Mal

### Opción 1: Editar la Integración Existente

1. Click en **"Configurar"** en la sección Integración
2. Verifica que:
   - **Integration type:** sea `Lambda function`
   - **Lambda function:** muestre `ecg-sagemaker-proxy`
3. Si no está bien, cambia:
   - Selecciona `Lambda function` en Integration type
   - Selecciona `ecg-sagemaker-proxy` en Lambda function
4. Click en **"Save"**

### Opción 2: Eliminar y Recrear la Integración

Si la integración no se puede editar:

1. Ve a **"Routes"** → Click en `POST /predict`
2. Busca la sección **"Integration"**
3. **Elimina la integración actual**
4. **Agrega nueva integración:**
   - Click en **"Add integration"** o **"Configure"**
   - **Integration type:** `Lambda function`
   - **Lambda function:** Selecciona `ecg-sagemaker-proxy`
   - Click en **"Save"**

---

## ✅ Qué Debe Aparecer en los Detalles de la Ruta

### Integración Correcta:

```
Integración
qp0qyyo  [Este es solo un ID, está bien]

Al hacer click en "Configurar", deberías ver:
- Integration type: Lambda function
- Lambda function: ecg-sagemaker-proxy
- Use default timeout: ✓
```

### Autorización:

```
No hay ningún autorizador asociado a esta ruta.
```

**Esto está BIEN para una demo.** No necesitas autorización para probar.

---

## 🚨 Problemas Comunes

### Problema 1: La integración no muestra el nombre de Lambda

**Solución:** 
- Click en "Configurar"
- Verifica que "Integration type" sea "Lambda function"
- Selecciona `ecg-sagemaker-proxy` en "Lambda function"
- Guarda

### Problema 2: Dice "No integration" o similar

**Solución:**
- Agrega una integración nueva
- Selecciona "Lambda function"
- Selecciona tu Lambda

### Problema 3: La Lambda no aparece en la lista

**Posibles causas:**
1. La Lambda está en otra región
   - Verifica que la Lambda esté en la misma región que API Gateway
2. Permisos
   - API Gateway necesita permiso para invocar Lambda
   - Esto normalmente se hace automáticamente, pero verifica

---

## ✅ Verificación Final

Después de configurar, deberías poder:

1. Ver `ecg-sagemaker-proxy` como la función Lambda integrada
2. Probar la ruta haciendo una petición POST a `/predict`

¿La integración muestra el nombre de tu Lambda cuando haces click en "Configurar"?

