# ⚡ Inicio Rápido

Guía rápida para poner en marcha el sistema completo.

## 🎯 Checklist de Configuración

### ✅ Paso 1: Configurar AWS (30-45 minutos)

1. **Crear Rol IAM** → Ver `INSTRUCCIONES_AWS.md` - Paso 1
2. **Crear Lambda** → Ver `INSTRUCCIONES_AWS.md` - Paso 2
3. **Crear API Gateway** → Ver `INSTRUCCIONES_AWS.md` - Paso 4
4. **Guardar URL de API Gateway** → La necesitarás para el frontend

### ✅ Paso 2: Configurar Frontend (5 minutos)

1. **Instalar dependencias:**
   ```bash
   cd Frontend
   npm install
   ```

2. **Configurar URL de API:**
   ```bash
   cp .env.example .env
   # Edita .env y pega tu URL de API Gateway
   ```

3. **Ejecutar:**
   ```bash
   npm run dev
   ```

## 🚀 Uso Rápido

1. Abre `http://localhost:5173`
2. Selecciona un ECG (click en tarjeta)
3. Click en "🚀 Enviar a Modelo"
4. Ver resultado

## 📋 Archivos Importantes

- `INSTRUCCIONES_AWS.md` → Guía paso a paso para AWS
- `DOCUMENTACION_COMPLETA.md` → Documentación técnica completa
- `README.md` → Información del frontend
- `lambda_function.py` → Código para subir a Lambda

## ⚠️ Problemas Comunes

### "VITE_API_URL no está configurada"
→ Crea `.env` con tu URL de API Gateway

### Error de CORS
→ Verifica que CORS esté habilitado en API Gateway

### Lambda timeout
→ Aumenta timeout a 60 segundos

## 📞 ¿Dónde Buscar Más Información?

- **Configuración AWS:** `INSTRUCCIONES_AWS.md`
- **Documentación técnica:** `DOCUMENTACION_COMPLETA.md`
- **Frontend:** `README.md`

