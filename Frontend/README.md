# 🫀 Frontend - Demo de Detección de Anomalías en ECG

Frontend React + Vite para interactuar con el modelo de ECG desplegado en AWS SageMaker.

## 📋 Requisitos Previos

- Node.js 18+ instalado
- npm o yarn
- API Gateway configurado (ver `INSTRUCCIONES_AWS.md`)
- URL de tu API Gateway

## 🚀 Instalación y Configuración

### 1. Instalar dependencias

```bash
npm install
```

### 2. Configurar variable de entorno

Crea un archivo `.env` en la raíz del proyecto (junto a `package.json`):

```bash
cp .env.example .env
```

Edita `.env` y agrega tu URL de API Gateway:

```env
VITE_API_URL=https://tu-api-gateway-url.execute-api.us-east-1.amazonaws.com
```

**Importante:** NO incluyas `/predict` al final, se agrega automáticamente.

### 3. Generar archivo de ECG de ejemplo (opcional)

Si tienes los datos de entrenamiento, puedes regenerar `src/data/ecg_samples.json`:

```bash
python generate_ecg_samples.py
```

Este script buscará los datos en `../data/Datos_supervisados/tensors_200hz/`.

### 4. Ejecutar en desarrollo

```bash
npm run dev
```

El frontend estará disponible en `http://localhost:5173`

## 📁 Estructura del Proyecto

```
Frontend/
├── src/
│   ├── api/
│   │   └── client.js          # Cliente para comunicarse con API Gateway
│   ├── data/
│   │   └── ecg_samples.json   # ECG de ejemplo para pruebas
│   ├── App.jsx                # Componente principal
│   ├── App.css                # Estilos del componente principal
│   ├── main.jsx               # Punto de entrada
│   └── index.css              # Estilos globales
├── lambda_function.py         # Función Lambda (para subir a AWS)
├── generate_ecg_samples.py    # Script para generar ECG de ejemplo
├── INSTRUCCIONES_AWS.md       # Guía paso a paso para AWS
├── package.json
├── vite.config.js
└── .env.example
```

## 🎯 Uso

1. **Seleccionar ECG:** Click en una de las tarjetas de ECG para seleccionarla
2. **Enviar al modelo:** Click en "Enviar a Modelo"
3. **Ver resultado:** 
   - Resumen amigable con predicción y confianza
   - JSON raw de la respuesta
   - Comparación con etiqueta real (si está disponible)

## 🔐 Seguridad

- ✅ **NO** se exponen credenciales AWS en el frontend
- ✅ Todas las peticiones van a través de API Gateway
- ✅ CORS configurado para desarrollo (localhost)
- ✅ La Lambda maneja las credenciales usando IAM roles

## 🛠️ Scripts Disponibles

- `npm run dev` - Ejecuta el servidor de desarrollo
- `npm run build` - Construye para producción
- `npm run preview` - Preview de la build de producción

## 🐛 Troubleshooting

### Error: "VITE_API_URL no está configurada"

**Solución:** Crea el archivo `.env` con la URL de tu API Gateway.

### Error de CORS en el navegador

**Solución:** 
1. Verifica que CORS esté habilitado en API Gateway
2. Agrega `http://localhost:5173` a los orígenes permitidos en API Gateway

### Error: "Error de conexión"

**Solución:**
1. Verifica que la URL en `.env` sea correcta
2. Verifica que el API Gateway esté desplegado
3. Verifica que la Lambda esté funcionando (revisa logs en CloudWatch)

### El modelo no responde

**Solución:**
1. Revisa los logs de Lambda en CloudWatch
2. Verifica que el endpoint de SageMaker esté en estado `InService`
3. Verifica que la variable de entorno `SAGEMAKER_ENDPOINT` esté configurada en Lambda

## 📚 Más Información

- Ver `INSTRUCCIONES_AWS.md` para configurar Lambda y API Gateway
- Ver `DOCUMENTACION_COMPLETA.md` para la documentación técnica completa

