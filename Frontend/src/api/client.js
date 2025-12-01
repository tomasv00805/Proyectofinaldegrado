/**
 * Cliente API para comunicarse con el endpoint de SageMaker a través de API Gateway
 * 
 * IMPORTANTE: Este archivo NO contiene credenciales AWS.
 * Las credenciales se manejan en el backend (Lambda) usando IAM roles.
 */

/**
 * Obtiene la URL de la API desde variables de entorno
 */
const getApiUrl = () => {
  const apiUrl = import.meta.env.VITE_API_URL
  
  if (!apiUrl) {
    throw new Error(
      'VITE_API_URL no está configurada. ' +
      'Crea un archivo .env en la raíz del proyecto con: VITE_API_URL=https://tu-api-gateway-url.execute-api.us-east-1.amazonaws.com'
    )
  }
  
  // Asegurar que la URL no termine con /
  const cleanUrl = apiUrl.replace(/\/$/, '')
  
  // Agregar /predict al final
  // NOTA: Si tu API Gateway tiene un stage (ej: /dev o /prod), 
  // inclúyelo en VITE_API_URL (ej: https://xxx.execute-api.us-east-1.amazonaws.com/dev)
  // El código agregará /predict automáticamente
  return `${cleanUrl}/predict`
}

/**
 * Realiza una predicción de ECG
 * 
 * @param {Object} ecgData - Datos del ECG en formato: { signals: [[[...]]] }
 * @returns {Promise<Object>} Respuesta del modelo con prediction y probability
 * @throws {Error} Si hay un error en la petición o en el servidor
 */
export const predictECG = async (ecgData) => {
  const apiUrl = getApiUrl()
  
  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(ecgData),
    })
    
    // Leer respuesta
    const data = await response.json()
    
    // Si la respuesta no es exitosa, lanzar error
    if (!response.ok) {
      const errorMessage = data.error || data.message || `Error ${response.status}: ${response.statusText}`
      throw new Error(errorMessage)
    }
    
    return data
  } catch (error) {
    // Si es un error de red o fetch falló
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
      // Obtener la URL configurada para mostrar en el error
      const configuredUrl = import.meta.env.VITE_API_URL || 'NO CONFIGURADA'
      
      throw new Error(
        `Error de conexión con la API.\n\n` +
        `URL configurada: ${configuredUrl}\n` +
        `URL completa: ${apiUrl}\n\n` +
        `Verifica:\n` +
        `1. Que la URL en .env sea correcta (sin /predict al final)\n` +
        `2. Que hayas reiniciado 'npm run dev' después de cambiar .env\n` +
        `3. Que API Gateway esté desplegada (tiene un stage)\n` +
        `4. Que CORS esté habilitado en API Gateway\n` +
        `5. Que la ruta POST /predict exista\n\n` +
        `Consulta TROUBLESHOOTING.md para más ayuda.`
      )
    }
    
    // Re-lanzar otros errores
    throw error
  }
}

/**
 * Verifica el estado de la API (opcional)
 */
export const checkApiHealth = async () => {
  const apiUrl = getApiUrl()
  
  try {
    // Hacer una petición OPTIONS para verificar CORS
    const response = await fetch(apiUrl, {
      method: 'OPTIONS',
    })
    
    return response.ok
  } catch (error) {
    return false
  }
}

