import { useState, useEffect } from 'react'
import { predictECG } from './api/client'
import ecgSamplesData from './data/ecg_samples.json'
import ECGVisualization from './ECGVisualization'
import './App.css'

function App() {
  const [selectedECG, setSelectedECG] = useState(null)
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('unknown')
  const [canvasWidth, setCanvasWidth] = useState(1200)

  const samples = ecgSamplesData.samples

  // Actualizar ancho del canvas cuando cambia el tamaño de ventana
  useEffect(() => {
    const updateWidth = () => {
      setCanvasWidth(Math.min(1200, window.innerWidth - 60))
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  // Verificar estado de la API al cargar
  useEffect(() => {
    checkApiStatus()
  }, [])

  const checkApiStatus = async () => {
    const apiUrl = import.meta.env.VITE_API_URL
    if (!apiUrl) {
      setApiStatus('not-configured')
      return
    }
    setApiStatus('configured')
  }

  const handleECGSelect = (sample) => {
    setSelectedECG(sample)
    setResponse(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!selectedECG) {
      setError('Por favor, selecciona un ECG primero')
      return
    }

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      // Preparar datos en el formato que espera el endpoint
      // El endpoint espera: { "signals": [[[...]]] } con forma [1, 2000, 3]
      const requestData = {
        signals: [selectedECG.signals]  // Envolver en array para batch size = 1
      }

      const result = await predictECG(requestData)
      setResponse(result)
    } catch (err) {
      setError(err.message || 'Error al predecir ECG')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getPredictionSummary = (response) => {
    if (!response) return null

    const prob = response.probability !== undefined 
      ? response.probability 
      : response.prediction

    if (prob === undefined) return null

    const isAnomaly = prob > 0.5
    const confidence = (isAnomaly ? prob : (1 - prob)) * 100

    return {
      isAnomaly,
      probability: prob,
      confidence: confidence.toFixed(2),
      label: isAnomaly ? 'ANÓMALO' : 'NORMAL'
    }
  }

  const summary = getPredictionSummary(response)

  return (
    <div className="app">
      <header className="app-header">
        <h1>🫀 Detección de Anomalías en ECG</h1>
        <p className="subtitle">Demo Técnica - Modelo CNN1D+LSTM en AWS SageMaker</p>
        
        {apiStatus === 'not-configured' && (
          <div className="warning-banner">
            ⚠️ Variable de entorno VITE_API_URL no configurada. 
            Verifica el archivo .env
          </div>
        )}
      </header>

      <div className="container">
        {/* Visualización del ECG seleccionado */}
        <section className="section">
          <h2>📊 Visualización del ECG Seleccionado</h2>
          {selectedECG ? (
            <div className="ecg-visualization-container">
              <ECGVisualization 
                ecgData={selectedECG} 
                width={canvasWidth} 
                height={500}
              />
              <div style={{ marginTop: '15px', color: '#999', fontSize: '0.9em' }}>
                <p><strong style={{ color: '#e0e0e0' }}>{selectedECG.name}</strong> - {selectedECG.description}</p>
                <p>Etiqueta real: <strong style={{ color: selectedECG.label === 1 ? '#ff6b9d' : '#00ff88' }}>{selectedECG.label_text}</strong></p>
                <p style={{ marginTop: '10px', color: '#666', fontSize: '0.85em' }}>
                  Forma de datos: [{selectedECG.signals.length}, {selectedECG.signals[0]?.length || 0}]
                </p>
              </div>
            </div>
          ) : (
            <div className="ecg-visualization-placeholder">
              <p>👆 Selecciona un ECG de la lista inferior para ver su visualización aquí</p>
            </div>
          )}
        </section>

        {/* Sección de selección de ECG */}
        <section className="section">
          <h2>1. Seleccionar ECG de Prueba</h2>
          <div className="ecg-grid">
            {samples.map((sample) => (
              <div
                key={sample.id}
                className={`ecg-card ${selectedECG?.id === sample.id ? 'selected' : ''} ${sample.label === 1 ? 'anomalo' : 'normal'}`}
                onClick={() => handleECGSelect(sample)}
              >
                <div className="ecg-card-header">
                  <span className={`label-badge ${sample.label === 1 ? 'badge-anomalo' : 'badge-normal'}`}>
                    {sample.label_text}
                  </span>
                </div>
                <h3>{sample.name}</h3>
                <p className="ecg-description">{sample.description}</p>
                <div className="ecg-info">
                  <span>Forma: [{sample.signals.length}, {sample.signals[0]?.length || 0}, {sample.signals[0]?.[0]?.length || 0}]</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Botón de predicción */}
        <section className="section">
          <h2>2. Enviar al Modelo</h2>
          <div className="predict-button-container">
            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={!selectedECG || loading}
            >
              {loading ? '🔄 Procesando...' : '🚀 Enviar a Modelo'}
            </button>
          </div>
          {!selectedECG && (
            <p style={{ textAlign: 'center', color: '#666', marginTop: '15px' }}>
              Selecciona un ECG arriba para habilitar la predicción
            </p>
          )}
        </section>

        {/* Resultados */}
        {(response || error) && (
          <section className="section">
            <h2>3. Resultado de la Predicción</h2>

            {error && (
              <div className="error-box">
                <h3>❌ Error</h3>
                <p>{error}</p>
              </div>
            )}

            {response && (
              <div className="result-container">
                {/* Resumen amigable */}
                {summary && (
                  <div className={`summary-box ${summary.isAnomaly ? 'summary-anomalo' : 'summary-normal'}`}>
                    <h3>📊 Resumen</h3>
                    <div className="summary-content">
                      <div className="summary-item">
                        <span className="summary-label">Predicción:</span>
                        <span className={`summary-value ${summary.isAnomaly ? 'value-anomalo' : 'value-normal'}`}>
                          {summary.label}
                        </span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Probabilidad de anomalía:</span>
                        <span className="summary-value">{summary.probability.toFixed(4)}</span>
                      </div>
                      <div className="summary-item">
                        <span className="summary-label">Confianza:</span>
                        <span className="summary-value">{summary.confidence}%</span>
                      </div>
                      
                      {/* Comparar con etiqueta real si está disponible */}
                      {selectedECG && (
                        <div className="summary-item comparison">
                          <span className="summary-label">Comparación:</span>
                          <span className={`summary-value ${selectedECG.label === (summary.isAnomaly ? 1 : 0) ? 'correct' : 'incorrect'}`}>
                            {selectedECG.label === (summary.isAnomaly ? 1 : 0) 
                              ? '✅ PREDICCIÓN CORRECTA' 
                              : '❌ PREDICCIÓN INCORRECTA'}
                          </span>
                          <div className="comparison-detail">
                            Real: {selectedECG.label_text} | Predicho: {summary.label}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* JSON Raw */}
                <div className="json-box">
                  <h3>📄 Respuesta JSON (Raw)</h3>
                  <pre>{JSON.stringify(response, null, 2)}</pre>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Información del modelo */}
        <section className="section info-section">
          <h2>ℹ️ Información del Modelo</h2>
          <div className="info-grid">
            <div className="info-item">
              <strong>Tipo:</strong> CNN1D + LSTM
            </div>
            <div className="info-item">
              <strong>Input:</strong> [1, 2000, 3] (2000 muestras, 3 canales)
            </div>
            <div className="info-item">
              <strong>Frecuencia:</strong> 200 Hz
            </div>
            <div className="info-item">
              <strong>Duración:</strong> 10 segundos
            </div>
            <div className="info-item">
              <strong>Output:</strong> Probabilidad de anomalía (0-1)
            </div>
            <div className="info-item">
              <strong>Threshold:</strong> &gt; 0.5 = Anómalo
            </div>
          </div>
          <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', color: '#999', fontSize: '0.9em' }}>
            <p><strong style={{ color: '#00ff88' }}>Arquitectura:</strong> Frontend → API Gateway → Lambda → SageMaker</p>
            <p><strong style={{ color: '#00ff88' }}>Seguridad:</strong> Sin credenciales AWS expuestas en el frontend</p>
          </div>
        </section>
      </div>

    </div>
  )
}

export default App

