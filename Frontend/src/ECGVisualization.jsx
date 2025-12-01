import { useEffect, useRef } from 'react'

/**
 * Componente para visualizar un ECG en un canvas
 * Muestra las 3 señales (canales) del ECG
 */
export default function ECGVisualization({ ecgData, width = 800, height = 400 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!ecgData || !ecgData.signals || ecgData.signals.length === 0) {
      console.log('ECGVisualization: No hay datos para mostrar')
      return
    }

    const canvas = canvasRef.current
    if (!canvas) {
      console.log('ECGVisualization: Canvas ref no disponible')
      return
    }

    const ctx = canvas.getContext('2d')
    const signals = ecgData.signals // Array de 2000 muestras con 3 canales cada una

    // Validar estructura
    if (!Array.isArray(signals) || signals.length === 0) {
      console.error('ECGVisualization: Formato de signals inválido', signals)
      return
    }

    if (!Array.isArray(signals[0]) || signals[0].length !== 3) {
      console.error('ECGVisualization: Cada muestra debe tener 3 canales', signals[0])
      return
    }

    console.log(`ECGVisualization: Dibujando ${signals.length} muestras`)

    // Configuración del canvas
    canvas.width = width
    canvas.height = height
    
    // Limpiar canvas
    ctx.clearRect(0, 0, width, height)

    // Colores para cada canal
    const colors = ['#00ff88', '#ff6b9d', '#4da6ff'] // Verde, Rosa, Azul
    const channelNames = ['Canal I', 'Canal II', 'Canal III']

    // Configuración del gráfico
    const padding = 60
    const plotWidth = width - padding * 2
    const plotHeight = (height - padding * 2) / 3 // Dividir en 3 canales
    const sampleCount = signals.length
    const timeStep = plotWidth / sampleCount

    // Fondo oscuro
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    // Dibujar cada canal
    for (let channel = 0; channel < 3; channel++) {
      const yOffset = padding + channel * (plotHeight + padding / 3)
      
      // Encontrar min y max para este canal (para normalizar)
      const channelValues = signals.map(sample => sample[channel])
      const min = Math.min(...channelValues)
      const max = Math.max(...channelValues)
      const range = max - min || 1 // Evitar división por cero

      // Dibujar grid horizontal
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 0.5
      const gridLines = 5
      for (let i = 0; i <= gridLines; i++) {
        const y = yOffset + (plotHeight / gridLines) * i
        ctx.beginPath()
        ctx.moveTo(padding, y)
        ctx.lineTo(width - padding, y)
        ctx.stroke()
      }

      // Dibujar grid vertical (líneas de tiempo)
      const timeLines = 10
      for (let i = 0; i <= timeLines; i++) {
        const x = padding + (plotWidth / timeLines) * i
        ctx.beginPath()
        ctx.moveTo(x, yOffset)
        ctx.lineTo(x, yOffset + plotHeight)
        ctx.stroke()
      }

      // Dibujar la señal
      ctx.strokeStyle = colors[channel]
      ctx.lineWidth = 1.5
      ctx.beginPath()

      for (let i = 0; i < sampleCount; i++) {
        const x = padding + i * timeStep
        // Normalizar el valor al rango del plot
        const normalized = (channelValues[i] - min) / range
        const y = yOffset + plotHeight - (normalized * plotHeight)

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }

      ctx.stroke()

      // Etiqueta del canal
      ctx.fillStyle = colors[channel]
      ctx.font = '14px monospace'
      ctx.fillText(channelNames[channel], 10, yOffset + plotHeight / 2)
    }

    // Dibujar eje X (tiempo)
    ctx.strokeStyle = '#666'
    ctx.fillStyle = '#999'
    ctx.font = '12px monospace'
    ctx.textAlign = 'center'
    ctx.beginPath()
    ctx.moveTo(padding, height - padding / 2)
    ctx.lineTo(width - padding, height - padding / 2)
    ctx.stroke()
    ctx.fillText('Tiempo (10 segundos)', width / 2, height - 10)

    // Etiquetas de valores
    ctx.textAlign = 'left'
    ctx.fillStyle = '#666'
    ctx.font = '10px monospace'
    ctx.fillText(`ECG: ${sampleCount} muestras, 3 canales`, padding, 20)
    
    console.log('ECGVisualization: Dibujo completado')
  }, [ecgData, width, height])

  if (!ecgData || !ecgData.signals) {
    return (
      <div className="ecg-visualization-placeholder">
        <p>Selecciona un ECG para visualizarlo</p>
      </div>
    )
  }

  return (
    <div className="ecg-visualization">
      <canvas ref={canvasRef} width={width} height={height} />
    </div>
  )
}

