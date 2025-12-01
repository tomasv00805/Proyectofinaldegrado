"""
Script para generar archivo JSON con ejemplos de ECG desde los datos reales.
Esto crea ecg_samples.json que el frontend usará para cargar ECGs de prueba.

Los datos originales provienen de:
- data/Datos_supervisados/tensors_200hz/X_test.pt (señales ECG)
- data/Datos_supervisados/tensors_200hz/y_test.pt (etiquetas: 0=normal, 1=anómalo)

Estos datos fueron generados previamente desde el dataset MIMIC-IV ECG mediante
el pipeline de procesamiento del proyecto.
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys

# Configurar la codificación de salida para evitar errores de emoji en Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Rutas de los archivos
DATA_DIR = Path("../data/Datos_supervisados/tensors_200hz")
OUTPUT_FILE = Path("src/data/ecg_samples.json")

# Configuración: cuántas muestras generar
NUM_NORMAL = 8      # Número de ECGs normales
NUM_ANOMALO = 8     # Número de ECGs anómalos
MAX_SEARCH = 1000   # Buscar en los primeros N ECGs del dataset

# Crear directorio de salida si no existe
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERANDO ARCHIVO DE MUESTRAS DE ECG")
print("=" * 70)
print(f"Configuración:")
print(f"  - ECGs Normales: {NUM_NORMAL}")
print(f"  - ECGs Anómalos: {NUM_ANOMALO}")
print(f"  - Total: {NUM_NORMAL + NUM_ANOMALO} muestras")
print("=" * 70)

try:
    # Cargar datos
    print(f"\n📂 Cargando datos desde: {DATA_DIR}")
    print(f"   Ruta absoluta: {DATA_DIR.resolve()}")
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio: {DATA_DIR.resolve()}")
    
    X_test_path = DATA_DIR / "X_test.pt"
    y_test_path = DATA_DIR / "y_test.pt"
    
    if not X_test_path.exists():
        raise FileNotFoundError(f"No se encontró: {X_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"No se encontró: {y_test_path}")
    
    X_test = torch.load(X_test_path, map_location='cpu')
    y_test = torch.load(y_test_path, map_location='cpu')
    
    # Convertir a numpy si es necesario
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    print(f"✅ Datos cargados exitosamente")
    print(f"   Total de ECGs disponibles: {len(X_test)}")
    print(f"   Forma de X_test: {X_test.shape}")
    print(f"   Forma de y_test: {y_test.shape}")
    
    # Validar forma de datos
    if len(X_test.shape) != 3 or X_test.shape[1] != 2000 or X_test.shape[2] != 3:
        print(f"⚠️  ADVERTENCIA: Forma inesperada. Esperado: (N, 2000, 3), Obtenido: {X_test.shape}")
    
    # Estadísticas del dataset
    if len(y_test) > 0:
        unique_labels, counts = np.unique(y_test, return_counts=True)
        print(f"\n📊 Estadísticas del dataset:")
        for label, count in zip(unique_labels, counts):
            label_text = "NORMAL" if label == 0 else "ANÓMALO"
            percentage = (count / len(y_test)) * 100
            print(f"   {label_text} (label={label}): {count} muestras ({percentage:.1f}%)")
    
    # Seleccionar ejemplos: buscar normales y anómalos
    print(f"\n🔍 Buscando {NUM_NORMAL} ECGs normales y {NUM_ANOMALO} ECGs anómalos...")
    normal_indices = []
    anomalo_indices = []
    
    search_limit = min(MAX_SEARCH, len(X_test))
    
    for i in range(search_limit):
        label = int(y_test[i].item() if hasattr(y_test[i], 'item') else y_test[i])
        
        if label == 0 and len(normal_indices) < NUM_NORMAL:
            normal_indices.append(i)
        elif label == 1 and len(anomalo_indices) < NUM_ANOMALO:
            anomalo_indices.append(i)
        
        # Detener si ya tenemos suficientes de ambos tipos
        if len(normal_indices) >= NUM_NORMAL and len(anomalo_indices) >= NUM_ANOMALO:
            break
    
    print(f"✅ ECGs encontrados:")
    print(f"   Normales: {len(normal_indices)}/{NUM_NORMAL}")
    print(f"   Anómalos: {len(anomalo_indices)}/{NUM_ANOMALO}")
    
    if len(normal_indices) < NUM_NORMAL:
        print(f"⚠️  ADVERTENCIA: Solo se encontraron {len(normal_indices)} ECGs normales de {NUM_NORMAL} solicitados")
    if len(anomalo_indices) < NUM_ANOMALO:
        print(f"⚠️  ADVERTENCIA: Solo se encontraron {len(anomalo_indices)} ECGs anómalos de {NUM_ANOMALO} solicitados")
    
    # Crear lista de muestras
    samples = []
    
    # Agregar normales
    normal_count = 0
    for idx in normal_indices:
        ecg_data = X_test[idx]
        if isinstance(ecg_data, torch.Tensor):
            ecg_data = ecg_data.numpy()
        
        normal_count += 1
        samples.append({
            'id': f'normal_{idx}',
            'name': f'ECG Normal #{normal_count}',
            'signals': ecg_data.tolist(),
            'label': 0,
            'label_text': 'NORMAL',
            'description': f'ECG normal de prueba (índice {idx} del dataset de test)',
            'dataset_index': int(idx)
        })
    
    # Agregar anómalos
    anomalo_count = 0
    for idx in anomalo_indices:
        ecg_data = X_test[idx]
        if isinstance(ecg_data, torch.Tensor):
            ecg_data = ecg_data.numpy()
        
        anomalo_count += 1
        samples.append({
            'id': f'anomalo_{idx}',
            'name': f'ECG Anómalo #{anomalo_count}',
            'signals': ecg_data.tolist(),
            'label': 1,
            'label_text': 'ANÓMALO',
            'description': f'ECG anómalo de prueba (índice {idx} del dataset de test)',
            'dataset_index': int(idx)
        })
    
    # Mezclar las muestras para que no estén todas juntas por tipo
    np.random.seed(42)  # Para reproducibilidad
    indices = np.random.permutation(len(samples))
    samples = [samples[i] for i in indices]
    
    # Crear estructura final
    output_data = {
        'samples': samples,
        'metadata': {
            'total_samples': len(samples),
            'normal_samples': len([s for s in samples if s['label'] == 0]),
            'anomalo_samples': len([s for s in samples if s['label'] == 1]),
            'generated_from': {
                'dataset_path': str(DATA_DIR),
                'source_files': ['X_test.pt', 'y_test.pt'],
                'dataset_size': len(X_test),
                'search_limit': search_limit
            },
            'format': {
                'shape': '[2000, 3]',
                'description': 'ECG con 2000 muestras temporales y 3 canales (I, II, III)',
                'frequency_hz': 200,
                'duration_seconds': 10,
                'channels': ['Canal I', 'Canal II', 'Canal III']
            }
        }
    }
    
    # Guardar archivo
    print(f"\n💾 Guardando muestras en: {OUTPUT_FILE}")
    print(f"   Ruta absoluta: {OUTPUT_FILE.resolve()}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Archivo generado exitosamente!")
    print(f"   📄 Archivo: {OUTPUT_FILE}")
    print(f"   📦 Tamaño: {file_size_mb:.2f} MB")
    print(f"   📊 Total de muestras: {len(samples)}")
    print(f"   ✅ Normales: {len([s for s in samples if s['label'] == 0])}")
    print(f"   ⚠️  Anómalos: {len([s for s in samples if s['label'] == 1])}")
    print(f"\n💡 Para usar estas muestras en el frontend:")
    print(f"   1. Reinicia el servidor de desarrollo (npm run dev)")
    print(f"   2. Las muestras aparecerán automáticamente en la interfaz")
    print(f"\n{'=' * 70}")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: No se encontraron los archivos de datos")
    print(f"   {e}")
    print(f"\n📝 Verificación:")
    print(f"   1. ¿Ejecutaste este script desde el directorio Frontend/?")
    print(f"      Actual: {Path.cwd()}")
    print(f"   2. ¿Existe el directorio de datos?")
    print(f"      Esperado: {DATA_DIR.resolve()}")
    print(f"   3. ¿Tienes los archivos X_test.pt y y_test.pt?")
    print(f"\n💡 Solución:")
    print(f"   cd Frontend")
    print(f"   python generate_ecg_samples.py")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
