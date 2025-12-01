"""
Script para generar un test event válido para Lambda
Genera un ECG de prueba con la forma correcta [1, 2000, 3]
"""

import json
import numpy as np

# Generar ECG de prueba con forma [1, 2000, 3]
print("Generando ECG de prueba...")
ecg_test = np.random.randn(1, 2000, 3).astype(float).tolist()

# Crear test event en el formato que espera Lambda
test_event = {
    "body": json.dumps({
        "signals": ecg_test
    })
}

print("\n" + "=" * 70)
print("TEST EVENT PARA LAMBDA")
print("=" * 70)
print("\nCopia este JSON y úsalo en Lambda Console como test event:\n")
print(json.dumps(test_event, indent=2))

print("\n" + "=" * 70)
print("INFORMACIÓN")
print("=" * 70)
print(f"Forma del ECG: [{len(ecg_test)}, {len(ecg_test[0])}, {len(ecg_test[0][0])}]")
print("✅ Forma correcta: [1, 2000, 3]")
print("\nTamaño aproximado del JSON:", len(json.dumps(test_event)), "bytes")

