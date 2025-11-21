"""
Script para limpiar referencias a arrays de memoria mapeada antes de recrear splits.
Ejecuta este código en una celda del notebook antes de ejecutar create_splits_to_disk.
"""

# Cerrar cualquier referencia a arrays de memoria mapeada
import gc

# Intentar eliminar referencias a arrays cargados previamente
try:
    if 'X_train' in globals():
        del X_train
    if 'X_val' in globals():
        del X_val
    if 'X_test' in globals():
        del X_test
    if 'y_train' in globals():
        del y_train
    if 'y_val' in globals():
        del y_val
    if 'y_test' in globals():
        del y_test
except:
    pass

# Forzar garbage collection para liberar archivos
gc.collect()

print("✓ Referencias a arrays anteriores eliminadas")
print("  Ahora puedes ejecutar create_splits_to_disk sin problemas")

