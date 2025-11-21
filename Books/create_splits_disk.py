"""
Script auxiliar para crear splits guardando directamente en disco.
Ejecuta este código en una celda del notebook en lugar de la celda original.
"""

# Reemplaza la celda del PASO 5 con este código:

print("=" * 80)
print("PASO 5: Creando splits train/val/test")
print("=" * 80)

try:
    start_time = time.time()
    
    print(f"\n  Creando splits estratificados (70/15/15) y guardando en disco...")
    print(f"  NOTA: Usando guardado directo en disco para evitar problemas de memoria")
    
    # Usar función que guarda directamente en disco
    result = create_splits_to_disk(
        X, y,
        output_dir=OUTPUT_DIR,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        chunk_size=10000,  # Procesar en chunks de 10k registros
        random_state=42,  # Usar el mismo random_state que create_splits
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = result
    
    # Crear metadata para cada split usando los índices retornados
    print(f"\n  Creando metadata para cada split...")
    metadata_train = metadata.iloc[train_idx].copy().reset_index(drop=True)
    metadata_val = metadata.iloc[val_idx].copy().reset_index(drop=True)
    metadata_test = metadata.iloc[test_idx].copy().reset_index(drop=True)
    
    # Limpiar índices (ya no se necesitan)
    del train_idx, val_idx, test_idx
    
    elapsed = time.time() - start_time
    
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    
    print(f"\n✓ Splits creados y guardados (tiempo: {elapsed:.2f}s)")
    print(f"\n  Train: {n_train} ({n_train/len(X)*100:.1f}%)")
    print(f"    Normales: {(y_train == 0).sum()}")
    print(f"    Anómalos: {(y_train == 1).sum()}")
    print(f"    Guardado en: {OUTPUT_DIR / 'numpy' / 'X_train.npy'}")
    
    print(f"\n  Val: {n_val} ({n_val/len(X)*100:.1f}%)")
    print(f"    Normales: {(y_val == 0).sum()}")
    print(f"    Anómalos: {(y_val == 1).sum()}")
    print(f"    Guardado en: {OUTPUT_DIR / 'numpy' / 'X_val.npy'}")
    
    print(f"\n  Test: {n_test} ({n_test/len(X)*100:.1f}%)")
    print(f"    Normales: {(y_test == 0).sum()}")
    print(f"    Anómalos: {(y_test == 1).sum()}")
    print(f"    Guardado en: {OUTPUT_DIR / 'numpy' / 'X_test.npy'}")

except Exception as e:
    print(f"\n✗ ERROR creando splits: {e}")
    import traceback
    traceback.print_exc()
    raise

