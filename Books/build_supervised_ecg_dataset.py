#!/usr/bin/env python
"""
Script principal para construir el dataset supervisado binario de ECG.

Este script ejecuta el pipeline completo:
1. Procesa datasets PTB-XL y MIMIC-IV-ECG
2. Etiqueta registros (NORMAL vs ANÓMALO)
3. Filtra señales de mala calidad
4. Aplica filtrado y normalización
5. Selecciona leads II, V1, V5
6. Resamplea a 10s y 500 Hz
7. Balancea el dataset
8. Genera splits train/val/test (70/15/15) + 10 folds
9. Guarda todo en /data/Datos_supervisados/

Uso:
    python build_supervised_ecg_dataset.py [--max-ptb N] [--max-mimic N] [--no-balance] [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importar el módulo del pipeline
from supervised_ecg_pipeline import (
    OUTPUT_DIR,
    create_splits,
    create_stratified_folds,
    save_dataset,
    plot_ecg_comparison,
    ensure_dir,
    balance_dataset,
)

# Intentar importar versión optimizada (paralela)
try:
    from supervised_ecg_pipeline_fast import (
        process_mimic_dataset_fast,
        process_ptbxl_dataset_fast,
    )
    FAST_VERSION_AVAILABLE = True
except ImportError:
    # Si no está disponible, usar versión lenta
    from supervised_ecg_pipeline import (
        process_mimic_dataset,
        process_ptbxl_dataset,
    )
    # Crear wrappers para compatibilidad
    def process_ptbxl_dataset_fast(*args, **kwargs):
        df = process_ptbxl_dataset(*args, **kwargs)
        if len(df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        signals = np.stack(df["signal"].values, axis=0)
        labels = df["label"].values
        metadata = df.drop(columns=["signal"])
        return signals, labels, metadata
    
    def process_mimic_dataset_fast(*args, **kwargs):
        df = process_mimic_dataset(*args, **kwargs)
        if len(df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        signals = np.stack(df["signal"].values, axis=0)
        labels = df["label"].values
        metadata = df.drop(columns=["signal"])
        return signals, labels, metadata
    
    FAST_VERSION_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Construir dataset supervisado binario de ECG (NORMAL vs ANÓMALO)"
    )
    parser.add_argument(
        "--max-ptb",
        type=int,
        default=None,
        help="Máximo de registros a procesar de PTB-XL (None = todos)",
    )
    parser.add_argument(
        "--max-mimic",
        type=int,
        default=None,
        help="Máximo de registros a procesar de MIMIC (None = todos)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="No balancear el dataset (usar todos los registros)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir archivos existentes",
    )
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=50.0,
        help="Frecuencia del filtro notch (50 o 60 Hz)",
    )
    parser.add_argument(
        "--normalize-method",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="Método de normalización",
    )
    parser.add_argument(
        "--reject-unvalidated",
        action="store_true",
        help="Rechazar reportes no validados en PTB-XL",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Deshabilitar verificación de calidad de señal",
    )
    parser.add_argument(
        "--create-examples",
        action="store_true",
        help="Crear ejemplos visuales de señales procesadas",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Número de ejemplos visuales a crear",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Número de workers para paralelización (None = auto)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Usar versión optimizada con paralelización (por defecto)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Usar versión lenta (sin paralelización)",
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Deshabilitar pre-filtrado rápido (más lento, pero procesa todos los registros)",
    )
    parser.add_argument(
        "--minimal-quality",
        action="store_true",
        help="Modo calidad mínima: deshabilita checks menos críticos para mayor velocidad",
    )
    
    args = parser.parse_args()
    
    # Determinar si usar versión rápida
    use_fast = args.fast and not args.slow and FAST_VERSION_AVAILABLE
    if use_fast and args.n_workers is None:
        import multiprocessing as mp
        args.n_workers = max(1, mp.cpu_count() - 1)
    
    print("=" * 80)
    print("PIPELINE DE DATASET SUPERVISADO ECG")
    print("=" * 80)
    print(f"\nConfiguración:")
    print(f"  PTB-XL máximo: {args.max_ptb if args.max_ptb else 'Todos'}")
    print(f"  MIMIC máximo: {args.max_mimic if args.max_mimic else 'Todos'}")
    print(f"  Balancear: {not args.no_balance}")
    print(f"  Notch frecuencia: {args.notch_freq} Hz")
    print(f"  Normalización: {args.normalize_method}")
    print(f"  Verificación de calidad: {not args.no_quality_check}")
    print(f"  Rechazar no validados: {args.reject_unvalidated}")
    print(f"  Sobrescribir: {args.overwrite}")
    print(f"  Versión: {'RÁPIDA (paralela)' if use_fast else 'LENTA (secuencial)'}")
    if use_fast:
        print(f"  Workers: {args.n_workers}")
    print(f"  Directorio de salida: {OUTPUT_DIR}")
    print()
    
    # =====================================================================================
    # 1. Procesar PTB-XL
    # =====================================================================================
    print("=" * 80)
    print("PASO 1: Procesando PTB-XL")
    print("=" * 80)
    
    # Inicializar variables
    ptbxl_signals = None
    ptbxl_labels = None
    ptbxl_df = None
    mimic_signals = None
    mimic_labels = None
    mimic_df = None
    
    try:
        if use_fast:
            ptbxl_signals, ptbxl_labels, ptbxl_df = process_ptbxl_dataset_fast(
                overwrite=args.overwrite,
                apply_quality_check=not args.no_quality_check and not args.minimal_quality,
                apply_notch=True,
                notch_freq=args.notch_freq,
                normalize_method=args.normalize_method,
                reject_unvalidated=args.reject_unvalidated,
                max_records=args.max_ptb,
                n_workers=args.n_workers,
                verbose=True,
                prefilter_labels=False,  # Procesar todos directamente (más rápido)
            )
            
            if len(ptbxl_signals) == 0:
                print("⚠ No se procesaron registros de PTB-XL")
                ptbxl_signals = None
                ptbxl_labels = None
                ptbxl_df = None
            else:
                print(f"✓ PTB-XL: {len(ptbxl_signals)} registros procesados")
                print(f"  - Normales: {(ptbxl_labels == 0).sum()}")
                print(f"  - Anómalos: {(ptbxl_labels == 1).sum()}")
        else:
            ptbxl_df_full = process_ptbxl_dataset(
                overwrite=args.overwrite,
                apply_quality_check=not args.no_quality_check,
                apply_notch=True,
                notch_freq=args.notch_freq,
                normalize_method=args.normalize_method,
                reject_unvalidated=args.reject_unvalidated,
                max_records=args.max_ptb,
                verbose=True,
            )
            
            if len(ptbxl_df_full) == 0:
                print("⚠ No se procesaron registros de PTB-XL")
                ptbxl_signals = None
                ptbxl_labels = None
                ptbxl_df = None
            else:
                ptbxl_signals = np.stack(ptbxl_df_full["signal"].values, axis=0)
                ptbxl_labels = ptbxl_df_full["label"].values
                ptbxl_df = ptbxl_df_full.drop(columns=["signal"])
                print(f"✓ PTB-XL: {len(ptbxl_signals)} registros procesados")
                print(f"  - Normales: {(ptbxl_labels == 0).sum()}")
                print(f"  - Anómalos: {(ptbxl_labels == 1).sum()}")
    
    except Exception as e:
        print(f"✗ Error procesando PTB-XL: {e}")
        import traceback
        traceback.print_exc()
        ptbxl_df = None
    
    # =====================================================================================
    # 2. Procesar MIMIC
    # =====================================================================================
    print("\n" + "=" * 80)
    print("PASO 2: Procesando MIMIC-IV-ECG")
    print("=" * 80)
    
    try:
        if use_fast:
            mimic_signals, mimic_labels, mimic_df = process_mimic_dataset_fast(
                overwrite=args.overwrite,
                apply_quality_check=not args.no_quality_check and not args.minimal_quality,
                apply_notch=True,
                notch_freq=args.notch_freq,
                normalize_method=args.normalize_method,
                max_records=args.max_mimic,
                n_workers=args.n_workers,
                verbose=True,
                prefilter_labels=False,  # Procesar todos directamente (más rápido)
            )
            
            if len(mimic_signals) == 0:
                print("⚠ No se procesaron registros de MIMIC")
                mimic_signals = None
                mimic_labels = None
                mimic_df = None
            else:
                print(f"  [DEBUG] Versión RÁPIDA: Calculando resumen...")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Calculando longitud...")
                sys.stdout.flush()
                n_registros = len(mimic_signals)
                print(f"✓ MIMIC: {n_registros} registros procesados")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Contando normales...")
                sys.stdout.flush()
                n_normales = int((mimic_labels == 0).sum())
                print(f"  - Normales: {n_normales}")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Contando anómalos...")
                sys.stdout.flush()
                n_anomalos = int((mimic_labels == 1).sum())
                print(f"  - Anómalos: {n_anomalos}")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Calculando memoria...")
                sys.stdout.flush()
                try:
                    memoria_gb = mimic_signals.nbytes / 1024**3
                    print(f"  - Memoria: {memoria_gb:.2f} GB")
                except Exception as e:
                    print(f"  - Memoria: [error: {e}]")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Versión rápida: MIMIC procesado exitosamente, continuando...")
                sys.stdout.flush()
        else:
            mimic_df_full = process_mimic_dataset(
                overwrite=args.overwrite,
                report_column="report_1",
                apply_quality_check=not args.no_quality_check,
                apply_notch=True,
                notch_freq=args.notch_freq,
                normalize_method=args.normalize_method,
                max_records=args.max_mimic,
                verbose=True,
            )
            
            if len(mimic_df_full) == 0:
                print("⚠ No se procesaron registros de MIMIC")
                mimic_signals = None
                mimic_labels = None
                mimic_df = None
            else:
                print(f"  [DEBUG] Versión LENTA: Extrayendo señales de DataFrame...")
                print(f"  [DEBUG] Total registros: {len(mimic_df_full)}")
                sys.stdout.flush()
                
                # Extraer señales de forma más eficiente (en chunks si es necesario)
                signals_list = mimic_df_full["signal"].values.tolist()
                print(f"  [DEBUG] Lista de señales creada, haciendo stack (esto puede tardar)...")
                sys.stdout.flush()
                
                # np.stack puede ser muy lento con muchos registros, hacer en chunks grandes
                try:
                    mimic_signals = np.stack(signals_list, axis=0)
                    del signals_list  # Liberar memoria
                    print(f"  [DEBUG] ✓ Stack completado: shape {mimic_signals.shape}")
                except MemoryError:
                    print(f"  [ERROR] Error de memoria al hacer stack. Considera usar --max-mimic para limitar registros.")
                    raise
                except Exception as e:
                    print(f"  [ERROR] Error al hacer stack: {e}")
                    raise
                
                sys.stdout.flush()
                
                print(f"  [DEBUG] Extrayendo labels...")
                sys.stdout.flush()
                mimic_labels = mimic_df_full["label"].values
                print(f"  [DEBUG] ✓ Labels extraídos")
                sys.stdout.flush()
                
                print(f"  [DEBUG] Limpiando DataFrame...")
                sys.stdout.flush()
                mimic_df = mimic_df_full.drop(columns=["signal"])
                del mimic_df_full  # Liberar memoria del DataFrame completo
                print(f"  [DEBUG] ✓ DataFrame limpio, imprimiendo resumen...")
                sys.stdout.flush()
                
                print(f"✓ MIMIC: {len(mimic_signals)} registros procesados")
                sys.stdout.flush()
                print(f"  - Normales: {(mimic_labels == 0).sum()}")
                sys.stdout.flush()
                print(f"  - Anómalos: {(mimic_labels == 1).sum()}")
                sys.stdout.flush()
                
                try:
                    memoria_gb = mimic_signals.nbytes / 1024**3
                    print(f"  - Memoria: {memoria_gb:.2f} GB")
                except Exception as e:
                    print(f"  - Memoria: [error calculando: {e}]")
                sys.stdout.flush()
                
                print(f"  [DEBUG] MIMIC procesado exitosamente, continuando...")
                sys.stdout.flush()
    
    except Exception as e:
        print(f"✗ Error procesando MIMIC: {e}")
        import traceback
        traceback.print_exc()
        mimic_signals = None
        mimic_labels = None
        mimic_df = None
    
    # Forzar flush de buffers y verificar estado
    sys.stdout.flush()
    sys.stderr.flush()
    
    print("\n[DEBUG] Verificando estado después de procesar MIMIC...")
    print(f"[DEBUG] PTB-XL signals: {'OK' if ptbxl_signals is not None and len(ptbxl_signals) > 0 else 'None/Vacío'}")
    print(f"[DEBUG] MIMIC signals: {'OK' if mimic_signals is not None and len(mimic_signals) > 0 else 'None/Vacío'}")
    sys.stdout.flush()
    
    # =====================================================================================
    # 3. Construir dataset combinado
    # =====================================================================================
    print("\n" + "=" * 80)
    print("PASO 3: Construyendo dataset combinado")
    print("=" * 80)
    sys.stdout.flush()
    
    if (ptbxl_signals is None or (hasattr(ptbxl_signals, '__len__') and len(ptbxl_signals) == 0)) and (mimic_signals is None or (hasattr(mimic_signals, '__len__') and len(mimic_signals) == 0)):
        print("✗ No hay datos para procesar")
        print(f"  PTB-XL: {ptbxl_signals is None}")
        print(f"  MIMIC: {mimic_signals is None}")
        sys.exit(1)
    
    try:
        print(f"[DEBUG] Iniciando construcción de dataset combinado...")
        
        # Las señales ya están extraídas en la versión rápida
        # Para la versión lenta, ya las extrajimos arriba
        ptbxl_df_clean = ptbxl_df if ptbxl_df is not None else pd.DataFrame()
        mimic_df_clean = mimic_df if mimic_df is not None else pd.DataFrame()
        
        # Combinar señales y labels
        all_signals = []
        all_labels = []
        all_metadata = []
        
        if ptbxl_signals is not None and len(ptbxl_signals) > 0:
            print(f"  Agregando PTB-XL: {len(ptbxl_signals)} señales")
            all_signals.append(ptbxl_signals)
            all_labels.append(ptbxl_labels)
            if len(ptbxl_df_clean) > 0:
                all_metadata.append(ptbxl_df_clean)
        
        if mimic_signals is not None and len(mimic_signals) > 0:
            print(f"  Agregando MIMIC: {len(mimic_signals)} señales")
            all_signals.append(mimic_signals)
            all_labels.append(mimic_labels)
            if len(mimic_df_clean) > 0:
                all_metadata.append(mimic_df_clean)
        
        if not all_signals:
            print("✗ No hay señales para combinar")
            sys.exit(1)
        
        print(f"  Combinando {len(all_signals)} datasets...")
        print(f"  Total de señales a combinar: {sum(len(s) for s in all_signals)}")
        sys.stdout.flush()
        
        print(f"  Combinando señales (esto puede tardar si hay muchas)...")
        sys.stdout.flush()
        X = np.concatenate(all_signals, axis=0)
        sys.stdout.flush()
        print(f"  ✓ Señales combinadas: {X.shape}")
        sys.stdout.flush()
        
        print(f"  Combinando labels...")
        sys.stdout.flush()
        y = np.concatenate(all_labels, axis=0)
        print(f"  ✓ Labels combinados: {len(y)} labels")
        sys.stdout.flush()
        
        if all_metadata:
            print(f"  Combinando metadatos ({len(all_metadata)} dataframes)...")
            sys.stdout.flush()
            metadata = pd.concat(all_metadata, ignore_index=True)
            print(f"  ✓ Metadatos combinados: {len(metadata)} registros")
        else:
            print(f"  ⚠ No hay metadatos para combinar, creando metadata vacío...")
            metadata = pd.DataFrame()
        sys.stdout.flush()
        
        # Liberar memoria de listas intermedias
        del all_signals, all_labels, all_metadata
        if ptbxl_signals is not None:
            del ptbxl_signals, ptbxl_labels
        if mimic_signals is not None:
            del mimic_signals, mimic_labels
        
        print(f"\n✓ Dataset combinado: {len(X)} registros")
        print(f"  - Normales: {(y == 0).sum()}")
        print(f"  - Anómalos: {(y == 1).sum()}")
        print(f"  - Shape de señales: {X.shape}")
        print(f"  - Memoria estimada: {X.nbytes / 1024**3:.2f} GB")
        
        # Balancear si se solicita
        if not args.no_balance:
            print("\nBalanceando dataset...")
            X_balanced, y_balanced, balanced_indices = balance_dataset(X, y, return_indices=True)
            
            # Mapear metadatos usando los índices seleccionados
            print(f"  Mapeando metadatos...")
            X = X_balanced
            y = y_balanced
            metadata = metadata.iloc[balanced_indices].copy()
            metadata.reset_index(drop=True, inplace=True)
            
            # Liberar memoria
            del X_balanced, y_balanced, balanced_indices
            
            print(f"✓ Dataset balanceado: {len(X)} registros")
            print(f"  - Normales: {(y == 0).sum()}")
            print(f"  - Anómalos: {(y == 1).sum()}")
            print(f"  - Memoria estimada: {X.nbytes / 1024**3:.2f} GB")
    
    except Exception as e:
        print(f"✗ Error construyendo dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =====================================================================================
    # 4. Crear splits train/val/test
    # =====================================================================================
    print("\n" + "=" * 80)
    print("PASO 4: Creando splits train/val/test")
    print("=" * 80)
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = create_splits(
            X, y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        # Crear metadata para cada split
        train_indices = np.arange(len(y_train))
        val_indices = np.arange(len(y_train), len(y_train) + len(y_val))
        test_indices = np.arange(len(y_train) + len(y_val), len(y_train) + len(y_val) + len(y_test))
        
        metadata_train = metadata.iloc[train_indices].copy()
        metadata_val = metadata.iloc[val_indices].copy()
        metadata_test = metadata.iloc[test_indices].copy()
        
        print(f"✓ Splits creados:")
        print(f"  - Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    * Normales: {(y_train == 0).sum()}")
        print(f"    * Anómalos: {(y_train == 1).sum()}")
        print(f"  - Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"    * Normales: {(y_val == 0).sum()}")
        print(f"    * Anómalos: {(y_val == 1).sum()}")
        print(f"  - Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"    * Normales: {(y_test == 0).sum()}")
        print(f"    * Anómalos: {(y_test == 1).sum()}")
    
    except Exception as e:
        print(f"✗ Error creando splits: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =====================================================================================
    # 5. Crear folds estratificados
    # =====================================================================================
    print("\n" + "=" * 80)
    print("PASO 5: Creando folds estratificados (cross-validation)")
    print("=" * 80)
    
    try:
        folds = create_stratified_folds(X_train, y_train, n_splits=10)
        folds_train = [fold[0] for fold in folds]
        folds_val = [fold[1] for fold in folds]
        
        print(f"✓ {len(folds)} folds estratificados creados")
        for i, (train_idx, val_idx) in enumerate(folds):
            print(f"  Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
            print(f"    * Train - Normales: {(y_train[train_idx] == 0).sum()}, "
                  f"Anómalos: {(y_train[train_idx] == 1).sum()}")
            print(f"    * Val - Normales: {(y_train[val_idx] == 0).sum()}, "
                  f"Anómalos: {(y_train[val_idx] == 1).sum()}")
    
    except Exception as e:
        print(f"✗ Error creando folds: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =====================================================================================
    # 6. Guardar dataset
    # =====================================================================================
    print("\n" + "=" * 80)
    print("PASO 6: Guardando dataset")
    print("=" * 80)
    
    try:
        print("\nPreparando para guardar...")
        print(f"  X_train: {X_train.shape}, tamaño estimado: {X_train.nbytes / 1024**3:.2f} GB")
        print(f"  X_val: {X_val.shape}, tamaño estimado: {X_val.nbytes / 1024**3:.2f} GB")
        print(f"  X_test: {X_test.shape}, tamaño estimado: {X_test.nbytes / 1024**3:.2f} GB")
        
        save_dataset(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            metadata_train, metadata_val, metadata_test,
            folds_train, folds_val,
            output_dir=OUTPUT_DIR,
        )
        print("✓ Dataset guardado exitosamente")
        
        # Liberar memoria después de guardar
        del X_train, y_train, X_val, y_val, X_test, y_test
        del metadata_train, metadata_val, metadata_test
        del folds_train, folds_val
    
    except Exception as e:
        print(f"✗ Error guardando dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =====================================================================================
    # 7. Crear ejemplos visuales (opcional)
    # =====================================================================================
    if args.create_examples:
        print("\n" + "=" * 80)
        print("PASO 7: Creando ejemplos visuales")
        print("=" * 80)
        
        try:
            ensure_dir(OUTPUT_DIR / "raw_examples")
            
            n_examples = min(args.n_examples, len(X_test))
            indices = np.random.choice(len(X_test), size=n_examples, replace=False)
            
            for i, idx in enumerate(indices):
                signal = X_test[idx]
                label = y_test[idx]
                record_id = metadata_test.iloc[idx]["record_id"]
                
                title = f"{record_id} - {'NORMAL' if label == 0 else 'ANÓMALO'}"
                save_path = OUTPUT_DIR / "raw_examples" / f"example_{i+1:03d}_{record_id.replace('/', '_')}.png"
                
                plot_ecg_comparison(
                    raw=signal,  # Ya está normalizado
                    filtered=None,
                    normalized=signal,
                    title=title,
                    save_path=save_path,
                )
            
            print(f"✓ {n_examples} ejemplos visuales creados en {OUTPUT_DIR / 'raw_examples'}")
        
        except Exception as e:
            print(f"⚠ Error creando ejemplos visuales: {e}")
            import traceback
            traceback.print_exc()
    
    # =====================================================================================
    # Resumen final
    # =====================================================================================
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"\n✓ Pipeline completado exitosamente")
    print(f"\nDataset guardado en: {OUTPUT_DIR}")
    print(f"\nEstructura:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    metadata/")
    print(f"      master_labels.csv")
    print(f"      master_labels_full.csv")
    print(f"      folds_train_indices.npy")
    print(f"      folds_val_indices.npy")
    print(f"    numpy/")
    print(f"      X_train.npy, y_train.npy")
    print(f"      X_val.npy, y_val.npy")
    print(f"      X_test.npy, y_test.npy")
    print(f"    raw_examples/")
    print(f"      (ejemplos visuales)")
    print(f"\nEstadísticas:")
    print(f"  Total: {len(X)} registros")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Shape de señales: {X.shape} (N, T, C) donde T={X.shape[1]}, C={X.shape[2]}")
    print(f"  Leads: II, V1, V5")
    print(f"  Frecuencia de muestreo: 500 Hz")
    print(f"  Duración: 10 segundos")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

