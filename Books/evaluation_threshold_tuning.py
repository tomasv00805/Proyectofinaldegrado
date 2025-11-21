"""
Módulo de evaluación y búsqueda de umbral óptimo para detección de anomalías en ECG.

Este módulo proporciona funciones para:
- Calcular errores de reconstrucción con etiquetas
- Buscar el umbral óptimo basado en percentiles
- Evaluar el modelo con métricas completas
- Visualizar resultados

Autor: Sistema de evaluación mejorado
Fecha: 2025-11-14

================================================================================
🎯 CÓMO CAMBIAR EL UMBRAL PARA DETECTAR ANOMALÍAS
================================================================================

El umbral determina cuándo un ECG se clasifica como anómalo:
  - Si error_reconstrucción > umbral → ECG es ANÓMALO (clase 1)
  - Si error_reconstrucción <= umbral → ECG es NORMAL (clase 0)

OPCIÓN 1: Búsqueda automática (recomendado)
  - Usa la función find_optimal_threshold()
  - Prueba varios percentiles y selecciona el mejor según F2-score
  - Parámetros clave:
    * percentiles: Lista de percentiles a probar [80, 85, 90, ...]
    * max_fpr: FPR máximo permitido (ej: 0.05 = 5%)

OPCIÓN 2: Umbral fijo manual
  - Define directamente: threshold = 0.0001
  - Usa predict_with_threshold() con ese valor

OPCIÓN 3: Basado en estadísticas
  - Usa percentil: threshold = np.percentile(errors, 95)
  - O media + desviación: threshold = mean + 3*std

================================================================================
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# Importar el tipo del modelo si está disponible
try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        # Solo para type checking, no se importa en runtime
        pass
except ImportError:
    pass


def compute_reconstruction_errors_with_labels(
    model: torch.nn.Module,
    normal_loader: Optional[DataLoader],
    anomalous_loader: Optional[DataLoader],
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula errores de reconstrucción para un conjunto de datos con etiquetas.

    Args:
        model: Modelo autoencoder entrenado
        normal_loader: DataLoader con ECG normales (etiqueta 0)
        anomalous_loader: DataLoader con ECG anómalos (etiqueta 1), puede ser None
        device: Dispositivo donde ejecutar el modelo

    Returns:
        Tuple[np.ndarray, np.ndarray]: (errores, etiquetas)
            - errores: Array 1D con errores de reconstrucción (MSE por muestra)
            - etiquetas: Array 1D con etiquetas (0=normal, 1=anómalo)
    """
    model.eval()
    all_errors = []
    all_labels = []

    model_device = torch.device(device)
    non_blocking = device.startswith("cuda")

    with torch.no_grad():
        # Procesar normales (etiqueta 0)
        if normal_loader is not None:
            for x_batch in normal_loader:
                x_batch = x_batch.to(model_device, non_blocking=non_blocking)
                recon = model(x_batch)
                # Calcular MSE por muestra: promedio sobre canales y tiempo
                batch_err = torch.mean((recon - x_batch) ** 2, dim=(1, 2))
                all_errors.extend(batch_err.cpu().numpy())
                all_labels.extend([0] * len(batch_err))

        # Procesar anómalos (etiqueta 1)
        if anomalous_loader is not None:
            for x_batch in anomalous_loader:
                x_batch = x_batch.to(model_device, non_blocking=non_blocking)
                recon = model(x_batch)
                # Calcular MSE por muestra: promedio sobre canales y tiempo
                batch_err = torch.mean((recon - x_batch) ** 2, dim=(1, 2))
                all_errors.extend(batch_err.cpu().numpy())
                all_labels.extend([1] * len(batch_err))

    return np.array(all_errors), np.array(all_labels)


def f2_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
    """
    Calcula el F-beta score, con beta=2 dando más peso al recall.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        beta: Parámetro beta (default=2.0 para F2-score)

    Returns:
        float: F-beta score
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if precision + recall == 0:
        return 0.0

    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)
    return float(f_beta)


def evaluate_threshold(
    val_errors: np.ndarray,
    val_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Evalúa un umbral específico y retorna métricas.

    Args:
        val_errors: Array con errores de reconstrucción
        val_labels: Array con etiquetas verdaderas (0=normal, 1=anómalo)
        threshold: Umbral a evaluar

    Returns:
        Dict con métricas: threshold, recall_anom, precision_anom, fpr_normal, f2_score
    """
    y_pred = (val_errors > threshold).astype(int)

    # Calcular matriz de confusión
    cm = confusion_matrix(val_labels, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Métricas
    recall_anom = tp / (tp + fn + 1e-8)  # Recall de clase anómala (1)
    fpr_normal = fp / (fp + tn + 1e-8)  # Tasa de falsos positivos en normales
    precision_anom = tp / (tp + fp + 1e-8)  # Precision de clase anómala
    f2 = f2_score(val_labels, y_pred, beta=2.0)

    return {
        "threshold": threshold,
        "recall_anom": recall_anom,
        "precision_anom": precision_anom,
        "fpr_normal": fpr_normal,
        "f2_score": f2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_optimal_threshold(
    val_errors: np.ndarray,
    val_labels: np.ndarray,
    percentiles: Optional[List[float]] = None,
    max_fpr: Optional[float] = 0.05,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Busca el umbral óptimo basado en percentiles del error de reconstrucción.
    
    ⚠️ IMPORTANTE: Esta función determina el umbral para clasificar ECGs como anómalos.
    
    📝 CÓMO CAMBIAR EL UMBRAL:
        - percentiles: Lista de percentiles a probar
          * Percentiles más altos (95-99) = umbral más alto = menos falsos positivos
          * Percentiles más bajos (80-90) = umbral más bajo = detecta más anomalías
        - max_fpr: FPR máximo permitido (ej: 0.05 = 5%)
          * max_fpr más bajo = umbral más alto = menos falsos positivos
          * max_fpr más alto = umbral más bajo = más detecciones
    
    🎯 CRITERIO DE SELECCIÓN:
        - Maximiza el F2-score (da más peso al recall de anomalías)
        - Filtra por max_fpr si se especifica
        - El mejor umbral se guarda en BEST_THR
    
    Args:
        val_errors: Array con errores de reconstrucción del validation set
        val_labels: Array con etiquetas verdaderas (0=normal, 1=anómalo)
        percentiles: Lista de percentiles a probar (default: [80, 85, 90, 92, 94, 96, 98, 99])
        max_fpr: FPR máximo permitido (None para no filtrar)
        verbose: Si True, imprime tabla de resultados

    Returns:
        Tuple[float, Dict, pd.DataFrame]:
            - Mejor umbral (usar como BEST_THR)
            - Diccionario con métricas del mejor umbral
            - DataFrame con resultados de todos los umbrales probados
    """
    if percentiles is None:
        percentiles = [80, 85, 90, 92, 94, 96, 98, 99]

    # Calcular umbrales candidatos basados en percentiles
    candidates = np.percentile(val_errors, percentiles)

    # Evaluar cada umbral
    results = []
    for thr in candidates:
        metrics = evaluate_threshold(val_errors, val_labels, thr)
        results.append(metrics)

    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)

    # Filtrar por FPR si se especifica
    if max_fpr is not None:
        df_filtered = df_results[df_results["fpr_normal"] <= max_fpr].copy()
        if len(df_filtered) > 0:
            df_results = df_filtered
        else:
            if verbose:
                print(f"⚠️ Ningún umbral cumple con max_fpr={max_fpr}. Usando todos los candidatos.")

    # Seleccionar mejor umbral: maximizar F2-score
    best_idx = df_results["f2_score"].idxmax()
    best_threshold = df_results.loc[best_idx, "threshold"]
    best_metrics = df_results.loc[best_idx].to_dict()

    if verbose:
        print("\n" + "=" * 80)
        print("BÚSQUEDA DE UMBRAL ÓPTIMO")
        print("=" * 80)
        print(f"\nCandidatos evaluados: {len(candidates)} umbrales")
        if max_fpr is not None:
            print(f"Filtro aplicado: FPR <= {max_fpr}")
        print("\nResultados por umbral:")
        print("-" * 80)
        display_cols = ["threshold", "recall_anom", "precision_anom", "fpr_normal", "f2_score"]
        print(df_results[display_cols].to_string(index=False, float_format="%.6f"))
        print("-" * 80)
        print(f"\n✓ Mejor umbral seleccionado: {best_threshold:.6f}")
        print(f"  - F2-score: {best_metrics['f2_score']:.6f}")
        print(f"  - Recall (anómalos): {best_metrics['recall_anom']:.6f}")
        print(f"  - Precision (anómalos): {best_metrics['precision_anom']:.6f}")
        print(f"  - FPR (normales): {best_metrics['fpr_normal']:.6f}")
        print("=" * 80)

    return best_threshold, best_metrics, df_results


def predict_with_threshold(
    model: torch.nn.Module,
    normal_loader: Optional[DataLoader],
    anomalous_loader: Optional[DataLoader],
    device: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predice etiquetas usando un umbral específico.

    Args:
        model: Modelo autoencoder entrenado
        normal_loader: DataLoader con ECG normales (etiqueta 0)
        anomalous_loader: DataLoader con ECG anómalos (etiqueta 1), puede ser None
        device: Dispositivo donde ejecutar el modelo
        threshold: Umbral para clasificación

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - all_labels: Etiquetas verdaderas
            - y_pred: Etiquetas predichas (0=normal, 1=anómalo)
            - all_errors: Errores de reconstrucción
    """
    all_errors, all_labels = compute_reconstruction_errors_with_labels(
        model, normal_loader, anomalous_loader, device
    )
    y_pred = (all_errors > threshold).astype(int)
    return all_labels, y_pred, all_errors


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calcula métricas completas de clasificación.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        y_scores: Scores continuos (opcional, para AUC)

    Returns:
        Dict con todas las métricas calculadas
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_normal": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_normal": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_anom": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_anom": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_normal": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_anom": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f2_anom": f2_score(y_true, y_pred, beta=2.0),
        "specificity": tn / (tn + fp + 1e-8),  # TNR = 1 - FPR
        "fpr": fp / (fp + tn + 1e-8),  # False Positive Rate
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Calcular AUC si se proporcionan scores
    if y_scores is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            metrics["auroc"] = roc_auc_score(y_true, y_scores)
            metrics["auprc"] = average_precision_score(y_true, y_scores)
        except Exception:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Matriz de Confusión",
    save_path: Optional[Path] = None,
    dpi: int = 140,
) -> None:
    """
    Visualiza la matriz de confusión.

    Args:
        cm: Matriz de confusión 2x2
        labels: Lista con nombres de las clases
        title: Título del gráfico
        save_path: Ruta donde guardar la imagen (opcional)
        dpi: Resolución de la imagen
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Configurar ticks y etiquetas
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[f"Pred: {lbl}" for lbl in labels],
        yticklabels=[f"Real: {lbl}" for lbl in labels],
        title=title,
        ylabel="Etiqueta Real",
        xlabel="Etiqueta Predicha",
    )

    # Agregar valores en las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Matriz de confusión guardada en: {save_path}")
    plt.close(fig)


def evaluate_test_set(
    model: torch.nn.Module,
    test_normal_loader: Optional[DataLoader],
    test_anomalous_loader: Optional[DataLoader],
    device: str,
    threshold: float,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evalúa el modelo en el conjunto de test con métricas completas.

    Args:
        model: Modelo autoencoder entrenado
        test_normal_loader: DataLoader con ECG normales de test
        test_anomalous_loader: DataLoader con ECG anómalos de test
        device: Dispositivo donde ejecutar el modelo
        threshold: Umbral para clasificación
        output_dir: Directorio donde guardar resultados (opcional)
        verbose: Si True, imprime métricas y matriz de confusión

    Returns:
        Dict con todas las métricas calculadas
    """
    # Predecir en test
    y_true_test, y_pred_test, y_scores_test = predict_with_threshold(
        model, test_normal_loader, test_anomalous_loader, device, threshold
    )

    # Calcular métricas
    metrics = compute_classification_metrics(y_true_test, y_pred_test, y_scores_test)

    # Matriz de confusión
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])

    if verbose:
        print("\n" + "=" * 80)
        print("EVALUACIÓN EN CONJUNTO DE TEST")
        print("=" * 80)
        print(f"\nUmbral utilizado: {threshold:.6f}")
        print(f"\nMuestras totales: {len(y_true_test)}")
        print(f"  - Normales (0): {(y_true_test == 0).sum()}")
        print(f"  - Anómalos (1): {(y_true_test == 1).sum()}")

        print("\n" + "-" * 80)
        print("MATRIZ DE CONFUSIÓN")
        print("-" * 80)
        print(f"\n{'':<15} {'Pred: Normal':<15} {'Pred: Anómalo':<15}")
        print(f"{'Real: Normal':<15} {cm_test[0,0]:<15} {cm_test[0,1]:<15}")
        print(f"{'Real: Anómalo':<15} {cm_test[1,0]:<15} {cm_test[1,1]:<15}")

        print("\n" + "-" * 80)
        print("MÉTRICAS DE CLASIFICACIÓN")
        print("-" * 80)
        print(f"\nMétricas generales:")
        print(f"  Accuracy:           {metrics['accuracy']:.6f}")
        print(f"  Specificity (TNR):  {metrics['specificity']:.6f}")
        print(f"  FPR:                {metrics['fpr']:.6f}")

        print(f"\nMétricas para clase NORMAL (0):")
        print(f"  Precision:          {metrics['precision_normal']:.6f}")
        print(f"  Recall:             {metrics['recall_normal']:.6f}")
        print(f"  F1-score:           {metrics['f1_normal']:.6f}")

        print(f"\nMétricas para clase ANÓMALA (1):")
        print(f"  Precision:          {metrics['precision_anom']:.6f}")
        print(f"  Recall:             {metrics['recall_anom']:.6f}")
        print(f"  F1-score:           {metrics['f1_anom']:.6f}")
        print(f"  F2-score:           {metrics['f2_anom']:.6f}")

        if "auroc" in metrics and not np.isnan(metrics["auroc"]):
            print(f"\nMétricas de ranking:")
            print(f"  AUROC:              {metrics['auroc']:.6f}")
            print(f"  AUPRC:              {metrics['auprc']:.6f}")

        print("\n" + "-" * 80)
        print("REPORTE DE CLASIFICACIÓN (sklearn)")
        print("-" * 80)
        print(classification_report(y_true_test, y_pred_test, target_names=["Normal", "Anómalo"], digits=4))
        print("=" * 80)

    # Guardar matriz de confusión si se especifica output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cm_path = output_dir / "confusion_matrix_test.png"
        plot_confusion_matrix(cm_test, ["Normal", "Anómalo"], "Matriz de Confusión - Test", cm_path)

        # Guardar métricas en CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_dir / "test_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        if verbose:
            print(f"\n✓ Métricas guardadas en: {metrics_path}")

    return metrics


def full_evaluation_pipeline(
    model: torch.nn.Module,
    val_normal_loader: Optional[DataLoader],
    val_anomalous_loader: Optional[DataLoader],
    test_normal_loader: Optional[DataLoader],
    test_anomalous_loader: Optional[DataLoader],
    device: str = "cuda",
    percentiles: Optional[List[float]] = None,
    max_fpr: Optional[float] = 0.05,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Pipeline completo de evaluación: busca umbral óptimo y evalúa en test.

    Args:
        model: Modelo autoencoder entrenado
        val_normal_loader: DataLoader con ECG normales de validación
        val_anomalous_loader: DataLoader con ECG anómalos de validación
        test_normal_loader: DataLoader con ECG normales de test
        test_anomalous_loader: DataLoader con ECG anómalos de test
        device: Dispositivo donde ejecutar el modelo
        percentiles: Lista de percentiles para búsqueda de umbral
        max_fpr: FPR máximo permitido para selección de umbral
        output_dir: Directorio donde guardar resultados
        verbose: Si True, imprime información detallada

    Returns:
        Tuple[float, Dict, Dict]:
            - Mejor umbral encontrado
            - Métricas en validación con mejor umbral
            - Métricas en test con mejor umbral
    """
    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETO DE EVALUACIÓN")
        print("=" * 80)

    # Paso 1: Calcular errores en validación
    if verbose:
        print("\n[1/3] Calculando errores de reconstrucción en validación...")
    val_errors, val_labels = compute_reconstruction_errors_with_labels(
        model, val_normal_loader, val_anomalous_loader, device
    )
    if verbose:
        print(f"  ✓ Errores calculados: {len(val_errors)} muestras")
        print(f"    - Normales: {(val_labels == 0).sum()}")
        print(f"    - Anómalos: {(val_labels == 1).sum()}")

    # Paso 2: Buscar umbral óptimo
    if verbose:
        print("\n[2/3] Buscando umbral óptimo...")
    best_threshold, best_val_metrics, df_thresholds = find_optimal_threshold(
        val_errors, val_labels, percentiles, max_fpr, verbose
    )

    # Paso 3: Evaluar en test
    if verbose:
        print("\n[3/3] Evaluando en conjunto de test...")
    test_metrics = evaluate_test_set(
        model, test_normal_loader, test_anomalous_loader, device, best_threshold, output_dir, verbose
    )

    # Guardar resultados de búsqueda de umbral si se especifica output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        thresholds_path = output_dir / "threshold_search_results.csv"
        df_thresholds.to_csv(thresholds_path, index=False)
        if verbose:
            print(f"\n✓ Resultados de búsqueda de umbral guardados en: {thresholds_path}")

    return best_threshold, best_val_metrics, test_metrics

