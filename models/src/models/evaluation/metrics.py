"""Métricas de evaluación (IoU, Brier score) -- definidas aquí
directamente en su ubicación final de P8 (`models/evaluation/`), en vez
de en `models/cellular_automata/` y movidas después. `models/
cellular_automata/calibrate.py` las importa de aquí; cuando P8 agregue
backtesting real, reutiliza estas mismas funciones sin duplicar código.
"""
import numpy as np


def iou_score(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection over Union entre dos máscaras binarias. Ambas vacías
    -> 1.0 (coincidencia perfecta trivial, no una división por cero)."""
    predicted = predicted_mask.astype(bool)
    true = true_mask.astype(bool)
    union = np.logical_or(predicted, true).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(predicted, true).sum()
    return float(intersection) / float(union)


def brier_score(predicted_prob: np.ndarray, true_binary: np.ndarray) -> float:
    """Error cuadrático medio entre probabilidades predichas ([0,1]) y el
    resultado binario real (0/1) -- menor es mejor, 0.0 es perfecto."""
    predicted = predicted_prob.astype("float64")
    true = true_binary.astype("float64")
    return float(np.mean((predicted - true) ** 2))
