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


def dice_score(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Coeficiente de Dice (2·|A∩B| / (|A|+|B|)) entre dos máscaras
    binarias -- ambas vacías -> 1.0 (misma convención que iou_score, no
    una división por cero). Hace `.astype(bool)` internamente: CUALQUIER
    valor no-cero cuenta como True, incluida una probabilidad cruda sin
    umbralizar -- ver test_dice_score_treats_any_nonzero_float_as_true."""
    predicted = predicted_mask.astype(bool)
    true = true_mask.astype(bool)
    total = predicted.sum() + true.sum()
    if total == 0:
        return 1.0
    intersection = np.logical_and(predicted, true).sum()
    return float(2 * intersection) / float(total)


def ece_score(predicted_prob: np.ndarray, true_binary: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: agrupa las predicciones en `n_bins`
    bins de ancho igual sobre [0,1], y para cada bin no vacío suma
    |confianza_promedio - exactitud_promedio| ponderado por la fracción
    de muestras en ese bin. 0.0 = perfectamente calibrado. Array vacío
    -> 0.0 (sin evidencia de descalibración, no un error)."""
    predicted = predicted_prob.astype("float64").ravel()
    true = true_binary.astype("float64").ravel()
    n = predicted.size
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True: un valor IGUAL a un borde interno cae en el bin de
    # ABAJO (verificado con el caso conocido de 2 bins -- ver
    # docs/decisions.md). clip por seguridad ante errores de punto
    # flotante en el borde superior.
    bin_indices = np.clip(np.digitize(predicted, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(predicted[in_bin].mean())
        accuracy = float(true[in_bin].mean())
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)
