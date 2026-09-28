"""Corre un `FireSpreadModel` (shared.model_protocol) contra un conjunto
de eventos de test, calcula las 4 métricas de `metrics.py` por evento, y
agrega cada una con un intervalo de confianza bootstrap (método
percentil, remuestreo de los VALORES por evento, nunca de píxeles --
mismo criterio de "nunca por debajo del nivel de evento" que
`features/dataset/split.py`)."""
from dataclasses import dataclass

import numpy as np
import xarray as xr
from shared.model_protocol import FireSpreadModel

from models.evaluation.metrics import brier_score, dice_score, ece_score, iou_score

_FIRE_MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class EventMetrics:
    event_id: int
    iou: float
    dice: float
    brier: float
    ece: float


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class BacktestResult:
    per_event: list[EventMetrics]
    aggregate: dict[str, BootstrapCI]


def _bootstrap_ci(
    values: list[float], n_bootstrap: int, seed: int, confidence: float
) -> BootstrapCI:
    array = np.array(values, dtype="float64")
    if array.size == 0:
        return BootstrapCI(point_estimate=float("nan"), lower=float("nan"), upper=float("nan"))
    point = float(array.mean())
    if array.size == 1:
        return BootstrapCI(point_estimate=point, lower=point, upper=point)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    resampled_means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(resampled_means, alpha))
    upper = float(np.quantile(resampled_means, 1 - alpha))
    return BootstrapCI(point_estimate=point, lower=lower, upper=upper)


def run_backtest(
    model: FireSpreadModel,
    events: list[xr.DataArray],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ece_bins: int = 10,
    confidence: float = 0.95,
) -> BacktestResult:
    per_event: list[EventMetrics] = []
    for event in events:
        predicted_prob = model.predict(event)
        channels = list(event.coords["channel"].values)
        fire_idx = channels.index("fire_mask")
        true_prob = event.values[:, fire_idx, :, :].astype("float64")
        true_mask = true_prob >= _FIRE_MASK_THRESHOLD
        predicted_mask = predicted_prob >= _FIRE_MASK_THRESHOLD

        per_event.append(
            EventMetrics(
                event_id=int(event.attrs["event_id"]),
                iou=iou_score(predicted_mask, true_mask),
                dice=dice_score(predicted_mask, true_mask),
                brier=brier_score(predicted_prob, true_prob),
                ece=ece_score(predicted_prob, true_prob, n_bins=ece_bins),
            )
        )

    aggregate = {
        name: _bootstrap_ci(
            [getattr(m, name) for m in per_event], n_bootstrap, seed, confidence
        )
        for name in ("iou", "dice", "brier", "ece")
    }
    return BacktestResult(per_event=per_event, aggregate=aggregate)
