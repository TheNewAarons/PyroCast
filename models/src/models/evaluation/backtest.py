"""Corre un `FireSpreadModel` (shared.model_protocol) contra un conjunto
de eventos de test, calcula las 4 métricas de `metrics.py` por evento, y
agrega cada una con un intervalo de confianza bootstrap (método
percentil, remuestreo de los VALORES por evento, nunca de píxeles --
mismo criterio de "nunca por debajo del nivel de evento" que
`features/dataset/split.py`).

Convención de comparación -- ACUMULADA, no día a día cruda: el canal
`fire_mask` del tensor es la extensión ACTIVA de fuego por día (no
acumulada -- una celda que ardió ayer y no hoy vuelve a `False`, ver
`features/fire_state/rasterize.py`), pero `CellularAutomatonModel` (y
cualquier autómata celular sin modelo de extinción, ver
`docs/cellular-automata.md`) predice el estado ACUMULADO ("¿esta celda
se encendió en algún día hasta hoy?", monótono no decreciente -- ver
`models/cellular_automata/simulate.py`). Comparar la predicción
acumulada contra la verdad cruda penalizaría al modelo por una
diferencia de CONVENCIÓN (fuego que se apagó según FIRMS) y no por un
error real de propagación -- encontrado en la revisión final del
2026-09-28. `run_backtest` acumula la verdad con
`np.logical_or.accumulate` antes de comparar, así ambas series miden
"¿ha ardido esta celda alguna vez hasta el día d?". Ver
`docs/decisions.md` y `docs/limitations.md`: esto NO corrige el hecho
de que el autómata celular no modela extinción -- solo evita medir esa
simplificación ya documentada como si fuera un error adicional."""
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


def _validate_prediction(predicted_prob: np.ndarray, event: xr.DataArray, event_id: int) -> None:
    # el Protocol (shared.model_protocol.FireSpreadModel) promete forma
    # (day, y, x) y valores en [0, 1] -- verificarlo acá da un error
    # legible (nombra el evento y las formas) en vez de un IndexError
    # opaco de numpy más abajo, o (peor) un número silenciosamente sin
    # sentido si el modelo devuelve logits en vez de probabilidades.
    expected_shape = (event.sizes["day"], event.sizes["y"], event.sizes["x"])
    if predicted_prob.shape != expected_shape:
        raise ValueError(
            f"predict() del modelo devolvió forma {predicted_prob.shape} para "
            f"event {event_id}, se esperaba {expected_shape} (day, y, x) -- "
            f"ver shared.model_protocol.FireSpreadModel."
        )
    if predicted_prob.size > 0 and (
        float(predicted_prob.min()) < 0.0 or float(predicted_prob.max()) > 1.0
    ):
        raise ValueError(
            f"predict() del modelo devolvió valores fuera de [0, 1] para "
            f"event {event_id} (min={float(predicted_prob.min())}, "
            f"max={float(predicted_prob.max())}) -- ver "
            f"shared.model_protocol.FireSpreadModel."
        )


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
        event_id = int(event.attrs["event_id"])
        predicted_prob = model.predict(event)
        _validate_prediction(predicted_prob, event, event_id)

        channels = list(event.coords["channel"].values)
        fire_idx = channels.index("fire_mask")
        true_prob_raw = event.values[:, fire_idx, :, :].astype("float64")
        # acumulada, no cruda -- ver docstring del módulo.
        true_mask = np.logical_or.accumulate(true_prob_raw >= _FIRE_MASK_THRESHOLD, axis=0)
        true_prob = true_mask.astype("float64")
        predicted_mask = predicted_prob >= _FIRE_MASK_THRESHOLD

        per_event.append(
            EventMetrics(
                event_id=event_id,
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
