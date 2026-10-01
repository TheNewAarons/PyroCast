"""Ensamble del autómata celular (P7) y el U-Net calibrado (P11).

Ambos implementan `shared.model_protocol.FireSpreadModel`, y el
ensamble también -- así `models.evaluation.backtest.run_backtest` lo
evalúa sin cambios, sobre los mismos eventos y métricas que cada modelo
por separado.

Dos variantes:

- `BlendEnsemble`: promedio ponderado de probabilidades; el peso del
  U-Net es un hiperparámetro (`select_blend_weight` lo elige por Brier
  sobre eventos de VAL, nunca de test).
- `StackingEnsemble`: regresión logística sobre los logits de ambas
  salidas, ajustada sobre eventos de VAL.

Fuga de datos: el calibrador isotónico del U-Net ya se ajustó sobre
val (`pyrocast-calibrate run --chile-val`), así que las salidas del
U-Net sobre val son optimistas y el peso/stacking elegido ahí tiende a
favorecerlo. Es una limitación documentada en `docs/backtest-2026.md`.
"""
from collections.abc import Sequence

import numpy as np
import xarray as xr
from shared.model_protocol import FireSpreadModel
from sklearn.linear_model import LogisticRegression

_FIRE_MASK_THRESHOLD = 0.5
_EPS = 1e-6
DEFAULT_WEIGHT_GRID: tuple[float, ...] = tuple(round(w, 2) for w in np.linspace(0.0, 1.0, 11))


def _cumulative_truth(event: xr.DataArray) -> np.ndarray:
    # misma convención que evaluation/backtest.py: verdad acumulada.
    fire_idx = list(event.coords["channel"].values).index("fire_mask")
    raw = event.values[:, fire_idx] >= _FIRE_MASK_THRESHOLD
    result: np.ndarray = np.logical_or.accumulate(raw, axis=0)
    return result


def _logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, _EPS, 1.0 - _EPS)
    result: np.ndarray = np.log(clipped / (1.0 - clipped))
    return result


class BlendEnsemble:
    """`(1 - w) * CA + w * U-Net`, con `w = weight_unet` en [0, 1]."""

    def __init__(self, ca: FireSpreadModel, unet: FireSpreadModel, weight_unet: float) -> None:
        if not 0.0 <= weight_unet <= 1.0:
            raise ValueError(f"weight_unet debe estar en [0, 1], recibido {weight_unet}")
        self.ca = ca
        self.unet = unet
        self.weight_unet = weight_unet

    def predict(self, event: xr.DataArray) -> np.ndarray:
        blended: np.ndarray = (
            (1.0 - self.weight_unet) * self.ca.predict(event)
            + self.weight_unet * self.unet.predict(event)
        )
        return np.clip(blended, 0.0, 1.0)


def select_blend_weight(
    ca: FireSpreadModel,
    unet: FireSpreadModel,
    val_events: Sequence[xr.DataArray],
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> float:
    """Peso del U-Net que minimiza el Brier medio sobre `val_events`
    (días >= 1; el día 0 es el estado conocido y es igual en ambos
    modelos). Empates: gana el peso más bajo (más cerca del baseline
    físico, que no se entrena)."""
    if len(val_events) == 0:
        raise ValueError("val_events no puede estar vacío")
    scored: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for event in val_events:
        truth = _cumulative_truth(event).astype("float64")
        scored.append((ca.predict(event)[1:], unet.predict(event)[1:], truth[1:]))

    best_weight, best_brier = None, np.inf
    for weight in sorted(grid):
        brier = float(np.mean([
            np.mean(((1 - weight) * p_ca + weight * p_unet - truth) ** 2)
            for p_ca, p_unet, truth in scored
        ]))
        if brier < best_brier - 1e-12:
            best_weight, best_brier = weight, brier
    assert best_weight is not None
    return float(best_weight)


class StackingEnsemble:
    """Regresión logística sobre `[logit(p_ca), logit(p_unet)]`, ajustada
    con `fit(val_events)`. Día 0 se deja como el estado conocido."""

    def __init__(self, ca: FireSpreadModel, unet: FireSpreadModel) -> None:
        self.ca = ca
        self.unet = unet
        self._clf: LogisticRegression | None = None

    @property
    def coefficients(self) -> tuple[float, float]:
        if self._clf is None:
            raise RuntimeError("StackingEnsemble: llamar fit() primero")
        ca_coef, unet_coef = self._clf.coef_[0]
        return float(ca_coef), float(unet_coef)

    @property
    def intercept(self) -> float:
        if self._clf is None:
            raise RuntimeError("StackingEnsemble: llamar fit() primero")
        return float(self._clf.intercept_[0])

    def _features(self, event: xr.DataArray) -> np.ndarray:
        p_ca = self.ca.predict(event)[1:]
        p_unet = self.unet.predict(event)[1:]
        return np.stack([_logit(p_ca).ravel(), _logit(p_unet).ravel()], axis=1)

    def fit(self, val_events: Sequence[xr.DataArray]) -> "StackingEnsemble":
        if len(val_events) == 0:
            raise ValueError("val_events no puede estar vacío")
        x = np.concatenate([self._features(e) for e in val_events])
        y = np.concatenate([_cumulative_truth(e)[1:].ravel() for e in val_events])
        if len(np.unique(y)) < 2:
            raise ValueError("val_events debe contener las dos clases (fuego / no fuego)")
        # sin penalización fuerte: solo 3 parámetros, muchas muestras.
        self._clf = LogisticRegression(C=1e6, max_iter=1000)
        self._clf.fit(x, y)
        return self

    def predict(self, event: xr.DataArray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("StackingEnsemble: llamar fit() primero")
        fire_idx = list(event.coords["channel"].values).index("fire_mask")
        out = np.zeros((event.sizes["day"], event.sizes["y"], event.sizes["x"]), dtype="float64")
        out[0] = np.clip(event.values[0, fire_idx].astype("float64"), 0.0, 1.0)
        if event.sizes["day"] > 1:
            proba = self._clf.predict_proba(self._features(event))[:, 1]
            out[1:] = proba.reshape(out[1:].shape)
        return out
