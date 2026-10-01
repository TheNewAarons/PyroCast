"""Tests del ensamble CA + U-Net: modelos fixture determinísticos, sin
torch ni checkpoints reales."""
import numpy as np
import pytest
import xarray as xr
from models.deep.ensemble import (
    BlendEnsemble,
    StackingEnsemble,
    select_blend_weight,
)
from shared.model_protocol import FireSpreadModel

CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, event: xr.DataArray) -> np.ndarray:
        n_days, _, height, width = event.shape
        return np.full((n_days, height, width), self.value, dtype="float64")


class _OracleModel:
    """Predice la verdad acumulada con ruido de signo fijo -- para
    verificar que el stacking aprende a confiar en el modelo informativo."""

    def __init__(self, hi: float, lo: float) -> None:
        self.hi, self.lo = hi, lo

    def predict(self, event: xr.DataArray) -> np.ndarray:
        fire_idx = list(event.coords["channel"].values).index("fire_mask")
        truth = np.logical_or.accumulate(event.values[:, fire_idx] >= 0.5, axis=0)
        return np.where(truth, self.hi, self.lo).astype("float64")


def _make_event(event_id: int, n_days: int = 3, size: int = 6) -> xr.DataArray:
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    data[:, fire_idx, : size // 2, :] = 1.0  # mitad superior ardiendo
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={"day": list(range(n_days)), "channel": list(CHANNEL_ORDER)},
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_blend_is_weighted_average_of_both_models():
    ensemble = BlendEnsemble(_ConstantModel(0.2), _ConstantModel(0.8), weight_unet=0.25)
    out = ensemble.predict(_make_event(1))
    assert out.shape == (3, 6, 6)
    np.testing.assert_allclose(out, 0.75 * 0.2 + 0.25 * 0.8)


@pytest.mark.parametrize("weight, expected", [(0.0, 0.2), (1.0, 0.8)])
def test_blend_extreme_weights_reduce_to_single_model(weight, expected):
    ensemble = BlendEnsemble(_ConstantModel(0.2), _ConstantModel(0.8), weight_unet=weight)
    np.testing.assert_allclose(ensemble.predict(_make_event(1)), expected)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_blend_rejects_weight_outside_unit_interval(weight):
    with pytest.raises(ValueError, match="weight_unet"):
        BlendEnsemble(_ConstantModel(0.2), _ConstantModel(0.8), weight_unet=weight)


def test_blend_satisfies_fire_spread_model_protocol():
    assert isinstance(BlendEnsemble(_ConstantModel(0.1), _ConstantModel(0.9), 0.5), FireSpreadModel)


def test_select_blend_weight_picks_best_model_by_val_brier():
    events = [_make_event(1), _make_event(2)]
    good, bad = _OracleModel(0.95, 0.05), _ConstantModel(0.5)
    assert select_blend_weight(ca=bad, unet=good, val_events=events) == 1.0
    assert select_blend_weight(ca=good, unet=bad, val_events=events) == 0.0


def test_select_blend_weight_returns_value_from_grid():
    events = [_make_event(1)]
    weight = select_blend_weight(
        _ConstantModel(0.3), _ConstantModel(0.9), events, grid=(0.0, 0.5, 1.0)
    )
    assert weight in (0.0, 0.5, 1.0)


def test_select_blend_weight_rejects_empty_val_events():
    with pytest.raises(ValueError, match="val_events"):
        select_blend_weight(_ConstantModel(0.3), _ConstantModel(0.9), [])


def test_stacking_requires_fit_before_predict():
    ensemble = StackingEnsemble(_ConstantModel(0.3), _ConstantModel(0.9))
    with pytest.raises(RuntimeError, match="fit"):
        ensemble.predict(_make_event(1))


def test_stacking_learns_to_trust_informative_model_and_outputs_probabilities():
    informative, noise = _OracleModel(0.9, 0.1), _ConstantModel(0.5)
    ensemble = StackingEnsemble(ca=noise, unet=informative)
    ensemble.fit([_make_event(1), _make_event(2)])
    event = _make_event(3)
    out = ensemble.predict(event)
    assert out.shape == (3, 6, 6)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # coeficiente sobre el modelo informativo positivo, sobre el ruido ~0
    ca_coef, unet_coef = ensemble.coefficients
    assert unet_coef > 0.5
    assert abs(ca_coef) < 1e-6
    # día >= 1: ardiendo > no ardiendo
    assert out[1, 0, 0] > 0.5 > out[1, -1, 0]


def test_stacking_day_zero_keeps_known_state():
    ensemble = StackingEnsemble(_OracleModel(0.9, 0.1), _ConstantModel(0.5))
    ensemble.fit([_make_event(1)])
    event = _make_event(2)
    out = ensemble.predict(event)
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    np.testing.assert_allclose(out[0], event.values[0, fire_idx])


def test_stacking_fit_rejects_single_class_val():
    ensemble = StackingEnsemble(_ConstantModel(0.3), _ConstantModel(0.9))
    empty = _make_event(1)
    empty.values[:, CHANNEL_ORDER.index("fire_mask")] = 0.0
    with pytest.raises(ValueError, match="clases"):
        ensemble.fit([empty])
