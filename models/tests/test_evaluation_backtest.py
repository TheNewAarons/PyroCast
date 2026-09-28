"""Tests del backtest: modelo dummy de fixture (predicción constante),
verifica que el pipeline entero corre y calcula métricas por evento y
agregadas con bootstrap."""
import numpy as np
import pytest
import xarray as xr
from models.evaluation.backtest import BacktestResult, run_backtest

CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


class _ConstantModel:
    """Predicción constante de 0.5 en todo el evento -- fixture simple,
    no necesita conocer nada del contenido real del tensor."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def predict(self, event: xr.DataArray) -> np.ndarray:
        n_days, _, height, width = event.shape
        return np.full((n_days, height, width), self.value, dtype="float64")


def _make_event(
    event_id: int, n_days: int = 3, size: int = 4, fire_frac: float = 0.5
) -> xr.DataArray:
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    n_fire_cells = int(size * size * fire_frac)
    flat = data[:, fire_idx, :, :].reshape(n_days, -1)
    flat[:, :n_fire_cells] = 1.0
    data[:, fire_idx, :, :] = flat.reshape(n_days, size, size)
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_run_backtest_computes_all_four_metrics_per_event():
    events = [_make_event(event_id=1), _make_event(event_id=2)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=1)
    assert isinstance(result, BacktestResult)
    assert len(result.per_event) == 2
    for metrics in result.per_event:
        assert metrics.event_id in (1, 2)
        for value in (metrics.iou, metrics.dice, metrics.brier, metrics.ece):
            assert 0.0 <= value <= 1.0


def test_run_backtest_aggregate_has_a_bootstrap_ci_per_metric():
    events = [_make_event(event_id=i) for i in range(1, 6)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=200, seed=1)
    assert set(result.aggregate) == {"iou", "dice", "brier", "ece"}
    for ci in result.aggregate.values():
        assert ci.lower <= ci.point_estimate <= ci.upper


def test_run_backtest_is_deterministic_with_a_fixed_seed():
    events = [_make_event(event_id=i) for i in range(1, 4)]
    first = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=3)
    second = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=3)
    assert first == second


def test_run_backtest_single_event_bootstrap_ci_does_not_crash():
    events = [_make_event(event_id=1)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=50, seed=1)
    for ci in result.aggregate.values():
        assert ci.lower == ci.point_estimate == ci.upper


def test_run_backtest_perfect_model_scores_perfectly():
    class _PerfectModel:
        def predict(self, event: xr.DataArray) -> np.ndarray:
            channels = list(event.coords["channel"].values)
            fire_idx = channels.index("fire_mask")
            result: np.ndarray = event.values[:, fire_idx, :, :].astype("float64")
            return result

    events = [_make_event(event_id=1, fire_frac=0.3)]
    result = run_backtest(_PerfectModel(), events, n_bootstrap=10, seed=1)
    metrics = result.per_event[0]
    assert metrics.iou == 1.0
    assert metrics.dice == 1.0
    assert metrics.brier == 0.0
    assert metrics.ece == pytest.approx(0.0)
