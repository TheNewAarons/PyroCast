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


def _make_toggling_event(event_id: int) -> xr.DataArray:
    """Día 0: celda (0,0) activa. Día 1: (0,0) YA NO (fire_mask es
    extensión activa diaria, no acumulada -- features/fire_state/
    rasterize.py), (1,1) se enciende de nuevo. Un modelo cuya
    predicción SÍ es acumulada (como CellularAutomatonModel, que nunca
    "apaga" una celda -- ver models/cellular_automata/simulate.py)
    predice ambas celdas encendidas el día 1."""
    size = 4
    data = np.zeros((2, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    data[0, fire_idx, 0, 0] = 1.0
    data[1, fire_idx, 1, 1] = 1.0
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-01", "2026-01-02"], "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_run_backtest_compares_predicted_against_cumulative_ground_truth():
    # el modelo predice acumulado (como el autómata celular real: una
    # vez encendida, una celda nunca se "apaga" dentro del horizonte
    # simulado) -- comparar contra la verdad CRUDA (día a día, no
    # acumulada) penalizaría al modelo por una diferencia de
    # CONVENCIÓN, no de error real. run_backtest debe acumular la
    # verdad antes de comparar.
    class _CumulativeModel:
        def predict(self, event: xr.DataArray) -> np.ndarray:
            channels = list(event.coords["channel"].values)
            fire_idx = channels.index("fire_mask")
            raw = event.values[:, fire_idx, :, :].astype("float64")
            return np.logical_or.accumulate(raw >= 0.5, axis=0).astype("float64")

    events = [_make_toggling_event(event_id=1)]
    result = run_backtest(_CumulativeModel(), events, n_bootstrap=10, seed=1)
    metrics = result.per_event[0]
    assert metrics.iou == 1.0
    assert metrics.dice == 1.0


def _make_non_square_event(event_id: int) -> xr.DataArray:
    height, width, n_days = 4, 6, 2  # no cuadrado -- una transposición cambia la forma
    data = np.zeros((n_days, len(CHANNEL_ORDER), height, width), dtype="float32")
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-01", "2026-01-02"], "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_run_backtest_rejects_a_predict_result_with_the_wrong_shape():
    class _WrongShapeModel:
        def predict(self, event: xr.DataArray) -> np.ndarray:
            n_days, _, height, width = event.shape
            return np.zeros((n_days, width, height), dtype="float64")  # transpuesto

    events = [_make_non_square_event(event_id=1)]
    with pytest.raises(ValueError, match="event 1"):
        run_backtest(_WrongShapeModel(), events, n_bootstrap=10, seed=1)


def test_run_backtest_rejects_a_predict_result_outside_zero_one_range():
    class _OutOfRangeModel:
        def predict(self, event: xr.DataArray) -> np.ndarray:
            n_days, _, height, width = event.shape
            return np.full((n_days, height, width), 7.0, dtype="float64")

    events = [_make_event(event_id=1)]
    with pytest.raises(ValueError, match="event 1"):
        run_backtest(_OutOfRangeModel(), events, n_bootstrap=10, seed=1)


def test_anchor_day_is_excluded_from_metrics_by_default():
    # un modelo que solo "predice" el ancla conocido del día 0 (lo mismo que
    # CalibratedUNet/CellularAutomatonModel hacen en el día 0) y nada después no
    # debe sacar IoU > 0: el día 0 coincide con la verdad por construcción.
    class _AnchorOnly:
        def predict(self, event):
            out = np.zeros((event.sizes["day"], event.sizes["y"], event.sizes["x"]))
            out[0] = event.values[0, CHANNEL_ORDER.index("fire_mask")]
            return out

    event = _make_event(1, n_days=3, size=4, fire_frac=0.25)
    event.values[1:, CHANNEL_ORDER.index("fire_mask")] = 1.0  # el fuego crece desde el día 1
    event.values[0, CHANNEL_ORDER.index("fire_mask")] = 0.0
    event.values[0, CHANNEL_ORDER.index("fire_mask"), 0, 0] = 1.0  # ancla: 1 celda
    default = run_backtest(_AnchorOnly(), [event], n_bootstrap=1).per_event[0]
    with_anchor = run_backtest(_AnchorOnly(), [event], n_bootstrap=1,
                               exclude_anchor_day=False).per_event[0]
    assert default.iou == 0.0 and default.dice == 0.0
    assert with_anchor.iou > 0.0


def test_exclude_anchor_day_needs_at_least_two_days():
    with pytest.raises(ValueError, match="al menos 2 días"):
        run_backtest(_ConstantModel(), [_make_event(1, n_days=1)], n_bootstrap=1)
