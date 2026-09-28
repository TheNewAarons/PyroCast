"""Tests del adaptador que expone simulate_fire_spread como
FireSpreadModel -- construye un evento sintético (mismo formato que
features.dataset.assemble.assemble_event_tensor produce, verificado
contra el objeto real antes de escribir este archivo) y verifica que la
predicción tiene la forma correcta y usa las capas correctas."""
import numpy as np
import xarray as xr
from models.cellular_automata.model import CellularAutomatonModel
from models.cellular_automata.rules import SpreadParameters
from shared.model_protocol import FireSpreadModel

CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


def _make_event(
    size: int = 21, n_days: int = 5, ignite_center: bool = True
) -> xr.DataArray:
    center = size // 2
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fuel_idx = CHANNEL_ORDER.index("fuel_type")
    data[:, fuel_idx, :, :] = 1.0  # pastizal, combustible, homogéneo
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    if ignite_center:
        data[0, fire_idx, center, center] = 1.0
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"crs": "EPSG:32719", "transform": (100.0, 0.0, 0.0, 0.0, -100.0, 0.0),
               "resolution_m": 100.0, "event_id": 42},
    )


def test_cellular_automaton_model_satisfies_the_protocol():
    assert isinstance(CellularAutomatonModel(), FireSpreadModel)


def test_predict_returns_expected_shape():
    event = _make_event(size=21, n_days=5)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    assert result.shape == (5, 21, 21)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_predict_seeds_initial_burning_from_day_zero_fire_mask():
    event = _make_event(size=21, n_days=3)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    center = 21 // 2
    assert result[0, center, center] == 1.0  # ya en llamas en el día 0


def test_predict_handles_an_event_with_no_fire_on_day_zero():
    # capas de padding antes de la primera detección real -- day-0
    # fire_mask enteramente False, no debe fallar.
    event = _make_event(size=11, n_days=3, ignite_center=False)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    assert result.shape == (3, 11, 11)
    assert np.all(result == 0.0)  # sin ignición y sin vecinos en llamas -> nunca se enciende


def test_predict_is_deterministic_with_a_fixed_seed():
    event = _make_event(size=15, n_days=4)
    first = CellularAutomatonModel(seed=7).predict(event)
    second = CellularAutomatonModel(seed=7).predict(event)
    assert np.array_equal(first, second)


def test_predict_uses_the_configured_spread_parameters():
    event = _make_event(size=21, n_days=6)
    aggressive = CellularAutomatonModel(
        params=SpreadParameters(base_spread_prob=0.99), seed=1
    ).predict(event)
    timid = CellularAutomatonModel(
        params=SpreadParameters(base_spread_prob=0.01), seed=1
    ).predict(event)
    assert np.sum(aggressive[-1] >= 0.5) > np.sum(timid[-1] >= 0.5)
