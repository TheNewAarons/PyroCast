"""Tests de la simulación día a día: caso analítico circular, sensibilidad
a pendiente, sensibilidad a viento, determinismo, y escala a una grilla
de tamaño realista. Todos los valores esperados fueron verificados
ejecutando el algoritmo antes de escribir este archivo (ver
docs/superpowers/plans/2026-09-27-models-cellular-automata.md)."""
import time

import numpy as np
import pytest
from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_SIZE = 41
_CENTER = _SIZE // 2


def _single_ignition() -> np.ndarray:
    initial = np.zeros((_SIZE, _SIZE), dtype=bool)
    initial[_CENTER, _CENTER] = True
    return initial


def _extents(burning: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(burning)
    north = _CENTER - int(rows.min())
    south = int(rows.max()) - _CENTER
    west = _CENTER - int(cols.min())
    east = int(cols.max()) - _CENTER
    return north, south, east, west


def test_flat_no_wind_homogeneous_fuel_spreads_approximately_circularly():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters(base_spread_prob=0.99)

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    north, south, east, west = _extents(burning)
    assert (north, south, east, west) == (10, 10, 10, 10)


def test_fire_advances_faster_uphill_than_downhill_in_the_same_number_of_steps():
    # rampa de elevación: 5 m más alto por celda hacia el norte (fila
    # decreciente) -- norte es cuesta arriba desde el punto de ignición.
    elevation = (_CENTER - np.arange(_SIZE))[:, None] * np.ones((1, _SIZE)) * 5.0
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters()  # defaults

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    north, south, _, _ = _extents(burning)
    assert north == 8   # cuesta arriba
    assert south == 5   # cuesta abajo
    assert north > south


def test_fire_elongates_in_the_wind_direction():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.full((_SIZE, _SIZE), 5.0)  # viento soplando hacia el este
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters()

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    _, _, east, west = _extents(burning)
    assert east == 10   # a favor del viento
    assert west == 2    # en contra del viento
    assert east > west


def test_simulation_is_deterministic_with_a_fixed_seed():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)

    first = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, seed=7,
    )
    second = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, seed=7,
    )
    assert np.array_equal(first, second)


def test_already_burning_cells_report_probability_one():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=3, seed=1,
    )
    assert probabilities[0, _CENTER, _CENTER] == 1.0
    assert probabilities[-1, _CENTER, _CENTER] == 1.0  # nunca se "apaga"


def test_rejects_non_finite_elevation_with_a_clear_error_instead_of_silent_nan():
    # features/dataset/resample.py rellena huecos de cobertura con NaN
    # (nunca fabrica un valor) -- si ese NaN llega hasta acá sin
    # detectarse, se propaga en silencio a través de exp()/comparaciones
    # y produce una simulación mayormente apagada sin ningún aviso.
    # Encontrado en la revisión final del 2026-09-27: una sola celda NaN
    # de elevación redujo una simulación de 55 celdas encendidas a 1.
    elevation = np.zeros((_SIZE, _SIZE))
    elevation[_CENTER, _CENTER] = np.nan
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    with pytest.raises(ValueError, match="elevation"):
        simulate_fire_spread(
            _single_ignition(), elevation, wind_u, wind_v, fuel_type,
            resolution_m=100.0, n_days=5, seed=1,
        )


def test_rejects_non_finite_wind_with_a_clear_error_instead_of_silent_nan():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_u[_CENTER, _CENTER] = np.inf
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    with pytest.raises(ValueError, match="wind_u"):
        simulate_fire_spread(
            _single_ignition(), elevation, wind_u, wind_v, fuel_type,
            resolution_m=100.0, n_days=5, seed=1,
        )


def test_scales_to_a_realistic_event_grid_size_in_a_few_seconds():
    size = 200
    center = size // 2
    initial = np.zeros((size, size), dtype=bool)
    initial[center, center] = True
    elevation = np.random.default_rng(0).random((size, size)) * 500.0
    wind_u = np.full((size, size), 3.0)
    wind_v = np.full((size, size), 1.0)
    fuel_type = np.ones((size, size), dtype=int)

    start = time.monotonic()
    simulate_fire_spread(
        initial, elevation, wind_u, wind_v, fuel_type,
        resolution_m=250.0, n_days=30, seed=42,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
