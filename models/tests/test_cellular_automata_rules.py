"""Tests de la regla de transición (probabilidad de ignición por celda):
término de vecinos en llamas, pendiente, viento, y tipo de combustible.
Valores esperados calculados a mano y verificados independientemente
antes de escribir este archivo (ver
docs/superpowers/plans/2026-09-27-models-cellular-automata.md)."""
import numpy as np
import pytest
from models.cellular_automata.rules import (
    DEFAULT_FUEL_FLAMMABILITY,
    SpreadParameters,
    compute_ignition_probability,
    flammability_grid,
)


def test_flammability_grid_maps_known_fuel_codes():
    fuel_type = np.array([[1, 2], [90, 99]])
    grid = flammability_grid(fuel_type, DEFAULT_FUEL_FLAMMABILITY)
    assert grid[0, 0] == 1.0   # pastizal
    assert grid[0, 1] == 0.8   # matorral
    assert grid[1, 0] == 0.0   # urbano/no combustible
    assert grid[1, 1] == 0.0   # desconocido


def test_flammability_grid_defaults_unmapped_code_to_zero():
    fuel_type = np.array([[255]])
    grid = flammability_grid(fuel_type, DEFAULT_FUEL_FLAMMABILITY)
    assert grid[0, 0] == 0.0


def test_compute_ignition_probability_is_zero_with_no_burning_neighbors():
    burning = np.zeros((3, 3), dtype=bool)
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert np.all(prob == 0.0)


def test_compute_ignition_probability_is_zero_for_non_combustible_target():
    # 4 vecinos en llamas (N,S,E,O), pero la celda objetivo no es
    # combustible (flammability=0) -- nunca debe encenderse.
    burning = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.zeros((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == 0.0


def test_compute_ignition_probability_increases_with_more_burning_neighbors():
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    one_neighbor = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    four_neighbors = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    params = SpreadParameters()
    p_one = compute_ignition_probability(
        one_neighbor, elevation, wind_u, wind_v, flammability, 100.0, params
    )[1, 1]
    p_four = compute_ignition_probability(
        four_neighbors, elevation, wind_u, wind_v, flammability, 100.0, params
    )[1, 1]
    assert p_four > p_one


def test_compute_ignition_probability_matches_hand_computed_value_uphill():
    # vecino en llamas al norte, 10 m más bajo que la celda objetivo
    # (celda objetivo cuesta ARRIBA respecto del vecino) a 100 m de
    # resolución -> slope=0.1. P = 0.3 * exp(4.0*0.1) = 0.4475474...
    burning = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    elevation = np.array([
        [0.0, 90.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.44754740929238107)


def test_compute_ignition_probability_matches_hand_computed_value_downhill():
    # mismo vecino, pero ahora la celda objetivo está 10 m más ABAJO
    # (cuesta abajo) -> slope=-0.1. P = 0.3 * exp(-0.4) = 0.2010960...
    # -- debe ser menor que el caso cuesta arriba de arriba.
    burning = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    elevation = np.array([
        [0.0, 100.0, 0.0],
        [0.0, 90.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.20109601381069178)


def test_compute_ignition_probability_matches_hand_computed_value_downwind():
    # vecino en llamas al oeste, viento soplando hacia el este a 5 m/s
    # -- alineado con la propagación oeste->centro.
    # P = 0.3 * exp(0.2*5) = 0.8154845...
    burning = np.array([
        [False, False, False],
        [True, False, False],
        [False, False, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.full((3, 3), 5.0)
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.8154845485377135)


def test_compute_ignition_probability_matches_hand_computed_value_upwind():
    # vecino en llamas al ESTE (propagación este->centro, hacia el
    # oeste), mismo viento hacia el este -- opuesto a la propagación.
    # P = 0.3 * exp(-1.0) = 0.1103638...  -- menor que el caso downwind.
    burning = np.array([
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.full((3, 3), 5.0)
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.1103638323514327)
