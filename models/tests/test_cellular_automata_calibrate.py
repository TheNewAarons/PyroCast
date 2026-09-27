"""Tests del grid search de calibración: sobre datos sintéticos donde el
parámetro "verdadero" se conoce, el grid search debe encontrarlo."""
import numpy as np
import pytest
from models.cellular_automata.calibrate import TrainingSample, grid_search_calibrate
from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_SIZE = 21
_CENTER = _SIZE // 2


def _make_sample(true_base_spread_prob: float) -> TrainingSample:
    initial = np.zeros((_SIZE, _SIZE), dtype=bool)
    initial[_CENTER, _CENTER] = True
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    true_params = SpreadParameters(base_spread_prob=true_base_spread_prob)
    observed = simulate_fire_spread(
        initial, elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=8, params=true_params, seed=1,
    )
    return TrainingSample(
        initial_burning=initial, elevation=elevation, wind_u=wind_u, wind_v=wind_v,
        fuel_type=fuel_type, resolution_m=100.0,
        observed_final_mask=observed[-1] >= 0.5,
    )


def test_grid_search_calibrate_recovers_the_true_base_spread_prob_with_iou():
    sample = _make_sample(true_base_spread_prob=0.9)
    best_params, best_score = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.05, 0.9],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="iou",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.9
    assert 0.0 <= best_score <= 1.0


def test_grid_search_calibrate_works_with_brier_metric():
    sample = _make_sample(true_base_spread_prob=0.9)
    best_params, _ = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.05, 0.9],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="brier",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.9


def test_grid_search_calibrate_handles_a_single_candidate_without_crashing():
    sample = _make_sample(true_base_spread_prob=0.5)
    best_params, best_score = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.5],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="iou",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.5
    assert best_score >= 0.0


def test_grid_search_calibrate_rejects_an_unknown_metric():
    sample = _make_sample(true_base_spread_prob=0.5)

    with pytest.raises(ValueError, match="métrica"):
        grid_search_calibrate(
            [sample],
            param_grid={"base_spread_prob": [0.5]},
            metric="not-a-real-metric",
        )
