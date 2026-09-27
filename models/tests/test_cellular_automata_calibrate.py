"""Tests del grid search de calibración. IMPORTANTE (encontrado en la
revisión final del 2026-09-27): con un `TrainingSample` de un solo paso
(ver `_score_sample`), la métrica IoU sobre el estado final es una
FUNCIÓN ESCALÓN de `base_spread_prob >= 0.5` cuando cada celda del anillo
de ignición tiene exactamente un vecino en llamas -- el grid search NO
"recupera" el valor verdadero del parámetro, solo distingue de qué lado
del umbral 0.5 cae cada candidato. Los tests reflejan esto explícitamente
en vez de nombrarse como si el grid search hiciera una recuperación
exacta que no hace."""
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


def test_grid_search_calibrate_picks_the_candidate_above_the_iou_threshold():
    # Cuando la grilla de candidatos SÍ está a ambos lados del umbral 0.5
    # (uno claramente por debajo, uno igual al valor verdadero), el
    # candidato "correcto" gana -- pero esto es un artefacto del umbral,
    # no una recuperación real del valor (ver
    # test_grid_search_calibrate_iou_score_is_a_step_function_of_the_threshold
    # más abajo, que demuestra que NO discrimina entre candidatos del
    # mismo lado). Valor de score verificado exactamente.
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
    assert best_score == pytest.approx(0.031578947368421054)


def test_grid_search_calibrate_iou_score_is_a_step_function_of_the_threshold():
    # Con un TrainingSample de un solo paso, cada celda del anillo de
    # ignición tiene exactamente UN vecino en llamas, así que
    # P(ignición)=base_spread_prob para todas ellas -- el IoU
    # umbralizado en >=0.5 da el MISMO score para cualquier candidato
    # >=0.5, sin importar cuán cerca esté del valor verdadero (0.9).
    # Documentado explícitamente en docs/limitations.md: el grid search
    # con IoU y un sample de un solo paso NO distingue 0.5 de 0.9, ambos
    # "aciertan" el umbral igual de bien.
    sample = _make_sample(true_base_spread_prob=0.9)
    best_at_threshold, score_at_threshold = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.5, 0.9],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="iou",
        seed=1,
    )
    # gana 0.5 (el primero en el grid con el mismo score que 0.9) -- NO
    # 0.9, aunque 0.9 sea el valor verdadero. Este es precisamente el
    # comportamiento que hace que "recupera el valor verdadero" sea una
    # descripción incorrecta del grid search con IoU de un solo paso.
    assert best_at_threshold.base_spread_prob == 0.5
    assert score_at_threshold == pytest.approx(0.031578947368421054)


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


def test_grid_search_calibrate_rejects_an_unknown_param_grid_key():
    # Antes del fix, una clave mal escrita (p. ej. "base_sprad_prob") se
    # ignoraba en silencio -- _build_params solo lee las tres claves
    # válidas con .get(), así que un typo hacía que TODA la búsqueda
    # evaluara los parámetros por DEFECTO N veces, devolviendo un
    # resultado que parece exitoso pero no calibró nada. Encontrado en
    # la revisión final del 2026-09-27.
    sample = _make_sample(true_base_spread_prob=0.5)
    with pytest.raises(ValueError, match="base_sprad_prob"):
        grid_search_calibrate(
            [sample],
            param_grid={"base_sprad_prob": [0.05, 0.9]},
            metric="iou",
        )
