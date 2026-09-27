"""Grid search simple para calibrar los parámetros libres de
`SpreadParameters` contra un subconjunto de eventos de entrenamiento,
optimizando IoU o Brier score (`models/evaluation/metrics.py`, P8 --
definidas ahí desde el principio, no duplicadas ni movidas después).

Solo calibra los tres parámetros escalares (`base_spread_prob`,
`slope_coefficient`, `wind_coefficient`) -- `fuel_flammability` es un
dict, no un escalar, y calibrarlo por grid search requeriría un espacio
de búsqueda combinatorio mucho más caro; queda para trabajo futuro (ver
docs/limitations.md).
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread
from models.evaluation.metrics import brier_score, iou_score

_VALID_PARAM_GRID_KEYS = frozenset({"base_spread_prob", "slope_coefficient", "wind_coefficient"})


@dataclass(frozen=True)
class TrainingSample:
    initial_burning: np.ndarray
    elevation: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    fuel_type: np.ndarray
    resolution_m: float
    observed_final_mask: np.ndarray


def _build_params(combo: dict[str, float]) -> SpreadParameters:
    # construcción explícita (en vez de `SpreadParameters(**combo)`): un
    # dict[str, float] genérico no le permite a mypy --strict verificar
    # que cada clave corresponde a un parámetro ESCALAR de
    # `SpreadParameters` (que también tiene `fuel_flammability: dict[int,
    # float]`) -- ver docs/decisions.md.
    base = SpreadParameters()
    return SpreadParameters(
        base_spread_prob=combo.get("base_spread_prob", base.base_spread_prob),
        slope_coefficient=combo.get("slope_coefficient", base.slope_coefficient),
        wind_coefficient=combo.get("wind_coefficient", base.wind_coefficient),
        fuel_flammability=base.fuel_flammability,
    )


def _score_sample(
    sample: TrainingSample, params: SpreadParameters, metric: str, seed: int
) -> float:
    # n_days=1 SIEMPRE, literal -- NO se re-deriva nada del sample. Esto
    # solo tiene sentido si `observed_final_mask` es alcanzable en un
    # único paso simulado desde `initial_burning` (cierto para los
    # samples sintéticos de este módulo, construidos así a propósito).
    # Un evento real multi-día de features/fire_state necesitaría correr
    # la trayectoria completa y comparar el ÚLTIMO día -- ver
    # docs/limitations.md. Consecuencia directa: `seed` no tiene ningún
    # efecto aquí (con n_days=1, el único array de probabilidad devuelto
    # se calcula ANTES del primer sorteo aleatorio) -- documentado, no
    # un descuido.
    n_days = 1
    probabilities = simulate_fire_spread(
        sample.initial_burning, sample.elevation, sample.wind_u, sample.wind_v,
        sample.fuel_type, sample.resolution_m, n_days=n_days, params=params, seed=seed,
    )
    final_prob = probabilities[-1]
    if metric == "iou":
        predicted_mask = final_prob >= 0.5
        return iou_score(predicted_mask, sample.observed_final_mask)
    if metric == "brier":
        return -brier_score(final_prob, sample.observed_final_mask.astype("float64"))
    raise ValueError(f"métrica desconocida: {metric!r} -- usar 'iou' o 'brier'")


def grid_search_calibrate(
    samples: list[TrainingSample],
    param_grid: dict[str, list[float]],
    metric: str = "iou",
    seed: int = 42,
) -> tuple[SpreadParameters, float]:
    unknown_keys = set(param_grid) - _VALID_PARAM_GRID_KEYS
    if unknown_keys:
        # Sin esto, una clave mal escrita (p. ej. "base_sprad_prob") se
        # ignoraba en silencio -- _build_params solo lee las tres claves
        # válidas con .get(), así que un typo hacía que TODA la búsqueda
        # evaluara los parámetros por DEFECTO N veces, devolviendo un
        # resultado que parece exitoso pero no calibró nada. Encontrado
        # en la revisión final del 2026-09-27.
        raise ValueError(
            f"param_grid tiene claves desconocidas: {sorted(unknown_keys)} -- "
            f"válidas: {sorted(_VALID_PARAM_GRID_KEYS)}"
        )
    keys = list(param_grid)
    best_params: SpreadParameters | None = None
    best_score = -math.inf
    for combo in itertools.product(*(param_grid[key] for key in keys)):
        combo_dict: dict[str, float] = dict(zip(keys, combo, strict=True))
        candidate = _build_params(combo_dict)
        total = sum(_score_sample(sample, candidate, metric, seed) for sample in samples)
        average_score = total / len(samples)
        if average_score > best_score:
            best_score = average_score
            best_params = candidate
    if best_params is None:
        raise ValueError("param_grid no produjo ninguna combinación de parámetros")
    return best_params, best_score
