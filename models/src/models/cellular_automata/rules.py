"""Regla de transición del autómata celular: probabilidad de que una
celda se encienda en el paso siguiente, inspirada en Rothermel
simplificado.

P(celda se enciende) = 1 - prod_{vecinos en llamas} (1 - p_dir)

donde, para cada una de las 8 direcciones de vecindad (Moore):

    p_dir = base_spread_prob
            * exp(slope_coefficient * pendiente_hacia_la_celda)
            * exp(wind_coefficient * componente_direccional_del_viento)
            * flammability[celda]
            (recortado a [0, 1])

Términos:
- **Vecinos en llamas**: el producto sobre TODOS los vecinos que están
  actualmente en llamas (los que no lo están no contribuyen) -- más
  vecinos en llamas siempre aumenta o mantiene la probabilidad, nunca la
  reduce.
- **Pendiente** (`pendiente_hacia_la_celda`): `(elevación[celda] -
  elevación[vecino]) / distancia`. Positiva cuando la celda está más
  alta que el vecino en llamas (fuego subiendo) -- aumenta la
  probabilidad ("el fuego sube más rápido cuesta arriba").
- **Viento** (`componente_direccional_del_viento`): proyección del
  vector de viento `(wind_u, wind_v)` (convención ERA5-Land: apunta
  hacia donde SOPLA el viento, no de dónde viene) sobre la dirección de
  propagación (del vecino hacia la celda). Positiva cuando el viento
  sopla en la misma dirección que la propagación -- aumenta la
  probabilidad. Sin viento, este término es 0 exactamente (nunca se
  divide por la velocidad del viento, así que no hay 0/0).
- **Tipo de combustible** (`flammability[celda]`): un multiplicador en
  [0, 1] de la celda OBJETIVO (no del vecino) -- 0 para combustibles no
  arden (agua, urbano, nieve/hielo, suelo desnudo, código desconocido).

Parámetros libres (`SpreadParameters`, todos calibrables — ver
`models/cellular_automata/calibrate.py`): `base_spread_prob`,
`slope_coefficient`, `wind_coefficient`, `fuel_flammability`. Ver
docs/cellular-automata.md para la justificación completa de cada uno y
sus valores por defecto.

*** SIMPLIFICACIÓN EXPLÍCITA: este NO es un modelo físico de Rothermel
completo (que requiere humedad de combustible, profundidad del lecho de
combustible, razón de empaquetamiento, calor de ignición, etc., ninguno
modelado aquí) — es una regla de transición probabilística de autómata
celular INSPIRADA en la intuición física de Rothermel (el fuego se
propaga más rápido cuesta arriba y a favor del viento), no una
implementación de sus ecuaciones. Ver docs/cellular-automata.md y
docs/limitations.md. ***
"""
from dataclasses import dataclass, field

import numpy as np

NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

# Códigos de tipo de combustible simplificado, duplicados desde
# ingestion/worldcover/fuel_type.py (no se importa desde ahí -- ver
# docs/decisions.md, mismo patrón que CLOUD_SCL_CLASSES). Cualquier
# código no listado aquí (incluyendo estos sentinels) vale 0.0.
DEFAULT_FUEL_FLAMMABILITY: dict[int, float] = {
    1: 1.0,   # pastizal
    2: 0.8,   # matorral
    3: 0.6,   # bosque
    4: 0.3,   # cultivo
    5: 0.1,   # humedal
    90: 0.0,  # urbano/no combustible
    91: 0.0,  # suelo desnudo/no combustible
    92: 0.0,  # agua/no combustible
    93: 0.0,  # nieve/hielo/no combustible
    99: 0.0,  # desconocido
}


@dataclass(frozen=True)
class SpreadParameters:
    base_spread_prob: float = 0.3
    slope_coefficient: float = 4.0
    wind_coefficient: float = 0.2
    fuel_flammability: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_FUEL_FLAMMABILITY)
    )


def _shifted(array: np.ndarray, d_row: int, d_col: int, fill: float) -> np.ndarray:
    """`array` desplazado (d_row, d_col): el resultado B cumple
    `B[row, col] == array[row + d_row, col + d_col]` cuando esa posición
    cae dentro de la grilla, y `fill` en caso contrario -- nunca
    wrap-around (a diferencia de `np.roll`)."""
    height, width = array.shape
    result = np.full_like(array, fill)
    row_src_start, row_src_end = max(0, d_row), height + min(0, d_row)
    row_dst_start, row_dst_end = max(0, -d_row), height + min(0, -d_row)
    col_src_start, col_src_end = max(0, d_col), width + min(0, d_col)
    col_dst_start, col_dst_end = max(0, -d_col), width + min(0, -d_col)
    result[row_dst_start:row_dst_end, col_dst_start:col_dst_end] = array[
        row_src_start:row_src_end, col_src_start:col_src_end
    ]
    return result


def flammability_grid(fuel_type: np.ndarray, fuel_flammability: dict[int, float]) -> np.ndarray:
    result = np.zeros(fuel_type.shape, dtype="float64")
    for code, value in fuel_flammability.items():
        result[fuel_type == code] = value
    return result


def compute_ignition_probability(
    burning: np.ndarray,
    elevation: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    flammability: np.ndarray,
    resolution_m: float,
    params: SpreadParameters,
) -> np.ndarray:
    not_burning_contrib = np.ones(burning.shape, dtype="float64")
    for d_row, d_col in NEIGHBOR_OFFSETS:
        neighbor_burning = _shifted(burning, d_row, d_col, False)
        neighbor_elevation = _shifted(elevation, d_row, d_col, 0.0)
        distance = resolution_m * float(np.hypot(d_row, d_col))
        slope = (elevation - neighbor_elevation) / distance
        directional_component = (wind_u * (-d_col) + wind_v * d_row) / float(
            np.hypot(d_row, d_col)
        )
        p_dir = (
            params.base_spread_prob
            * np.exp(params.slope_coefficient * slope)
            * np.exp(params.wind_coefficient * directional_component)
            * flammability
        )
        p_dir = np.clip(p_dir, 0.0, 1.0)
        contribution = np.where(neighbor_burning, p_dir, 0.0)
        not_burning_contrib = not_burning_contrib * (1.0 - contribution)
    result: np.ndarray = 1.0 - not_burning_contrib
    return result
