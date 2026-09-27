"""Simulación día a día del autómata celular: vectorizada con numpy
(ningún loop por celda en Python puro — el único loop en Python es sobre
DÍAS, que el propio enunciado pide simular uno a uno).

Estado: `burning` es MONÓTONO CRECIENTE (una celda que se enciende nunca
se "apaga" dentro del horizonte simulado — ver la simplificación
explícita de no-burnout en rules.py y docs/cellular-automata.md).

Componente probabilístico: cada día se calcula la probabilidad de
ignición de cada celda (determinista dado el estado actual) y luego se
decide qué celdas se encienden REALMENTE ese día mediante un sorteo
Bernoulli vectorizado con un `numpy.random.Generator` sembrado con
`seed` -- misma semilla + mismas entradas -> misma trayectoria completa,
siempre.
"""
import numpy as np

from models.cellular_automata.rules import (
    SpreadParameters,
    compute_ignition_probability,
    flammability_grid,
)

_DEFAULT_PARAMS = SpreadParameters()


def simulate_fire_spread(
    initial_burning: np.ndarray,
    elevation: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    fuel_type: np.ndarray,
    resolution_m: float,
    n_days: int,
    params: SpreadParameters = _DEFAULT_PARAMS,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flammability = flammability_grid(fuel_type, params.fuel_flammability)
    burning = initial_burning.astype(bool).copy()
    probabilities = np.zeros((n_days, *initial_burning.shape), dtype="float64")

    for day in range(n_days):
        wind_u_day = wind_u[day] if wind_u.ndim == 3 else wind_u
        wind_v_day = wind_v[day] if wind_v.ndim == 3 else wind_v
        prob = compute_ignition_probability(
            burning, elevation, wind_u_day, wind_v_day, flammability, resolution_m, params
        )
        # celdas ya en llamas: estado conocido, probabilidad 1.0 (no se
        # vuelve a sortear si ya estaban encendidas).
        prob = np.where(burning, 1.0, prob)
        probabilities[day] = prob

        draws = rng.random(prob.shape)
        newly_ignited = (draws < prob) & ~burning
        burning = burning | newly_ignited

    return probabilities
