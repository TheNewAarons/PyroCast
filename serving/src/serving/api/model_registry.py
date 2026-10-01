"""Modelo servido por defecto.

Decisión (docs/backtest-2026.md sección 8, docs/decisions.md): el
autómata celular con parámetros por defecto. Ningún ensamble ni el U-Net
es el default -- ver esos documentos. Se carga UNA vez al iniciar la app
(lifespan), no por request.
"""
from dataclasses import dataclass

from models.cellular_automata.model import CellularAutomatonModel
from shared.model_protocol import FireSpreadModel

DEFAULT_MODEL_NAME = "cellular_automata"


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: FireSpreadModel
    # el autómata celular NO está calibrado: sus probabilidades son un
    # puntaje relativo, no una probabilidad calibrada (docs/limitations.md).
    calibrated: bool
    # canales del tensor que el modelo realmente consume: si faltan o son
    # NaN, no se puede predecir (error explícito); los demás faltantes
    # solo se reportan como advertencia.
    required_channels: tuple[str, ...]


def load_default_model(seed: int = 42) -> LoadedModel:
    return LoadedModel(
        name=DEFAULT_MODEL_NAME,
        model=CellularAutomatonModel(seed=seed),
        calibrated=False,
        required_channels=("elevation", "wind_u", "wind_v", "fuel_type"),
    )
