"""Adapta `simulate_fire_spread` a `shared.model_protocol.FireSpreadModel`
-- el autómata celular (P7) es el primer implementador de esa interfaz
común (el futuro U-Net, P9-P11, será el segundo)."""
import numpy as np
import xarray as xr

from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_DEFAULT_PARAMS = SpreadParameters()


class CellularAutomatonModel:
    def __init__(
        self, params: SpreadParameters = _DEFAULT_PARAMS, seed: int = 42
    ) -> None:
        self.params = params
        self.seed = seed

    def predict(self, event: xr.DataArray) -> np.ndarray:
        channels = list(event.coords["channel"].values)

        def band(name: str) -> np.ndarray:
            result: np.ndarray = event.values[:, channels.index(name), :, :]
            return result

        # capas estáticas: mismo valor todos los días -- se toma el día 0.
        elevation = band("elevation")[0]
        # fuel_type se guarda como float32 en el tensor (mismo criterio
        # que ingestion/worldcover, ver docs/decisions.md) -- las tablas
        # de flammability usan códigos int.
        fuel_type = band("fuel_type")[0].astype(int)
        wind_u = band("wind_u")
        wind_v = band("wind_v")

        fire_mask = band("fire_mask")
        # día 0 es el único "ancla" real disponible -- no existe un
        # "día -1" del que sembrar. Un evento sin fuego en el día 0 (p.
        # ej. un día de padding antes de la primera detección) produce
        # initial_burning todo-False, que simulate_fire_spread ya maneja
        # sin fallar.
        initial_burning = fire_mask[0] > 0.5

        resolution_m = float(event.attrs["resolution_m"])
        n_days = int(event.sizes["day"])

        result: np.ndarray = simulate_fire_spread(
            initial_burning, elevation, wind_u, wind_v, fuel_type,
            resolution_m, n_days, params=self.params, seed=self.seed,
        )
        return result
