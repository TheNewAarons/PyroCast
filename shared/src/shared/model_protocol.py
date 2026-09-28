"""Interfaz común de modelo de propagación de incendios -- implementada
por igual por `models/cellular_automata` (P7) y el futuro U-Net
(P9-P11), para que `models/evaluation/backtest.py` y el reporte final
(P16) no dupliquen lógica por modelo.

Vive en `shared/` (no en `models/`) para que cualquier paquete pueda
tipar contra ella sin depender de `models/` -- p. ej. un futuro
`serving/` que sirva predicciones no necesitaría importar todo
`models/cellular_automata` solo por el tipo.
"""
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr


@runtime_checkable
class FireSpreadModel(Protocol):
    def predict(self, event: xr.DataArray) -> np.ndarray:
        """`event`: tensor `(day, channel, y, x)` tal como lo produce
        `features.dataset.assemble.assemble_event_tensor` (mismo orden
        de canales, `CHANNEL_ORDER`, mismos atributos `crs`/`transform`/
        `resolution_m`/`event_id`). Devuelve un array `(day, y, x)` de
        probabilidad de fuego por celda y por día, en `[0, 1]` -- mismas
        dimensiones espaciales y mismo número de días que `event`."""
        ...
