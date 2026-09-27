"""Ensamblado del tensor espaciotemporal (día, canal, alto, ancho) de un
evento de incendio, y su persistencia como Zarr.

Orden de canales FIJO (`CHANNEL_ORDER`) -- el mismo para todo evento,
todo el tiempo: estáticos (elevación, pendiente, orientación, tipo de
combustible) se repiten idénticos en cada día del tensor; dinámicos
(viento u/v, temperatura, humedad relativa, precipitación, NDVI, máscara
de fuego) varían día a día. Esto es una decisión deliberada de forma: un
solo array 4D uniforme, no un dict de arrays de dimensión mixta -- ver
docs/dataset-card.md.
"""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

CHANNEL_ORDER: tuple[str, ...] = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


@dataclass(frozen=True)
class EventChannels:
    days: tuple[dt.date, ...]
    static: dict[str, np.ndarray]
    dynamic: dict[str, dict[dt.date, np.ndarray]]


def assemble_event_tensor(channels: EventChannels) -> xr.DataArray:
    n_days = len(channels.days)
    height, width = next(iter(channels.static.values())).shape
    data = np.empty((n_days, len(CHANNEL_ORDER), height, width), dtype="float32")
    for c_idx, name in enumerate(CHANNEL_ORDER):
        if name in channels.static:
            data[:, c_idx, :, :] = channels.static[name][np.newaxis, :, :]
        else:
            for d_idx, day in enumerate(channels.days):
                data[d_idx, c_idx, :, :] = channels.dynamic[name][day]
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={
            "day": [day.isoformat() for day in channels.days],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
    )


def save_event_to_zarr(tensor: xr.DataArray, output_dir: Path, event_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"event_{event_id:04d}.zarr"
    tensor.to_dataset().to_zarr(path, mode="w")
    return path
