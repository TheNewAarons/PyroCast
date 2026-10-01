"""Carga de eventos del dataset de Chile, igual para entrenamiento,
calibración y evaluación (un solo lugar: hallazgo H2 de docs/review.md)."""
import json
from pathlib import Path

import numpy as np
import xarray as xr


def trim_to_first_fire_day(event: xr.DataArray) -> xr.DataArray:
    # padded_days_for_event (features/dataset/pipeline.py) antepone
    # DEFAULT_PRE_EVENT_PADDING_DAYS días SIN fuego antes del primer día
    # real del evento -- el día 0 del tensor NO tiene ninguna celda en
    # llamas. Pero tanto CalibratedUNet.predict como
    # CellularAutomatonModel.predict asumen que el día 0 es el ancla
    # conocida (el estado ACTUAL del fuego del que hay que propagar):
    # sin recortar el padding, ambos modelos parten de "nada ardiendo" y
    # miden si el modelo puede ADIVINAR una ignición sin ninguna
    # información, no si puede propagar un fuego ya iniciado -- un
    # desajuste del protocolo de evaluación, no una limitación de
    # ningún modelo (verificado contra event_2582836092 de
    # docs/backtest-2026.md: 5 días de padding, IoU=Dice=0.0 para AMBOS
    # modelos). Se recorta al primer día con >=1 celda en llamas.
    channels = list(event.coords["channel"].values)
    fire_idx = channels.index("fire_mask")
    fire_by_day = event.values[:, fire_idx].reshape(event.sizes["day"], -1).any(axis=1)
    if not fire_by_day.any():
        return event
    first_fire_day = int(np.argmax(fire_by_day))
    return event.isel(day=slice(first_fire_day, None))


def load_split_events(dataset_dir: Path, split: str) -> list[xr.DataArray]:
    """Lee `splits.json` + los eventos Zarr de un split ("train", "val",
    "test"), por la misma convención de rutas que `features/dataset/` ya
    establece -- recortando el padding previo sin fuego de cada evento
    (ver `trim_to_first_fire_day`)."""
    splits = json.loads((dataset_dir / "splits.json").read_text())
    events = []
    for event_id in splits[split]:
        zarr_path = dataset_dir / f"event_{event_id:04d}.zarr"
        opened = xr.open_zarr(zarr_path)["fire_event_tensor"]
        events.append(trim_to_first_fire_day(opened))
    return events


def load_test_events(dataset_dir: Path) -> list[xr.DataArray]:
    return load_split_events(dataset_dir, "test")
