"""Migración de los tensores de evento YA construidos con el NDVI incorrecto
(offset BOA duplicado, hallazgo C2 de docs/review.md).

Un `build-dataset` nuevo ya sale bien (el cálculo de NDVI está corregido). Esto
repara los Zarr locales sin volver a descargar nada: cada evento se construyó con
UN composite mensual de Sentinel-2 cacheado en `data/raw/sentinel2/`; se
identifica cuál reproduciendo EXACTAMENTE el NDVI viejo (offset -1000) y
comparándolo con el canal `ndvi` guardado -- así no se adivina el composite por
nombre ni por fecha -- y se reemplaza el canal por el NDVI correcto.
"""
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.warp import Resampling

from features.dataset.resample import resample_to_grid
from features.grid.grid import WorkGrid
from features.vegetation.ndvi import compute_and_save_vegetation

OLD_BUGGY_BOA_OFFSET = -1000.0
_MATCH_TOLERANCE = 1e-3


@dataclass(frozen=True)
class NdviMigration:
    composite: Path
    old_match_max_abs_diff: float
    new_ndvi: np.ndarray  # (y, x)


def _grid_of(tensor: xr.DataArray) -> WorkGrid:
    return WorkGrid(
        crs=str(tensor.attrs["crs"]), transform=Affine(*tensor.attrs["transform"]),
        width=int(tensor.sizes["x"]), height=int(tensor.sizes["y"]),
        resolution_m=float(tensor.attrs["resolution_m"]),
    )


def _ndvi_on_grid(
    composite: Path, grid: WorkGrid, boa_offset: float, check_range: bool,
    mask_dark: bool = True,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = compute_and_save_vegetation(
            composite, Path(tmp), grid.crs, int(grid.resolution_m),
            boa_offset=boa_offset, check_range=check_range, mask_dark=mask_dark,
        )
        return resample_to_grid(path, grid, Resampling.bilinear)


def migrate_event_ndvi(tensor: xr.DataArray, composites: list[Path]) -> NdviMigration:
    """Encuentra el composite del evento (el que reproduce el NDVI viejo) y
    devuelve el NDVI corregido en la grilla del evento. Falla ruidosamente si
    ninguno reproduce el canal guardado."""
    grid = _grid_of(tensor)
    channels = list(tensor.coords["channel"].values)
    stored = tensor.values[0, channels.index("ndvi")]
    best: tuple[float, Path] | None = None
    for composite in composites:
        with rasterio.open(composite) as src:
            if src.crs is None:
                continue
        old = _ndvi_on_grid(composite, grid, OLD_BUGGY_BOA_OFFSET, check_range=False,
                            mask_dark=False)
        both = np.isfinite(old) & np.isfinite(stored)
        if not both.any():
            continue
        diff = float(np.max(np.abs(old[both] - stored[both])))
        if best is None or diff < best[0]:
            best = (diff, composite)
    if best is None or best[0] > _MATCH_TOLERANCE:
        raise ValueError(
            f"Ningún composite reproduce el NDVI guardado del evento "
            f"{tensor.attrs.get('event_id')} (mejor diferencia máx. "
            f"{'n/a' if best is None else f'{best[0]:.4f}'}): no se migra a ciegas."
        )
    new = _ndvi_on_grid(best[1], grid, 0.0, check_range=True)
    return NdviMigration(composite=best[1], old_match_max_abs_diff=best[0], new_ndvi=new)
