"""Grilla de trabajo canónica de PyroCast: un único objeto (`WorkGrid`),
derivable de forma determinista desde `shared/config.py`.

Determinismo: misma bbox (WGS84) + CRS + resolución -> siempre la misma
`WorkGrid` (mismos bounds, mismo ancho/alto). El bbox se reproyecta al CRS
de destino y luego se "snapea" hacia afuera al múltiplo de la resolución
más cercano (floor para el borde oeste/sur, ceil para el este/norte) —
esto es lo que garantiza un número entero de celdas y un origen fijo
reproducible, en vez de depender de los bounds reproyectados exactos (que
en general no caen en un múltiplo exacto de la resolución).

Estado actual (ver docs/decisions.md): los módulos de ingesta existentes
(DEM, ERA5-Land, Sentinel-2, WorldCover) todavía reproyectan cada uno de
forma independiente a partir de los bounds de su propio mosaico, NO desde
esta grilla — migrarlos queda para `features/dataset/` (sin implementar).
Este módulo hace que la grilla canónica exista y sea correcta; no fuerza
todavía su uso en los pipelines existentes.
"""
import math
from dataclasses import dataclass

import numpy as np
import rasterio.transform
import rioxarray  # noqa: F401  (registra el accessor .rio en xarray)
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shared.config import Settings


@dataclass(frozen=True)
class WorkGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution_m: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        west, south, east, north = rasterio.transform.array_bounds(
            self.height, self.width, self.transform
        )
        return (float(west), float(south), float(east), float(north))


def build_grid(
    bbox: tuple[float, float, float, float], crs: str, resolution_m: float
) -> WorkGrid:
    """bbox = (west, south, east, north) en WGS84 (EPSG:4326)."""
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox)

    snapped_west = math.floor(west / resolution_m) * resolution_m
    snapped_south = math.floor(south / resolution_m) * resolution_m
    snapped_east = math.ceil(east / resolution_m) * resolution_m
    snapped_north = math.ceil(north / resolution_m) * resolution_m

    width = round((snapped_east - snapped_west) / resolution_m)
    height = round((snapped_north - snapped_south) / resolution_m)
    transform = from_origin(snapped_west, snapped_north, resolution_m, resolution_m)

    return WorkGrid(
        crs=crs, transform=transform, width=width, height=height, resolution_m=resolution_m
    )


def build_grid_from_settings(settings: Settings) -> WorkGrid:
    return build_grid(settings.study_area_bbox, settings.crs, float(settings.spatial_resolution_m))


def grid_template(
    grid: WorkGrid, fill_value: float = 0.0, dtype: str = "float32"
) -> xr.DataArray:
    """DataArray con las coords x/y y CRS/transform de `grid`, relleno con
    `fill_value` — template reutilizable (p. ej.
    `xr.full_like(grid_template(grid), otro_valor)`)."""
    xs = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    ys = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    data = np.full((grid.height, grid.width), fill_value, dtype=dtype)
    da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"), name="grid_template")
    da = da.rio.write_crs(grid.crs)
    # anotación explícita: rioxarray no está completamente tipado, y sin
    # esto mypy --strict ve `Any` devuelto por `.rio.write_transform` y
    # se queja de "Returning Any from function declared to return DataArray".
    result: xr.DataArray = da.rio.write_transform(grid.transform)
    return result
