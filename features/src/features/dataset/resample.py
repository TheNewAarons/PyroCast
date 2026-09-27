"""Resampleo genérico de un raster de una sola banda a una `WorkGrid`
arbitraria -- usado para cada canal del tensor de evento (terreno, clima,
vegetación, tipo de combustible), sea cual sea su CRS/resolución/bounds
de origen. Reutiliza el mismo patrón de reproyección (nodata explícito
end-to-end) que `ingestion/dem`, `ingestion/worldcover` y
`features/weather/derive.py` ya usan cada uno por su cuenta -- este
módulo es lo que finalmente pone `features/grid/` a trabajar: en vez de
migrar cada pipeline de ingesta a la grilla canónica (todavía sin hacer,
ver docs/decisions.md), se resamplea la SALIDA ya procesada de cada uno
sobre la `WorkGrid` del evento aquí."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from features.grid.grid import WorkGrid


def resample_to_grid(
    source_path: Path, grid: WorkGrid, resampling: Resampling = Resampling.bilinear
) -> np.ndarray:
    with rasterio.open(source_path) as src:
        src_nodata = src.nodata
        dst_nodata = src_nodata if src_nodata is not None else float("nan")
        dst_array = np.full((grid.height, grid.width), dst_nodata, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    if dst_nodata is not None and not (isinstance(dst_nodata, float) and np.isnan(dst_nodata)):
        return np.where(dst_array == dst_nodata, np.nan, dst_array)
    return dst_array
