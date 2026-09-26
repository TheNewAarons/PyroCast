"""Orquesta: cache -> descarga de tiles faltantes -> mosaico -> reproyección
bilineal a la grilla de trabajo del proyecto (ver shared/config.py)."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ingestion.dem.cache import cache_key_for
from ingestion.dem.client import download_tile
from ingestion.dem.tiles import tile_key, tiles_for_bbox


def build_dem(
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    crs: str,
    raw_tiles_dir: Path,
    cache_dir: Path,
    download_fn: Callable[[str, Path], Path] = download_tile,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dem_{cache_key_for(bbox, resolution_m, crs)}.tif"
    if cache_path.exists():
        return cache_path

    raw_tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for lat, lon in tiles_for_bbox(bbox):
        key = tile_key(lat, lon)
        dest = raw_tiles_dir / f"{key}.tif"
        tile_paths.append(download_fn(key, dest))

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_array, mosaic_transform = merge(sources)
        src_crs = sources[0].crs
        src_nodata = sources[0].nodata
    finally:
        for src in sources:
            src.close()

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        crs,
        mosaic_array.shape[-1],
        mosaic_array.shape[-2],
        *rasterio.transform.array_bounds(
            mosaic_array.shape[-2], mosaic_array.shape[-1], mosaic_transform
        ),
        resolution=(resolution_m, resolution_m),
    )

    dst_array = np.empty((mosaic_array.shape[0], dst_height, dst_width), dtype=mosaic_array.dtype)
    reproject(
        source=mosaic_array,
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=src_nodata,
        resampling=Resampling.bilinear,
    )

    with rasterio.open(
        cache_path, "w", driver="GTiff",
        height=dst_height, width=dst_width, count=dst_array.shape[0],
        dtype=dst_array.dtype, crs=crs, transform=dst_transform, nodata=src_nodata,
    ) as dst:
        dst.write(dst_array)

    return cache_path
