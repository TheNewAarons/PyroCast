"""Orquesta: cache -> descarga de tiles WorldCover faltantes -> mosaico ->
reproyección con remuestreo NEAREST (categórico — nunca bilineal)."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ingestion.worldcover.cache import cache_key_for
from ingestion.worldcover.client import WorldCoverDownloadError, download_tile
from ingestion.worldcover.tiles import tile_key, tiles_for_bbox

_DEFAULT_NODATA = 0.0  # WorldCover ya declara nodata=0 en sus tiles reales;
# se fuerza explícitamente igual que en ingestion/dem, por si un tile
# individual (o una versión futura) no lo trajera.


def build_worldcover(
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    crs: str,
    raw_tiles_dir: Path,
    cache_dir: Path,
    version: str = "v200",
    year: str = "2021",
    download_fn: Callable[[str, Path], Path] = download_tile,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"worldcover_{cache_key_for(bbox, resolution_m, crs, version)}.tif"
    if cache_path.exists():
        return cache_path

    raw_tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    requested_tiles = tiles_for_bbox(bbox)
    for lat, lon in requested_tiles:
        key = tile_key(lat, lon, version=version, year=year)
        dest = raw_tiles_dir / f"{key}.tif"
        try:
            tile_paths.append(download_fn(key, dest))
        except WorldCoverDownloadError:
            continue

    if not tile_paths:
        raise WorldCoverDownloadError(
            f"Ninguno de los {len(requested_tiles)} tile(s) requeridos para "
            f"este bbox pudo descargarse."
        )

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        src_nodata = sources[0].nodata
        effective_nodata = src_nodata if src_nodata is not None else _DEFAULT_NODATA
        mosaic_array, mosaic_transform = merge(sources, nodata=effective_nodata)
        src_crs = sources[0].crs
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

    dst_array = np.full(
        (mosaic_array.shape[0], dst_height, dst_width), effective_nodata, dtype=mosaic_array.dtype
    )
    reproject(
        source=mosaic_array,
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        src_nodata=effective_nodata,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=effective_nodata,
        # CATEGÓRICO: nunca bilineal. Interpolar códigos de clase
        # fabricaría clases inexistentes (p. ej. Tree cover=10 mezclado
        # con Water=80 daría 45, que no es ninguna clase real).
        resampling=Resampling.nearest,
    )

    with rasterio.open(
        cache_path, "w", driver="GTiff",
        height=dst_height, width=dst_width, count=dst_array.shape[0],
        dtype=dst_array.dtype, crs=crs, transform=dst_transform, nodata=effective_nodata,
    ) as dst:
        dst.write(dst_array)

    return cache_path
