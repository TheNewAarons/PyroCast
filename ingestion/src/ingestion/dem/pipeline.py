"""Orquesta: cache -> descarga de tiles faltantes -> mosaico -> reproyección
bilineal a la grilla de trabajo del proyecto (ver shared/config.py)."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ingestion.dem.cache import cache_key_for
from ingestion.dem.client import DemDownloadError, TileNotFoundError, download_tile
from ingestion.dem.tiles import tile_key, tiles_for_bbox

# Los tiles reales de Copernicus DEM GLO-30 declaran nodata=None (verificado
# contra el bucket real) — no hay forma de distinguir "sin dato" de "0 m" en
# el archivo de origen. Sin un valor de nodata explícito, cualquier celda
# fuera del mosaico (o un hueco entre tiles) se rellena con 0.0 sin marcar,
# y features/terrain la interpretaría como terreno real a nivel del mar,
# fabricando pendientes falsas en el borde. Se fuerza un nodata propio.
_DEFAULT_DEM_NODATA = -32767.0


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
    requested_tiles = tiles_for_bbox(bbox)
    for lat, lon in requested_tiles:
        key = tile_key(lat, lon)
        dest = raw_tiles_dir / f"{key}.tif"
        try:
            tile_paths.append(download_fn(key, dest))
        except TileNotFoundError:
            # GLO-30 Public tiene huecos de cobertura (variante Public vs.
            # -R no liberada) y tiles oceánicos genuinamente no existen —
            # se tolera un tile faltante (queda como hueco en el mosaico,
            # cubierto por el nodata de abajo) y solo se falla si NINGÚN
            # tile de los pedidos pudo descargarse. SOLO un 404 se
            # tolera: una falla de red/5xx/cuota (SourceUnavailableError,
            # QuotaExceededError) propaga -- tolerarla dejaría un hueco
            # silencioso en el mosaico que parece un tile oceánico.
            continue

    if not tile_paths:
        raise DemDownloadError(
            f"Ninguno de los {len(requested_tiles)} tile(s) requeridos para "
            f"este bbox pudo descargarse."
        )

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        src_nodata = sources[0].nodata
        # Los tiles reales de Copernicus DEM declaran nodata=None — sin un
        # valor propio, cualquier hueco del mosaico (tile faltante, borde)
        # se rellena con 0.0 sin marcar, y features/terrain lo tomaría
        # como terreno real a nivel del mar.
        effective_nodata = src_nodata if src_nodata is not None else _DEFAULT_DEM_NODATA
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
        resampling=Resampling.bilinear,
    )

    with rasterio.open(
        cache_path, "w", driver="GTiff",
        height=dst_height, width=dst_width, count=dst_array.shape[0],
        dtype=dst_array.dtype, crs=crs, transform=dst_transform, nodata=effective_nodata,
    ) as dst:
        dst.write(dst_array)

    return cache_path
