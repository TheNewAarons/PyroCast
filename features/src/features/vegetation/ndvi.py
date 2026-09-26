"""NDVI (Normalized Difference Vegetation Index) desde Sentinel-2 L2A, con
corrección de offset BOA y remuestreo a la grilla de trabajo.

Fórmula: NDVI = (NIR - RED) / (NIR + RED)               (adimensional, [-1, 1])

*** NDVI es un proxy del estado/vigor de la vegetación (verdor,
actividad fotosintética), NO una medición directa de humedad de
combustible. Vegetación con NDVI alto puede seguir teniendo bajo
contenido de humedad si está fenológicamente senescente o bajo estrés
hídrico no visible en el verdor foliar — no usar NDVI como sustituto de
una medición real de humedad de combustible. ***

Enmascarado de nubes: se hace ÚNICAMENTE server-side, en
`ingestion/sentinel2/client.py`, vía la banda SCL antes de la reducción
temporal. El composite que llega aquí ya no trae SCL (se descarta
después de construir la máscara, porque es un código categórico y una
mediana temporal sobre él fabricaría clases inexistentes) — este módulo
no tiene forma de re-aplicar el enmascarado localmente sin esa banda, y
no debe fabricar una. La única defensa en profundidad real que este
módulo aporta es no inventar reflectancia donde el composite declara
nodata (`compute_ndvi_masked`).

Corrección radiométrica: Sentinel-2 L2A con processing baseline 04.00+
(vigente para toda la temporada 2025-26 de este proyecto, desde
2022-01-25) agrega un offset aditivo `BOA_ADD_OFFSET = -1000` a los DN
de reflectancia de superficie. La colección `SENTINEL2_L2A` de CDSE no
publica esta metadata en su descripción (verificado 2026-09-26), así
que no hay forma de leerlo del propio dato — se asume el offset del
baseline vigente y se aplica explícitamente antes de calcular el
cociente (el NDVI es invariante a un factor multiplicativo, pero NO a
un offset aditivo: un DN sin corregir de RED=1500/NIR=4500 da NDVI=0.5;
corregido da 0.75 — un error de 0.25 si se ignora).

Remuestreo: **bilineal** — NDVI es una magnitud continua, igual que la
elevación del DEM o los campos de ERA5-Land (a diferencia de WorldCover,
que es categórico y usa nearest).
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

_NDVI_NODATA = -9999.0
# Sentinel-2 L2A processing baseline 04.00+ (>= 2022-01-25).
SENTINEL2_BOA_ADD_OFFSET = -1000.0


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denominator = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / denominator
    return np.where(denominator == 0, _NDVI_NODATA, ndvi)


def compute_ndvi_masked(
    red_dn: np.ndarray, nir_dn: np.ndarray, src_nodata: float | None
) -> np.ndarray:
    """NDVI desde DN crudos de reflectancia, aplicando la corrección de
    offset BOA y marcando como nodata explícito cualquier píxel inválido
    (NaN, o igual al nodata declarado por la fuente) ANTES de calcular el
    cociente — un sentinel de nodata entero sin esta guarda produciría un
    NDVI fabricado (p. ej. nodata=-32768 da (0)/(-65536) = -0.0), que
    además contaminaría celdas vecinas al reproyectar con bilineal."""
    valid = ~np.isnan(red_dn) & ~np.isnan(nir_dn)
    if src_nodata is not None:
        valid &= (red_dn != src_nodata) & (nir_dn != src_nodata)

    red = red_dn + SENTINEL2_BOA_ADD_OFFSET
    nir = nir_dn + SENTINEL2_BOA_ADD_OFFSET
    ndvi = compute_ndvi(red, nir)
    return np.where(valid, ndvi, _NDVI_NODATA).astype("float32")


def compute_and_save_vegetation(
    composite_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int
) -> Path:
    with rasterio.open(composite_path) as src:
        red_dn = src.read(1).astype("float32")
        nir_dn = src.read(2).astype("float32")
        src_nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform
        height, width = src.height, src.width

    ndvi = compute_ndvi_masked(red_dn, nir_dn, src_nodata)

    west, south, east, north = rasterio.transform.array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, width, height, west, south, east, north,
        resolution=(target_resolution_m, target_resolution_m),
    )
    dst_array = np.full((dst_height, dst_width), _NDVI_NODATA, dtype="float32")
    reproject(
        source=ndvi,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=_NDVI_NODATA,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        dst_nodata=_NDVI_NODATA,
        resampling=Resampling.bilinear,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ndvi_path = output_dir / "ndvi.tif"
    with rasterio.open(
        ndvi_path, "w", driver="GTiff", height=dst_height, width=dst_width, count=1,
        dtype="float32", crs=target_crs, transform=dst_transform, nodata=_NDVI_NODATA,
    ) as dst:
        dst.write(dst_array, 1)

    return ndvi_path
