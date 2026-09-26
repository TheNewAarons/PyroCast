"""NDVI (Normalized Difference Vegetation Index) desde Sentinel-2 L2A, con
enmascarado de nubes por SCL y remuestreo a la grilla de trabajo.

Fórmula: NDVI = (NIR - RED) / (NIR + RED)               (adimensional, [-1, 1])

*** NDVI es un proxy del estado/vigor de la vegetación (verdor,
actividad fotosintética), NO una medición directa de humedad de
combustible. Vegetación con NDVI alto puede seguir teniendo bajo
contenido de humedad si está fenológicamente senescente o bajo estrés
hídrico no visible en el verdor foliar — no usar NDVI como sustituto de
una medición real de humedad de combustible. ***

Enmascarado de nubes: usa la banda SCL (Scene Classification Layer) de
Sentinel-2 — las clases {3,8,9,10} (sombra de nube, nube prob.
media/alta, cirros delgados) se tratan como sin dato. `CLOUD_SCL_CLASSES`
está duplicado aquí y en ingestion/sentinel2/client.py (mismo conjunto,
aplicado también server-side vía openEO) — no se importa desde
ingestion porque eso invertiría la dependencia features->ingestion
(ingestion ya depende de features para el caso de ingestion/dem/cli.py,
ver docs/decisions.md; la dirección contraria no está justificada por
un frozenset de 4 enteros). Si ambas copias llegaran a divergir, es un
bug — mantenerlas sincronizadas manualmente.

Remuestreo: **bilineal** — NDVI es una magnitud continua, igual que la
elevación del DEM o los campos de ERA5-Land (a diferencia de WorldCover,
que es categórico y usa nearest).
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

_NDVI_NODATA = -9999.0
CLOUD_SCL_CLASSES: frozenset[int] = frozenset({3, 8, 9, 10})


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denominator = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / denominator
    return np.where(denominator == 0, _NDVI_NODATA, ndvi)


def mask_clouds(band: np.ndarray, scl: np.ndarray, cloud_classes: frozenset[int]) -> np.ndarray:
    cloud_mask = np.isin(scl, list(cloud_classes))
    return np.where(cloud_mask, np.nan, band)


def compute_and_save_vegetation(
    composite_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int
) -> Path:
    with rasterio.open(composite_path) as src:
        red = src.read(1).astype("float64")
        nir = src.read(2).astype("float64")
        scl = src.read(3).astype("float64")
        src_crs = src.crs
        src_transform = src.transform
        height, width = src.height, src.width

    red_masked = mask_clouds(red, scl, CLOUD_SCL_CLASSES)
    nir_masked = mask_clouds(nir, scl, CLOUD_SCL_CLASSES)
    ndvi = compute_ndvi(red_masked, nir_masked)
    # Cualquier NaN que haya sobrevivido (celda nublada -> NaN en red/nir ->
    # NaN se propaga a través de la resta/suma en compute_ndvi, ya que
    # denominator == 0 no captura "denominator is NaN") se convierte a
    # nodata explícito antes de escribir — un NaN sin marcar sobreviviría
    # a la reproyección bilineal y contaminaría celdas vecinas al
    # promediar.
    ndvi = np.where(np.isnan(ndvi), _NDVI_NODATA, ndvi).astype("float32")

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
