"""NDVI (Normalized Difference Vegetation Index) desde Sentinel-2 L2A, con
con chequeo de rango y remuestreo a la grilla de trabajo.

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

Corrección radiométrica -- CORREGIDA en la revisión independiente (docs/review.md,
hallazgo C2): una versión anterior restaba un offset BOA de -1000 a los DN,
suponiendo que el composite traía el offset del processing baseline 04.00+. Los
composites REALES que entrega openEO/CDSE ya vienen con el offset aplicado
(verificado el 2026-10-01 contra los 15 composites descargados: DN mínimos de
5-15 y valores negativos, imposibles con un offset +1000 presente; mediana de
RED ~600 y de NIR ~2500). Restar -1000 de nuevo duplicaba la corrección y daba
NDVI de hasta 5 (mediana ~1.9, con denominadores casi nulos o negativos) en TODOS
los tensores de evento. El offset por defecto es ahora 0.0 (`boa_offset` sigue
siendo un parámetro explícito por si algún día se usa una fuente con el offset
sin aplicar), y `compute_and_save_vegetation` rechaza con un error cualquier NDVI
fuera de [-1, 1] en vez de escribirlo en silencio.

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
SENTINEL2_BOA_ADD_OFFSET = 0.0
# 0.0: los composites de CDSE/openEO ya traen el offset aplicado (ver docstring).
_NDVI_VALID_RANGE_TOLERANCE = 1e-3
_MIN_REFLECTANCE_SUM_DN = 100.0
# con el offset ya aplicado, píxeles oscuros (agua, sombra) tienen DN ~0 o negativos
# por ruido: el cociente NDVI ahí es inestable (denominador ~0) y sin sentido físico.
# Se marcan nodata si RED o NIR <= 0 o si RED+NIR < 100 DN (reflectancia suma < 0.01).


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denominator = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / denominator
    return np.where(denominator == 0, _NDVI_NODATA, ndvi)


def compute_ndvi_masked(
    red_dn: np.ndarray, nir_dn: np.ndarray, src_nodata: float | None,
    boa_offset: float = SENTINEL2_BOA_ADD_OFFSET, check_range: bool = True,
    mask_dark: bool = True,
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

    red = red_dn + boa_offset
    nir = nir_dn + boa_offset
    if mask_dark:
        valid &= (red > 0) & (nir > 0) & ((red + nir) >= _MIN_REFLECTANCE_SUM_DN)
    ndvi = compute_ndvi(red, nir)
    out = np.where(valid, ndvi, _NDVI_NODATA).astype("float32")
    real = out[out != _NDVI_NODATA]
    if check_range and real.size and (
        float(real.max()) > 1.0 + _NDVI_VALID_RANGE_TOLERANCE
        or float(real.min()) < -1.0 - _NDVI_VALID_RANGE_TOLERANCE
    ):
        raise ValueError(
            f"NDVI fuera de [-1, 1] (min={float(real.min()):.3f}, max={float(real.max()):.3f}): "
            f"los DN no corresponden a reflectancia con boa_offset={boa_offset}. Un NDVI así "
            f"es un error de unidades/offset, no un dato -- no se escribe."
        )
    return out


def compute_and_save_vegetation(
    composite_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int,
    boa_offset: float = SENTINEL2_BOA_ADD_OFFSET, check_range: bool = True,
    mask_dark: bool = True,
) -> Path:
    with rasterio.open(composite_path) as src:
        red_dn = src.read(1).astype("float32")
        nir_dn = src.read(2).astype("float32")
        src_nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform
        height, width = src.height, src.width

    ndvi = compute_ndvi_masked(
        red_dn, nir_dn, src_nodata, boa_offset=boa_offset, check_range=check_range,
        mask_dark=mask_dark,
    )

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
