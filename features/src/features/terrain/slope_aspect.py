"""Pendiente y orientación (aspect) a partir de un DEM, método de Horn (1981)
— el mismo algoritmo de kernel 3x3 que usan GDAL `gdaldem` y ESRI.

Fórmulas (fila = eje Y, aumenta hacia el sur bajo una transform north-up
estándar; columna = eje X, aumenta hacia el este):

    dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)

    donde la ventana 3x3 centrada en el píxel es:
        a b c
        d e f
        g h i

    slope_deg  = grados(atan(hipot(dz/dx, dz/dy)))                    [0, 90]
    aspect_deg = (grados(atan2(-dz/dx, dz/dy))) mod 360                [0, 360)
                 -1.0 si la celda es plana (dz/dx == dz/dy == 0)

aspect es el rumbo de la dirección CUESTA ABAJO (hacia dónde "mira" la
ladera), medido en sentido horario desde el norte (0=N, 90=E, 180=S,
270=O) — la convención estándar en GIS para orientación de terreno.

Limitación conocida: el padding de borde usa modo 'edge' (replica el
píxel más cercano). Las celdas cuya elevación de origen ES nodata se
marcan como nodata en la salida (no se fabrica un valor), pero las
celdas VECINAS a un hueco de datos siguen usando ese hueco en su
kernel 3x3 — su pendiente/orientación no es confiable cerca del borde
del hueco. Ver docs/data-sources.md.
"""
from pathlib import Path

import numpy as np
import rasterio


def compute_slope_aspect(
    elevation: np.ndarray, cellsize_x: float, cellsize_y: float
) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(elevation, pad_width=1, mode="edge")

    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)

    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    flat = (dzdx == 0.0) & (dzdy == 0.0)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = np.degrees(aspect_rad) % 360.0
    aspect_deg = np.where(flat, -1.0, aspect_deg)

    return slope_deg, aspect_deg


_OUTPUT_NODATA = -9999.0


def compute_and_save_terrain(dem_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with rasterio.open(dem_path) as src:
        if src.crs is None or src.crs.is_geographic:
            raise ValueError(
                f"compute_and_save_terrain requiere un DEM ya reproyectado a un "
                f"CRS proyectado (metros) — recibido: {src.crs}. Calcular "
                f"pendiente/orientación sobre un CRS geográfico (grados) usa el "
                f"tamaño de celda en grados como si fuera metros, fabricando "
                f"pendientes incorrectas."
            )
        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError(
                "compute_and_save_terrain requiere una transform sin rotación "
                "(transform.b == transform.d == 0)."
            )
        elevation = src.read(1).astype("float64")
        cellsize_x = src.transform.a
        cellsize_y = -src.transform.e  # e es negativo en rasters north-up
        profile = src.profile
        src_nodata = src.nodata

    slope_deg, aspect_deg = compute_slope_aspect(elevation, cellsize_x, cellsize_y)

    if src_nodata is not None:
        nodata_mask = elevation == src_nodata
        slope_deg = np.where(nodata_mask, _OUTPUT_NODATA, slope_deg)
        aspect_deg = np.where(nodata_mask, _OUTPUT_NODATA, aspect_deg)

    output_dir.mkdir(parents=True, exist_ok=True)
    slope_path = output_dir / "slope_deg.tif"
    aspect_path = output_dir / "aspect_deg.tif"

    out_profile = {**profile, "dtype": "float32", "count": 1, "nodata": _OUTPUT_NODATA}
    with rasterio.open(slope_path, "w", **out_profile) as dst:
        dst.write(slope_deg.astype("float32"), 1)
    with rasterio.open(aspect_path, "w", **out_profile) as dst:
        dst.write(aspect_deg.astype("float32"), 1)

    return slope_path, aspect_path
