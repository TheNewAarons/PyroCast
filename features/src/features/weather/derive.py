"""Deriva viento (velocidad/dirección) y humedad relativa aproximada desde
ERA5-Land, y remuestrea de ~9 km nativos a la grilla de trabajo del
proyecto (250 m por defecto, EPSG:32719).

Fórmulas:

Velocidad del viento (m/s):
    speed = hipot(u, v)

Dirección del viento (convención meteorológica: rumbo desde donde SOPLA
el viento, no hacia dónde va; grados, sentido horario desde el norte):
    direction = (grados(atan2(u, v)) + 180) mod 360

Humedad relativa aproximada (%) — fórmula de Magnus-Tetens con los
coeficientes de Alduchov & Eskridge (1996), la variante mejorada más
usada en meteorología operativa:
    RH = 100 * exp(17.625*Td / (243.04+Td)) / exp(17.625*T / (243.04+T))
    (T, Td en grados Celsius)
Válida entre -40°C y 50°C, con un error máximo documentado de ±0.4% RH
en ese rango (Alduchov & Eskridge, J. Appl. Meteor., 1996). Es una
aproximación, no una medición: no reemplaza humedad relativa observada.
Se recorta a [0, 100] — ERA5-Land puede entregar Td ligeramente mayor a
T (saturación/niebla), lo que sin recorte daría RH > 100%.

*** LIMITACIÓN DE DISEÑO, NO UN DETALLE MENOR ***
El remuestreo de ~9 km (grilla nativa de ERA5-Land) a 250 m es
downscaling por INTERPOLACIÓN (bilineal), no una modelación física de
procesos de sub-grilla. No introduce información real a esa escala; solo
suaviza la transición entre celdas de 9 km. Ver docs/limitations.md.

Procesa TODOS los días del NetCDF diario de entrada (no solo el primero)
— un archivo típico cubre un rango de fechas completo.
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

_NODATA = float("nan")


def wind_speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0
    return speed, direction


def relative_humidity_approx(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    numerator = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    rh = 100.0 * numerator / denominator
    return np.clip(rh, 0.0, 100.0)


def _ensure_descending_latitude(
    lat: np.ndarray, fields: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """rasterio/GDAL asumen north-up: fila 0 = latitud más al norte. ERA5
    normalmente entrega latitud descendente (ya north-up), pero si algún
    caller pasara una grilla ascendente, construir la transform con
    lat.max() como origen norte sin voltear los datos georreferenciaría
    todo al revés — sin error, sin warning. Se normaliza explícitamente."""
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        fields = {name: field[::-1, :] for name, field in fields.items()}
    return lat, fields


def _source_transform(lat: np.ndarray, lon: np.ndarray) -> rasterio.Affine:
    xres = abs(float(lon[1] - lon[0]))
    yres = abs(float(lat[1] - lat[0]))
    west = float(lon.min()) - xres / 2
    north = float(lat.max()) + yres / 2
    return from_origin(west, north, xres, yres)


def _reproject_field(
    data: np.ndarray,
    src_transform: rasterio.Affine,
    target_crs: str,
    target_resolution_m: int,
) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    height, width = data.shape
    west, south, east, north = rasterio.transform.array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        "EPSG:4326", target_crs, width, height, west, south, east, north,
        resolution=(target_resolution_m, target_resolution_m),
    )
    dst_array = np.full((dst_height, dst_width), _NODATA, dtype="float32")
    reproject(
        source=data.astype("float32"),
        destination=dst_array,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        src_nodata=_NODATA,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        dst_nodata=_NODATA,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform, dst_width, dst_height


def _write_geotiff(
    path: Path, data: np.ndarray, transform: rasterio.Affine, crs: str
) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=_NODATA,
    ) as dst:
        dst.write(data, 1)
    return path


def compute_and_save_weather(
    daily_nc_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int
) -> dict[str, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, dict[str, Path]] = {
        "wind_speed": {}, "wind_direction": {}, "relative_humidity": {},
        "temperature": {}, "precipitation": {},
    }

    with xr.open_dataset(daily_nc_path, engine="h5netcdf") as ds:
        for time_index in range(ds.sizes["time"]):
            day = ds.isel(time=time_index)
            date_iso = str(day["time"].values)[:10]
            lat = day["latitude"].values
            lon = day["longitude"].values

            fields = {
                "u10": day["u10"].values,
                "v10": day["v10"].values,
                "t2m": day["t2m"].values,
                "d2m": day["d2m"].values,
                "tp": day["tp"].values,
            }
            lat, fields = _ensure_descending_latitude(lat, fields)

            speed, direction = wind_speed_direction(fields["u10"], fields["v10"])
            rh = relative_humidity_approx(fields["t2m"], fields["d2m"])

            src_transform = _source_transform(lat, lon)
            day_fields = {
                "wind_speed": speed,
                "wind_direction": direction,
                "relative_humidity": rh,
                "temperature": fields["t2m"],
                "precipitation": fields["tp"],
            }
            for name, field in day_fields.items():
                reprojected, transform, _, _ = _reproject_field(
                    field, src_transform, target_crs, target_resolution_m
                )
                paths[name][date_iso] = _write_geotiff(
                    output_dir / f"{name}_{date_iso}.tif", reprojected, transform, target_crs
                )

    return paths
