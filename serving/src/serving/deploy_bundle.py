"""Paquete de datos compacto para desplegar `serving/` (DEPLOY.md).

Toma `data/processed/` (2+ GB, sobre todo clima a 250 m) y produce un .tar.gz
de decenas de MB con SOLO lo que `/predict` lee:

- capas estáticas (DEM, pendiente, orientación, combustible, NDVI), sin cambiar
  su resolución, solo recomprimidas (DEFLATE);
- clima: los 5 campos que entran al tensor (no `wind_speed`/`wind_direction`),
  promediados de 250 m a 2 km. ERA5-Land tiene ~9 km nativos y la grilla de
  250 m era una interpolación bilineal de esa fuente: a 2 km no se pierde
  información real, pero las predicciones del despliegue pueden diferir
  mínimamente de las locales (documentado en DEPLOY.md).

El archivo es determinista (orden fijo, mtime 0): mismos datos -> mismo sha256.
Incluye `ATTRIBUTION.txt` con los avisos que exigen las licencias de las fuentes
(redistribuir derivados requiere atribución, ver docs/data-sources.md).
"""
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

WEATHER_FIELDS_SERVED = ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation")
DEFAULT_WEATHER_RESOLUTION_M = 2000.0
_COMPRESSION = {"compress": "deflate", "zlevel": 9, "tiled": True,
                "blockxsize": 256, "blockysize": 256}

ATTRIBUTION = """PyroCast: paquete de datos de despliegue (derivados procesados).
Herramienta de investigación. No usar para decisiones operativas de combate de
incendios sin validación de CONAF/SENAPRED.

Este paquete contiene datos DERIVADOS (reproyectados, remuestreados) de:

- Copernicus DEM GLO-30: produced using Copernicus WorldDEM-30 © DLR e.V.
  2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
  COPERNICUS by the European Union and ESA; all rights reserved.
- ERA5-Land: Generated using Copernicus Climate Change Service information
  (2025-2026). Muñoz Sabater, J. (2019), ERA5-Land hourly data, C3S Climate Data
  Store, doi:10.24381/cds.e2161bac. Licencia CC BY 4.0. Neither the European
  Commission nor ECMWF is responsible for any use that may be made of the
  Copernicus information or data it contains.
- Sentinel-2 L2A (NDVI): Contains modified Copernicus Sentinel data (2025-2026).
- ESA WorldCover 10 m 2021 v200 (tipo de combustible): © ESA WorldCover project
  2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA
  WorldCover consortium. Zanaga et al. (2022), doi:10.5281/zenodo.7254221.
  Licencia CC BY 4.0.

Detalle de licencias: docs/data-sources.md del repositorio PyroCast.
"""


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    sha256: str
    bytes: int
    files: tuple[str, ...]


def _rewrite(src_path: Path, dst_path: Path, resolution_m: float | None) -> None:
    """Copia recomprimida; con `resolution_m`, además promedia a esa resolución
    (mismo CRS y misma extensión)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(_COMPRESSION)
        if resolution_m is None:
            data = src.read()
            if src.width < 256 or src.height < 256:
                profile.update(tiled=False)
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            return
        west, south, east, north = src.bounds
        width = max(1, round((east - west) / resolution_m))
        height = max(1, round((north - south) / resolution_m))
        transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
        profile.update(width=width, height=height, transform=transform)
        if width < 256 or height < 256:
            profile.update(tiled=False)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                fill = src.nodata if src.nodata is not None else np.nan
                out = np.full((height, width), fill, dtype=src.dtypes[band - 1])
                reproject(
                    source=rasterio.band(src, band), destination=out,
                    src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                    dst_transform=transform, dst_crs=src.crs, dst_nodata=src.nodata,
                    resampling=Resampling.average,
                )
                dst.write(out, band)


def _select(processed: Path) -> list[tuple[Path, str, float | None]]:
    """(archivo de origen, ruta dentro del paquete, resolución nueva o None)."""
    chosen: list[tuple[Path, str, float | None]] = []
    for pattern in ("dem/*.tif", "terrain/slope_deg.tif", "terrain/aspect_deg.tif",
                    "vegetation/fuel_type.tif", "vegetation/ndvi_*.tif"):
        for path in sorted(processed.glob(pattern)):
            chosen.append((path, path.relative_to(processed).as_posix(), None))
    for field in WEATHER_FIELDS_SERVED:
        for path in sorted((processed / "weather").glob(f"{field}_*.tif")):
            name = path.relative_to(processed).as_posix()
            chosen.append((path, name, DEFAULT_WEATHER_RESOLUTION_M))
    return chosen


def build_bundle(
    processed: Path, output: Path, weather_resolution_m: float = DEFAULT_WEATHER_RESOLUTION_M,
) -> BundleInfo:
    """Escribe `output` (.tar.gz) con la estructura de `data/processed/`."""
    selected = _select(processed)
    if not any(name.startswith("dem/") for _, name, _ in selected):
        raise ValueError(f"No hay DEM en {processed}/dem: nada que empaquetar.")
    if not any(name.startswith("weather/") for _, name, _ in selected):
        raise ValueError(f"No hay clima en {processed}/weather: /predict no podría responder.")
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        names: list[str] = []
        for src, name, resolution in selected:
            target_resolution = weather_resolution_m if resolution is not None else None
            _rewrite(src, staging / name, target_resolution)
            names.append(name)
        (staging / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
        (staging / "MANIFEST.json").write_text(json.dumps({
            "files": names, "weather_fields": list(WEATHER_FIELDS_SERVED),
            "weather_resolution_m": weather_resolution_m,
        }, indent=2, sort_keys=True))
        names += ["ATTRIBUTION.txt", "MANIFEST.json"]

        output.parent.mkdir(parents=True, exist_ok=True)
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(names):
                info = tar.gettarinfo(str(staging / name), arcname=f"processed/{name}")
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                info.mode = 0o644
                with open(staging / name, "rb") as fh:
                    tar.addfile(info, fh)
        with open(output, "wb") as out_fh, gzip.GzipFile(
            fileobj=out_fh, mode="wb", mtime=0, compresslevel=9, filename=""
        ) as gz:
            gz.write(raw.getvalue())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return BundleInfo(output, digest, output.stat().st_size, tuple(sorted(names)))


def extract_bundle(bundle: Path, destination: Path) -> Path:
    """Extrae (con el filtro seguro de tarfile) y devuelve `destination/processed`."""
    if (destination / "processed").exists():
        shutil.rmtree(destination / "processed")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    return destination / "processed"
