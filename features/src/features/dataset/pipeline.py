"""Orquesta, POR EVENTO: bbox recortado (detecciones + buffer de
contexto) -> grilla propia del evento -> resampleo de cada fuente ya
procesada -> máscara de fuego diaria -> tensor. No descarga ni procesa
nada de cero: asume que `pyrocast-ingest dem/era5/sentinel2/worldcover`
ya corrieron para (al menos) el rango de fechas del evento MÁS su
padding previo -- ver docs/dataset-card.md."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling
from shared.config import Settings
from shared.schemas import FireDetection

from features.dataset.assemble import EventChannels, assemble_event_tensor
from features.dataset.resample import resample_to_grid
from features.fire_state.clustering import FireEvent
from features.fire_state.rasterize import DEFAULT_BUFFER_M, build_fire_state
from features.grid.grid import build_grid

DEFAULT_PRE_EVENT_PADDING_DAYS = 5
# "unos días antes" en el enunciado no da un valor -- heurística sin
# calibrar contra incendios reales, igual que los defaults de
# features/fire_state (ver docs/dataset-card.md).

DEFAULT_CONTEXT_BUFFER_M = 2000.0
# Buffer espacial alrededor de las detecciones para definir el bbox del
# EVENTO (contexto para que un modelo de propagación tenga hacia dónde
# propagar) -- deliberadamente DISTINTO de DEFAULT_BUFFER_M (radio de la
# máscara de fuego, ~375 m). Antes del fix, se usaba el mismo buffer
# para ambos: un evento de una sola detección terminaba con un tensor de
# 4x4 píxeles (encontrado en la revisión final del 2026-09-27) -- casi
# sin margen alrededor del fuego. 2000 m es una heurística sin calibrar,
# no un valor derivado de la velocidad de propagación real esperada.

_WEATHER_FIELDS: tuple[str, ...] = (
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
)


@dataclass(frozen=True)
class EventSources:
    elevation_path: Path
    slope_path: Path
    aspect_path: Path
    fuel_type_path: Path
    ndvi_paths_by_month: dict[str, Path]
    weather_paths_by_day: dict[dt.date, dict[str, Path]]


def padded_days_for_event(
    event: FireEvent, pre_event_padding_days: int = DEFAULT_PRE_EVENT_PADDING_DAYS
) -> list[dt.date]:
    """Único lugar que calcula la ventana [start_date - padding,
    end_date] de un evento -- antes de este fix, `features/cli.py`
    recalculaba esta misma ventana con un `5` hardcodeado por su cuenta,
    pudiendo desincronizarse de `DEFAULT_PRE_EVENT_PADDING_DAYS` si uno
    de los dos cambiara sin el otro."""
    padded_start = event.start_date - dt.timedelta(days=pre_event_padding_days)
    total_days = (event.end_date - padded_start).days + 1
    return [padded_start + dt.timedelta(days=i) for i in range(total_days)]


def _event_bbox_wgs84(
    detections: tuple[FireDetection, ...], crs: str, buffer_m: float
) -> tuple[float, float, float, float]:
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for detection in detections:
        x, y = to_crs.transform(detection.longitude, detection.latitude)
        xs.append(x)
        ys.append(y)
    west, south = min(xs) - buffer_m, min(ys) - buffer_m
    east, north = max(xs) + buffer_m, max(ys) + buffer_m
    lon_w, lat_s = to_wgs84.transform(west, south)
    lon_e, lat_n = to_wgs84.transform(east, north)
    return (lon_w, lat_s, lon_e, lat_n)


def _nearest_month_path(
    paths_by_month: dict[str, Path], target_month: str, max_month_distance: int | None = None
) -> Path | None:
    if not paths_by_month:
        return None

    def month_index(key: str) -> int:
        year_str, month_str = key.split("-")
        return int(year_str) * 12 + int(month_str)

    target_idx = month_index(target_month)
    # sorted(): en un empate de distancia, gana el mes más ANTIGUO,
    # siempre -- determinista sin importar el orden de iteración del
    # dict de entrada (a su vez dependiente de un glob() sin orden
    # garantizado). Antes del fix, la empatada la resolvía el orden de
    # inserción del dict, es decir, el orden del glob (encontrado en la
    # revisión final del 2026-09-27).
    closest_key = min(sorted(paths_by_month), key=lambda k: abs(month_index(k) - target_idx))
    distance = abs(month_index(closest_key) - target_idx)
    if max_month_distance is not None and distance > max_month_distance:
        # un composite demasiado lejano en el tiempo no debe presentarse
        # como si fuera el estado vigente de la vegetación (p. ej. un
        # NDVI de hace 6 años) -- mejor NaN explícito (ver el llamador)
        # que un valor plausible pero obsoleto.
        return None
    return paths_by_month[closest_key]


def _single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 archivo en {directory} que matcheara "
            f"{pattern!r}, se encontraron {len(matches)}. Convención de un "
            f"único estudio de área a la vez -- ver docs/limitations.md."
        )
    return matches[0]


def resolve_event_sources(days: list[dt.date], settings: Settings) -> EventSources:
    processed = settings.data_processed_dir
    ndvi_paths_by_month = {
        p.stem.removeprefix("ndvi_"): p
        for p in (processed / "vegetation").glob("ndvi_*.tif")
    }
    weather_paths_by_day: dict[dt.date, dict[str, Path]] = {}
    for day in days:
        iso = day.isoformat()
        # por campo, no todo-o-nada: si falta UN campo (p. ej.
        # precipitation) pero los otros 4 existen, se conservan esos 4 en
        # vez de descartar el día completo -- antes del fix, un solo
        # campo faltante tiraba las otras 4 capas reales a NaN también
        # (encontrado en la revisión final del 2026-09-27).
        available = {
            field: processed / "weather" / f"{field}_{iso}.tif"
            for field in _WEATHER_FIELDS
            if (processed / "weather" / f"{field}_{iso}.tif").exists()
        }
        if available:
            weather_paths_by_day[day] = available

    return EventSources(
        elevation_path=_single_file(processed / "dem", "*.tif"),
        slope_path=processed / "terrain" / "slope_deg.tif",
        aspect_path=processed / "terrain" / "aspect_deg.tif",
        fuel_type_path=processed / "vegetation" / "fuel_type.tif",
        ndvi_paths_by_month=ndvi_paths_by_month,
        weather_paths_by_day=weather_paths_by_day,
    )


def build_dataset_for_event(
    event: FireEvent,
    sources: EventSources,
    resolution_m: float,
    crs: str,
    event_id: int | None = None,
    pre_event_padding_days: int = DEFAULT_PRE_EVENT_PADDING_DAYS,
    context_buffer_m: float = DEFAULT_CONTEXT_BUFFER_M,
    fire_buffer_m: float = DEFAULT_BUFFER_M,
) -> tuple[xr.DataArray, tuple[float, float, float, float]]:
    days = tuple(padded_days_for_event(event, pre_event_padding_days))

    event_bbox = _event_bbox_wgs84(event.detections, crs, buffer_m=context_buffer_m)
    grid = build_grid(event_bbox, crs, resolution_m)

    static = {
        "elevation": resample_to_grid(sources.elevation_path, grid, Resampling.bilinear),
        "slope_deg": resample_to_grid(sources.slope_path, grid, Resampling.bilinear),
        # aspect_deg es una magnitud CIRCULAR (0-360, más el sentinel -1
        # para terreno plano) -- interpolar bilinealmente promedia p.ej.
        # 358 y 2 grados (ambos "norte") a ~180 (sur), un error de 180
        # grados en la variable que impulsa el término de viento/pendiente
        # del autómata celular. nearest nunca fabrica un ángulo
        # intermedio -- mismo criterio que fuel_type (categórico), no
        # porque aspect sea categórico sino porque bilineal es
        # inválido para ángulos igual que para clases. Encontrado en la
        # revisión final del 2026-09-27.
        "aspect_deg": resample_to_grid(sources.aspect_path, grid, Resampling.nearest),
        "fuel_type": resample_to_grid(sources.fuel_type_path, grid, Resampling.nearest),
    }

    fire_masks = build_fire_state(event, grid, buffer_m=fire_buffer_m)

    dynamic: dict[str, dict[dt.date, np.ndarray]] = {field: {} for field in _WEATHER_FIELDS}
    dynamic["ndvi"] = {}
    dynamic["fire_mask"] = {}
    for day in days:
        weather_for_day = sources.weather_paths_by_day.get(day, {})
        for field in _WEATHER_FIELDS:
            path = weather_for_day.get(field)
            dynamic[field][day] = (
                resample_to_grid(path, grid, Resampling.bilinear)
                if path is not None
                else _nan_array(grid.height, grid.width)
            )
        month_key = f"{day.year:04d}-{day.month:02d}"
        ndvi_path = _nearest_month_path(
            sources.ndvi_paths_by_month, month_key, max_month_distance=3
        )
        dynamic["ndvi"][day] = (
            resample_to_grid(ndvi_path, grid, Resampling.bilinear)
            if ndvi_path is not None
            else _nan_array(grid.height, grid.width)
        )
        fire_mask = fire_masks.get(day)
        dynamic["fire_mask"][day] = (
            fire_mask.astype("float32") if fire_mask is not None
            else _zeros_array(grid.height, grid.width)
        )

    channels = EventChannels(days=days, static=static, dynamic=dynamic)
    resolved_event_id = event_id if event_id is not None else event.event_id
    tensor = assemble_event_tensor(channels, grid, resolved_event_id)
    # bbox_cut en WGS84 (grados) -- NO grid.bounds (que está en el CRS
    # proyectado del evento, EPSG:32719/metros). Antes del fix se
    # devolvía grid.bounds y se persistía en PostGIS como si fuera
    # SRID=4326, produciendo un polígono geométricamente sin sentido
    # (encontrado en la revisión final del 2026-09-27).
    return tensor, event_bbox


def _nan_array(height: int, width: int) -> np.ndarray:
    return np.full((height, width), np.nan, dtype="float32")


def _zeros_array(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype="float32")
