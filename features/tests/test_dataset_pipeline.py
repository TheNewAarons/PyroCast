"""Tests de la orquestación por evento: bbox recortado, ventana de
padding previo, resolución de fuentes por convención de archivos, y
ensamblado end-to-end con rasters de fixture."""
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
from features.dataset.assemble import CHANNEL_ORDER
from features.dataset.pipeline import (
    DEFAULT_CONTEXT_BUFFER_M,
    DEFAULT_PRE_EVENT_PADDING_DAYS,
    _event_bbox_wgs84,
    _nearest_month_path,
    build_dataset_for_event,
    padded_days_for_event,
    resolve_event_sources,
)
from features.fire_state.clustering import FireEvent
from features.grid.grid import build_grid
from rasterio.transform import from_origin
from shared.config import Settings
from shared.schemas import FireDetection


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_event_bbox_wgs84_pads_around_single_detection_without_crashing():
    # una detección sola tiene extensión espacial CERO antes del buffer --
    # no debe dividir por cero ni producir un bbox degenerado.
    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(event_id=0, detections=(_det(-38.0, -72.5, at),))
    west, south, east, north = _event_bbox_wgs84(event.detections, "EPSG:32719", buffer_m=1000.0)
    assert west < east
    assert south < north
    assert west < -72.5 < east
    assert south < -38.0 < north


def test_padded_days_for_event_matches_pre_event_padding_days():
    # Un solo lugar calcula la ventana de padding -- antes del fix, el
    # CLI la recalculaba con un `5` hardcodeado por su cuenta, pudiendo
    # desincronizarse de DEFAULT_PRE_EVENT_PADDING_DAYS (encontrado en la
    # revisión final del 2026-09-27).
    at1 = dt.datetime(2026, 1, 10, 12, 0, tzinfo=dt.UTC)
    at2 = dt.datetime(2026, 1, 12, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(event_id=0, detections=(_det(-38.0, -72.5, at1), _det(-38.0, -72.5, at2)))
    days = padded_days_for_event(event)
    assert days[0] == event.start_date - dt.timedelta(days=DEFAULT_PRE_EVENT_PADDING_DAYS)
    assert days[-1] == event.end_date
    assert len(days) == (event.end_date - days[0]).days + 1


def test_nearest_month_path_picks_closest_available_month():
    paths = {"2025-11": Path("nov.tif"), "2026-02": Path("feb.tif")}
    # "2026-01" está a 2 meses de nov (2025-11) y 1 mes de feb (2026-02)
    assert _nearest_month_path(paths, "2026-01") == Path("feb.tif")


def test_nearest_month_path_returns_none_when_no_months_available():
    assert _nearest_month_path({}, "2026-01") is None


def test_nearest_month_path_breaks_ties_toward_the_earlier_month():
    # Dos meses EQUIDISTANTES del objetivo -- antes del fix, cuál ganaba
    # dependía del orden de iteración del dict (a su vez dependiente del
    # orden de un glob() no garantizado). Se fuerza a que el mes más
    # ANTIGUO gane la empatada, siempre, sin importar el orden de entrada.
    tie_a_first = {"2025-12": Path("dec.tif"), "2026-02": Path("feb.tif")}
    tie_b_first = {"2026-02": Path("feb.tif"), "2025-12": Path("dec.tif")}
    assert _nearest_month_path(tie_a_first, "2026-01") == Path("dec.tif")
    assert _nearest_month_path(tie_b_first, "2026-01") == Path("dec.tif")


def test_nearest_month_path_returns_none_beyond_max_distance():
    paths = {"2019-06": Path("old.tif")}
    # "old.tif" está a más de 3 meses de "2026-01" -- no debe presentarse
    # como si fuera el NDVI vigente de un composite de hace años
    # (encontrado en la revisión final del 2026-09-27).
    assert _nearest_month_path(paths, "2026-01", max_month_distance=3) is None


def _write_tif(path: Path, value: float, size: int, transform, crs: str, nodata=-9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((size, size), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_resolve_event_sources_finds_files_by_convention(tmp_path):
    processed = tmp_path / "processed"
    transform = from_origin(190000, 5791000, 250, 250)
    _write_tif(processed / "dem" / "dem_abc123.tif", 100.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "slope_deg.tif", 5.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "aspect_deg.tif", 180.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "vegetation" / "fuel_type.tif", 3.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, 20, transform, "EPSG:32719")
    day = dt.date(2026, 1, 15)
    for field in ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"):
        _write_tif(
            processed / "weather" / f"{field}_{day.isoformat()}.tif",
            1.0, 20, transform, "EPSG:32719",
        )

    settings = Settings(
        firms_map_key="x", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )
    sources = resolve_event_sources([day], settings)
    assert sources.elevation_path == processed / "dem" / "dem_abc123.tif"
    assert sources.slope_path.name == "slope_deg.tif"
    assert sources.ndvi_paths_by_month == {"2026-01": processed / "vegetation" / "ndvi_2026-01.tif"}
    assert day in sources.weather_paths_by_day
    assert set(sources.weather_paths_by_day[day]) == {
        "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"
    }


def test_resolve_event_sources_keeps_partial_weather_days_instead_of_dropping_them(tmp_path):
    # Antes del fix, si faltaba UN SOLO campo de clima de un día, el día
    # completo se trataba como "sin clima" (los 5 campos a NaN) --
    # descartando 4 capas reales. Encontrado en la revisión final del
    # 2026-09-27.
    processed = tmp_path / "processed"
    transform = from_origin(190000, 5791000, 250, 250)
    _write_tif(processed / "dem" / "dem_abc123.tif", 100.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "slope_deg.tif", 5.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "aspect_deg.tif", 180.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "vegetation" / "fuel_type.tif", 3.0, 20, transform, "EPSG:32719")
    day = dt.date(2026, 1, 15)
    # solo wind_u existe -- precipitation, etc. faltan.
    _write_tif(
        processed / "weather" / f"wind_u_{day.isoformat()}.tif", 1.0, 20, transform, "EPSG:32719"
    )

    settings = Settings(
        firms_map_key="x", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )
    sources = resolve_event_sources([day], settings)
    assert day in sources.weather_paths_by_day
    assert set(sources.weather_paths_by_day[day]) == {"wind_u"}


def test_build_dataset_for_event_pads_before_first_detection_and_fills_fire_mask(tmp_path):
    processed = tmp_path / "processed"
    resolution_m = 250.0
    crs = "EPSG:32719"

    day1 = dt.datetime(2026, 1, 10, 12, 0, tzinfo=dt.UTC)
    day2 = dt.datetime(2026, 1, 11, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(
        event_id=3,
        detections=(_det(-38.0, -72.5, day1), _det(-38.0, -72.5, day2)),
    )
    event_bbox = _event_bbox_wgs84(event.detections, crs, buffer_m=DEFAULT_CONTEXT_BUFFER_M)
    grid = build_grid(event_bbox, crs, resolution_m)
    transform, size = grid.transform, max(grid.height, grid.width)

    _write_tif(processed / "dem" / "dem_x.tif", 100.0, size, transform, crs)
    _write_tif(processed / "terrain" / "slope_deg.tif", 5.0, size, transform, crs)
    # dos bloques de orientación real (358, 2 grados -- norte), como en
    # un DEM real que oscila alrededor de 0/360 -- si se interpolara
    # bilinealmente, daría ~180 (sur), el error de C2.
    aspect_path = processed / "terrain" / "aspect_deg.tif"
    aspect_path.parent.mkdir(parents=True, exist_ok=True)
    aspect_data = np.full((size, size), 358.0, dtype="float32")
    aspect_data[:, size // 2:] = 2.0
    with rasterio.open(
        aspect_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(aspect_data, 1)
    _write_tif(processed / "vegetation" / "fuel_type.tif", 3.0, size, transform, crs)
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, size, transform, crs)

    all_days = padded_days_for_event(event)
    for day in all_days:
        for field in ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"):
            _write_tif(
                processed / "weather" / f"{field}_{day.isoformat()}.tif", 1.0, size, transform, crs
            )

    settings = Settings(
        firms_map_key="x", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )
    sources = resolve_event_sources(all_days, settings)
    tensor, bbox_cut = build_dataset_for_event(
        event, sources, resolution_m, crs, event_id=event.event_id
    )

    assert tensor.dims == ("day", "channel", "y", "x")
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert len(tensor.coords["day"]) == len(all_days)
    assert tensor.attrs["event_id"] == event.event_id

    # C1: el bbox devuelto debe estar en WGS84 (grados), no en metros
    # UTM -- antes del fix devolvía grid.bounds (EPSG:32719).
    assert bbox_cut == event_bbox
    assert -180.0 <= bbox_cut[0] <= 180.0
    assert -90.0 <= bbox_cut[1] <= 90.0

    # C2: aspect_deg no debe fabricar ~180 (promedio bilineal de 358/2) --
    # nearest solo puede devolver alguno de los dos valores reales.
    aspect_idx = CHANNEL_ORDER.index("aspect_deg")
    aspect_values = set(np.unique(tensor.values[0, aspect_idx, :, :]))
    assert aspect_values <= {358.0, 2.0}

    fire_mask_idx = CHANNEL_ORDER.index("fire_mask")
    padded_start = all_days[0]
    first_day_iso = padded_start.isoformat()
    day_index = list(tensor.coords["day"].values).index(first_day_iso)
    # día de padding (antes de la primera detección real) -- sin fuego.
    assert np.all(tensor.values[day_index, fire_mask_idx, :, :] == 0.0)

    # días con detección real: el centro del evento SÍ está en llamas.
    for detection_day in (day1.date(), day2.date()):
        detection_day_index = list(tensor.coords["day"].values).index(detection_day.isoformat())
        assert np.any(tensor.values[detection_day_index, fire_mask_idx, :, :] == 1.0)

    # I2: con el buffer de contexto por defecto (no el radio de la
    # máscara de fuego, mucho más chico), el tensor debe ser
    # sustancialmente más grande que 4x4 -- antes del fix, un evento de 2
    # detecciones colocadas casi en el mismo punto producía un tensor
    # minúsculo, sin margen para que un modelo de propagación tenga algo
    # hacia dónde propagar.
    assert tensor.shape[-2] >= 15
    assert tensor.shape[-1] >= 15
