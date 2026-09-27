"""Tests de derivación de viento/humedad y remuestreo, sin red."""
import datetime as dt

import numpy as np
import pytest
import rasterio
import xarray as xr
from features.weather.derive import (
    compute_and_save_weather,
    relative_humidity_approx,
    wind_speed_direction,
)


def test_wind_speed_direction_northerly_wind():
    # Viento soplando DESDE el norte hacia el sur: u=0 (sin componente
    # este-oeste), v=-1 (moviéndose hacia el sur) -> dirección 0/360 (N).
    speed, direction = wind_speed_direction(np.array([0.0]), np.array([-1.0]))
    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(0.0, abs=1e-6)


def test_wind_speed_direction_westerly_wind():
    # Soplando DESDE el oeste hacia el este: u=+1, v=0 -> dirección 270 (O).
    speed, direction = wind_speed_direction(np.array([1.0]), np.array([0.0]))
    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(270.0, abs=1e-6)


def test_relative_humidity_is_100_percent_when_dewpoint_equals_temperature():
    temp_k = np.array([293.15])  # 20 C
    rh = relative_humidity_approx(temp_k, temp_k.copy())
    assert rh[0] == pytest.approx(100.0, abs=0.5)


def test_relative_humidity_decreases_as_dewpoint_spread_widens():
    temp_k = np.full(3, 293.15)
    dewpoints_k = np.array([293.15, 288.15, 273.15])  # spread creciente
    rh = relative_humidity_approx(temp_k, dewpoints_k)
    assert rh[0] > rh[1] > rh[2]
    assert 0.0 <= rh[2] < rh[0] <= 100.0


def test_relative_humidity_is_clamped_to_100_when_dewpoint_exceeds_temperature():
    # ERA5-Land puede entregar Td > T (saturación/niebla) — sin recorte,
    # la fórmula de Magnus-Tetens da RH > 100%, físicamente imposible.
    temp_k = np.array([293.15])
    dewpoint_k = np.array([293.65])  # Td 0.5 K por encima de T
    rh = relative_humidity_approx(temp_k, dewpoint_k)
    assert rh[0] == 100.0


def _write_synthetic_daily_nc(path, lat, lon, n_days: int = 1) -> None:
    shape = (n_days, len(lat), len(lon))
    times = [dt.datetime(2026, 1, 15) + dt.timedelta(days=d) for d in range(n_days)]
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full(shape, 293.15)),
            "d2m": (("time", "latitude", "longitude"), np.full(shape, 283.15)),
            "u10": (("time", "latitude", "longitude"), np.full(shape, 2.0)),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 3.0)),
            "tp": (("time", "latitude", "longitude"), np.full(shape, 0.01)),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_compute_and_save_weather_writes_geotiffs_at_target_resolution(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])  # descendente, típico ERA5
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat, lon)

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert set(paths) == {
        "wind_speed", "wind_direction", "wind_u", "wind_v",
        "relative_humidity", "temperature", "precipitation"
    }
    for per_date in paths.values():
        assert set(per_date) == {"2026-01-15"}
        path = per_date["2026-01-15"]
        assert path.exists()
        with rasterio.open(path) as ds:
            assert ds.crs.to_string() == "EPSG:32719"
            assert ds.res == pytest.approx((250.0, 250.0), abs=1.0)
            assert ds.nodata is not None
            assert np.isnan(ds.nodata)


def test_compute_and_save_weather_processes_every_day_in_the_file(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat, lon, n_days=3)

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert set(paths["wind_speed"]) == {"2026-01-15", "2026-01-16", "2026-01-17"}
    for path in paths["wind_speed"].values():
        assert path.exists()


def test_compute_and_save_weather_marks_source_nodata_cells_as_nodata_not_zero(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    shape = (1, len(lat), len(lon))
    t2m = np.full(shape, 293.15)
    t2m[0, 0, 0] = np.nan  # celda oceánica bajo la máscara tierra/mar
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), np.full(shape, 283.15)),
            "u10": (("time", "latitude", "longitude"), np.full(shape, 2.0)),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 3.0)),
            "tp": (("time", "latitude", "longitude"), np.full(shape, 0.01)),
        },
        coords={"time": [dt.datetime(2026, 1, 15)], "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(daily_nc, engine="h5netcdf")

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )
    with rasterio.open(paths["temperature"]["2026-01-15"]) as ds_out:
        arr = ds_out.read(1)
    # ninguna celda de origen sin dato debe sobrevivir como 0.0 fabricado
    assert not np.any(arr == 0.0)


def test_compute_and_save_weather_wind_components_match_source_u10_v10(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat, lon)  # fixture: u10=2.0, v10=3.0 uniforme

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    with rasterio.open(paths["wind_u"]["2026-01-15"]) as ds:
        u = ds.read(1)
    with rasterio.open(paths["wind_v"]["2026-01-15"]) as ds:
        v = ds.read(1)
    # campo uniforme en el origen -> uniforme tras reproyección bilineal
    assert np.allclose(u[~np.isnan(u)], 2.0, atol=1e-3)
    assert np.allclose(v[~np.isnan(v)], 3.0, atol=1e-3)


def test_compute_and_save_weather_handles_ascending_latitude_input(tmp_path):
    # La grilla real de ERA5-Land viene descendente (north-up); si algún
    # caller entregara latitud ascendente, sin normalizar se georreferencia
    # todo al revés sin ningún error.
    lat_ascending = np.array([-39.0, -38.0, -37.0, -36.0])
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat_ascending, lon)

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )
    # con datos uniformes no se puede verificar el valor, pero al menos
    # debe completar sin error y producir un raster north-up válido.
    with rasterio.open(paths["temperature"]["2026-01-15"]) as ds_out:
        assert ds_out.transform.e < 0  # north-up: e negativo (fila hacia el sur)
