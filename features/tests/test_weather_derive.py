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
    assert 0.0 <= rh[2] < rh[0] <= 100.5


def _write_synthetic_daily_nc(path, lat, lon) -> None:
    shape = (1, len(lat), len(lon))
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full(shape, 293.15)),
            "d2m": (("time", "latitude", "longitude"), np.full(shape, 283.15)),
            "u10": (("time", "latitude", "longitude"), np.full(shape, 2.0)),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 3.0)),
            "tp": (("time", "latitude", "longitude"), np.full(shape, 0.01)),
        },
        coords={"time": [dt.datetime(2026, 1, 15)], "latitude": lat, "longitude": lon},
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
        "wind_speed", "wind_direction", "relative_humidity", "temperature", "precipitation"
    }
    for path in paths.values():
        assert path.exists()
        with rasterio.open(path) as ds:
            assert ds.crs.to_string() == "EPSG:32719"
            assert ds.res == pytest.approx((250.0, 250.0), abs=1.0)
