"""Tests de agregación horaria -> diaria de ERA5-Land, sin red."""
import datetime as dt

import numpy as np
import pytest
import xarray as xr
from ingestion.era5.aggregate import aggregate_hourly_to_daily


def _write_synthetic_hourly_nc(path, start: dt.datetime, n_hours: int) -> None:
    times = [start + dt.timedelta(hours=h) for h in range(n_hours)]
    lat = np.array([-37.0, -37.1])
    lon = np.array([-72.0, -71.9])
    shape = (n_hours, len(lat), len(lon))
    # t2m sube 1 K cada hora desde 280 K; tp es 0.001 m constante por hora
    # (para que la suma diaria dé un total predecible: 24 * 0.001 = 0.024 m).
    t2m = 280.0 + np.arange(n_hours, dtype="float64").reshape(-1, 1, 1) * np.ones(shape)
    d2m = t2m - 5.0
    u10 = np.full(shape, 2.0)
    v10 = np.full(shape, 3.0)
    tp = np.full(shape, 0.001)
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), d2m),
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
            "tp": (("time", "latitude", "longitude"), tp),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_aggregate_hourly_to_daily_uses_mean_for_temperature(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=24)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        # media de 280..303 (24 valores, paso 1) = 291.5
        assert float(ds["t2m"].isel(time=0, latitude=0, longitude=0)) == pytest.approx(291.5)


def test_aggregate_hourly_to_daily_uses_sum_for_precipitation(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=24)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        total = float(ds["tp"].isel(time=0, latitude=0, longitude=0))
        assert total == pytest.approx(0.024)  # 24 horas x 0.001 m, NO el promedio (0.001)


def test_aggregate_hourly_to_daily_clips_dates_outside_requested_range(tmp_path):
    # 3 dias de datos horarios (14, 15, 16 de enero), pero solo se pide el 15
    # -- simula el sobre-fetch documentado de CDS por listas year/month/day.
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 14, 0), n_hours=72)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        assert str(ds["time"].values[0])[:10] == "2026-01-15"
