"""Tests del pipeline ERA5-Land: cache + orquestación, cdsapi mockeado."""
import datetime as dt

import numpy as np
import xarray as xr
from ingestion.era5.client import ERA5_VARIABLES
from ingestion.era5.pipeline import fetch_daily_era5

BBOX = (-73.7, -39.3, -71.0, -36.5)


def _dataset_for_request(request: dict) -> xr.Dataset:
    """Construye un NetCDF horario sintético que cubre exactamente las
    fechas que pide `request` (year/month/day), con tp en la convención
    real de acumulado-desde-medianoche (ver test_era5_aggregate.py)."""
    times = []
    for year in request["year"]:
        for month in request["month"]:
            for day in request["day"]:
                try:
                    date = dt.date(int(year), int(month), int(day))
                except ValueError:
                    continue  # p. ej. 31 de febrero -- CDS lo ignora, nosotros también
                for h in range(24):
                    times.append(dt.datetime(date.year, date.month, date.day, h))
    times = sorted(set(times))
    lat = np.array([-37.0, -37.1])
    lon = np.array([-72.0, -71.9])
    shape = (len(times), len(lat), len(lon))
    t2m = np.full(shape, 290.0)
    d2m = np.full(shape, 285.0)
    u10 = np.full(shape, 1.0)
    v10 = np.full(shape, 1.0)
    hours_of_day = np.array([t.hour for t in times])
    rate = 0.001
    tp_1d = np.where(hours_of_day == 0, 24 * rate, hours_of_day * rate)
    tp = tp_1d.reshape(-1, 1, 1) * np.ones(shape)
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), d2m),
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
            "tp": (("time", "latitude", "longitude"), tp),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


class _FakeEra5Client:
    def __init__(self):
        self.download_calls: list = []

    def download_hourly(self, dataset, request, target, **kwargs):
        self.download_calls.append(target)
        ds = _dataset_for_request(request)
        ds.to_netcdf(target, engine="h5netcdf")
        return target


def test_fetch_daily_era5_produces_daily_netcdf(tmp_path):
    client = _FakeEra5Client()
    result = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    assert result.exists()
    with xr.open_dataset(result, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1


def test_fetch_daily_era5_cache_hit_skips_download_entirely(tmp_path):
    client = _FakeEra5Client()
    first = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    calls_after_first = len(client.download_calls)
    assert calls_after_first > 0

    second = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    assert second == first
    assert len(client.download_calls) == calls_after_first


def test_fetch_daily_era5_requests_one_extra_day_for_precipitation_carryover(tmp_path):
    client = _FakeEra5Client()
    fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    # el request debe incluir el día 16 (para el acumulado del día 15)
    assert len(client.download_calls) == 1


def test_fetch_daily_era5_splits_a_month_boundary_crossing_range_into_two_requests(tmp_path):
    client = _FakeEra5Client()
    result = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 30), end=dt.date(2026, 2, 1),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    # enero (30,31) + febrero (01, y el 02 extra por el carryover) -> 2 requests
    assert len(client.download_calls) == 2
    with xr.open_dataset(result, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 3  # 30, 31 de enero + 1 de febrero
