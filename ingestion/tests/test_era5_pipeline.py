"""Tests del pipeline ERA5-Land: cache + orquestación, cdsapi mockeado."""
import datetime as dt

import numpy as np
import xarray as xr
from ingestion.era5.client import ERA5_VARIABLES
from ingestion.era5.pipeline import fetch_daily_era5

BBOX = (-73.7, -39.3, -71.0, -36.5)


class _FakeEra5Client:
    def __init__(self):
        self.download_calls: list = []

    def download_hourly(self, dataset, request, target, **kwargs):
        self.download_calls.append(target)
        lat = np.array([-37.0, -37.1])
        lon = np.array([-72.0, -71.9])
        times = [dt.datetime(2026, 1, 15, h) for h in range(24)]
        shape = (24, 2, 2)
        ds = xr.Dataset(
            {
                "t2m": (("time", "latitude", "longitude"), np.full(shape, 290.0)),
                "d2m": (("time", "latitude", "longitude"), np.full(shape, 285.0)),
                "u10": (("time", "latitude", "longitude"), np.full(shape, 1.0)),
                "v10": (("time", "latitude", "longitude"), np.full(shape, 1.0)),
                "tp": (("time", "latitude", "longitude"), np.full(shape, 0.001)),
            },
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
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
