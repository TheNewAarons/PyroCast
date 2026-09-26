"""Tests de agregación horaria -> diaria de ERA5-Land, sin red."""
import datetime as dt

import numpy as np
import pytest
import xarray as xr
from ingestion.era5.aggregate import aggregate_hourly_to_daily


def _write_synthetic_hourly_nc(
    path, start: dt.datetime, n_hours: int, rate_m_per_hour: float = 0.001,
    time_coord_name: str = "time",
) -> None:
    times = [start + dt.timedelta(hours=h) for h in range(n_hours)]
    lat = np.array([-37.0, -37.1])
    lon = np.array([-72.0, -71.9])
    shape = (n_hours, len(lat), len(lon))
    # t2m sube 1 K cada hora desde 280 K.
    t2m = 280.0 + np.arange(n_hours, dtype="float64").reshape(-1, 1, 1) * np.ones(shape)
    d2m = t2m - 5.0
    u10 = np.full(shape, 2.0)
    v10 = np.full(shape, 3.0)
    # tp: convención real de ERA5-Land — acumulado corrido desde las 00 UTC
    # de cada día. La muestra de la hora 0 de un día es el acumulado
    # COMPLETO del día anterior (24 * rate); las horas 1..23 son el
    # acumulado parcial del día en curso (hora * rate).
    hours_of_day = np.array([t.hour for t in times])
    tp_1d = np.where(
        hours_of_day == 0, 24 * rate_m_per_hour, hours_of_day * rate_m_per_hour
    )
    tp = tp_1d.reshape(-1, 1, 1) * np.ones(shape)
    ds = xr.Dataset(
        {
            "t2m": ((time_coord_name, "latitude", "longitude"), t2m),
            "d2m": ((time_coord_name, "latitude", "longitude"), d2m),
            "u10": ((time_coord_name, "latitude", "longitude"), u10),
            "v10": ((time_coord_name, "latitude", "longitude"), v10),
            "tp": ((time_coord_name, "latitude", "longitude"), tp),
        },
        coords={time_coord_name: times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_aggregate_hourly_to_daily_uses_mean_for_temperature(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    # día 15 completo (00:00..23:00) + el primer paso del día 16 (necesario
    # para el total de precipitación del día 15, ver más abajo).
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=25)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        # media de 280..303 (24 valores, paso 1) = 291.5
        assert float(ds["t2m"].isel(time=0, latitude=0, longitude=0)) == pytest.approx(291.5)


def test_aggregate_hourly_to_daily_precipitation_uses_carryover_not_sum(tmp_path):
    # Este es el hallazgo crítico de la revisión final: ERA5-Land NO es
    # una tasa horaria independiente — es un acumulado corrido desde las
    # 00 UTC. El total real del día 15 es el valor de tp en la muestra
    # del día 16 a las 00:00 (24 * 0.001 = 0.024), NUNCA la suma de las
    # 24 muestras horarias del día 15 (eso daría un valor completamente
    # distinto y contaminado con el acumulado del día anterior).
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=25)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        total = float(ds["tp"].isel(time=0, latitude=0, longitude=0))
        assert total == pytest.approx(0.024)


def test_aggregate_hourly_to_daily_clips_dates_outside_requested_range(tmp_path):
    # 3 dias completos de datos horarios (14, 15, 16 de enero) + el primer
    # paso del día 17, pero solo se pide el 15 -- simula el sobre-fetch
    # documentado de CDS por listas year/month/day.
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 14, 0), n_hours=97)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        assert str(ds["time"].values[0])[:10] == "2026-01-15"


def test_aggregate_hourly_to_daily_normalizes_valid_time_coordinate(tmp_path):
    # El formato NetCDF nuevo de CDS (data_format=netcdf) entrega la
    # coordenada temporal como "valid_time", no "time" — sin normalizar
    # esto, ni siquiera se podría abrir un archivo real.
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(
        hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=25, time_coord_name="valid_time"
    )
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert "time" in ds.coords
        assert ds.sizes["time"] == 1


def test_aggregate_hourly_to_daily_raises_on_unrecognized_variable(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    times = [dt.datetime(2026, 1, 15, 0) + dt.timedelta(hours=h) for h in range(25)]
    ds = xr.Dataset(
        {"ssrd": (("time", "latitude", "longitude"), np.zeros((25, 1, 1)))},
        coords={"time": times, "latitude": [-37.0], "longitude": [-72.0]},
    )
    ds.to_netcdf(hourly_path, engine="h5netcdf")

    with pytest.raises(ValueError, match="ssrd"):
        aggregate_hourly_to_daily(
            hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), tmp_path / "daily.nc"
        )


def test_aggregate_hourly_to_daily_concatenates_multiple_monthly_files(tmp_path):
    # ingestion/era5/pipeline.py pide un archivo por mes calendario
    # (ver month_chunks) — un rango que cruza un mes llega aquí como una
    # lista de archivos, no uno solo.
    file_a = tmp_path / "hourly_jan.nc"
    file_b = tmp_path / "hourly_feb.nc"
    _write_synthetic_hourly_nc(file_a, dt.datetime(2026, 1, 30, 0), n_hours=48)
    _write_synthetic_hourly_nc(file_b, dt.datetime(2026, 2, 1, 0), n_hours=25)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        [file_a, file_b], dt.date(2026, 1, 30), dt.date(2026, 2, 1), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 3  # 30, 31 de enero + 1 de febrero
