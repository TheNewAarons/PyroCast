"""Agrega ERA5-Land horario a diario: media para temperatura/punto de
rocío/viento, SUMA para precipitación total (es un campo acumulado, no
una tasa instantánea — promediarlo daría un valor ~24x menor al total
real del día). Recorta primero al rango [start, end] exacto, porque una
solicitud CDS construida con listas year/month/day puede devolver días
de más cuando el rango cruza un límite de mes (ver
ingestion/era5/client.py y docs/decisions.md)."""
import datetime as dt
from pathlib import Path

import xarray as xr

_SUM_VARIABLES = {"tp"}


def aggregate_hourly_to_daily(
    hourly_nc_path: Path, start: dt.date, end: dt.date, output_path: Path
) -> Path:
    with xr.open_dataset(hourly_nc_path, engine="h5netcdf") as ds:
        clipped = ds.sel(
            time=slice(
                dt.datetime.combine(start, dt.time.min),
                dt.datetime.combine(end, dt.time.max),
            )
        )

        sum_vars = [v for v in clipped.data_vars if v in _SUM_VARIABLES]
        mean_vars = [v for v in clipped.data_vars if v not in _SUM_VARIABLES]

        daily_mean = clipped[mean_vars].resample(time="1D").mean() if mean_vars else None
        daily_sum = clipped[sum_vars].resample(time="1D").sum() if sum_vars else None

        if daily_mean is not None and daily_sum is not None:
            daily = xr.merge([daily_mean, daily_sum])
        else:
            daily = daily_mean if daily_mean is not None else daily_sum

        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_netcdf(output_path, engine="h5netcdf")

    return output_path
