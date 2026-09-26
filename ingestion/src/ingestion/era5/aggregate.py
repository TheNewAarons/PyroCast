"""Agrega ERA5-Land horario a diario.

Dos métodos de agregación, según ingestion.era5.client.VARIABLE_SPEC
(tabla cerrada y explícita — una variable no reconocida hace fallar la
agregación en vez de promediarla en silencio):

- "mean": temperatura, punto de rocío, viento — magnitudes instantáneas,
  se promedian sobre las 24 muestras horarias del día.
- "carryover": precipitación total (tp). *** ERA5-Land acumula tp desde
  las 00 UTC de cada día (no es una tasa horaria independiente) —
  verificado contra la documentación de ECMWF: el total del día d es el
  valor de tp en la muestra de (d+1) a las 00:00, NO la suma de las 24
  muestras horarias del día d (eso cuenta ~11-12x de más y además
  mezcla el acumulado del día anterior, porque cada muestra horaria ya
  es un acumulado corrido desde la medianoche). Por eso el rango pedido
  al cliente debe extenderse un día más allá de `end` — ver
  ingestion/era5/pipeline.py. ***

Concatena y ordena por tiempo si se pasa más de un archivo horario
(ingestion/era5/pipeline.py pide un archivo por mes calendario — ver
ingestion/era5/client.py:month_chunks). Normaliza el nombre de la
coordenada temporal: el formato NetCDF nuevo de CDS (data_format=netcdf)
entrega `valid_time`, no `time` (verificado contra la documentación de
conversión GRIB->NetCDF de ECMWF) — sin esto, todo lo demás en este
archivo fallaría contra una descarga real."""
import datetime as dt
from pathlib import Path

import xarray as xr

from ingestion.era5.client import VARIABLE_SPEC

_SHORT_NAME_TO_METHOD: dict[str, str] = {short: method for short, method in VARIABLE_SPEC.values()}


def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.variables:
        ds = ds.rename({"valid_time": "time"})
    # El formato NetCDF nuevo de CDS puede agregar dimensiones singleton
    # (number, expver) que no existen en el formato legado.
    squeeze_dims = [d for d in ("number", "expver") if d in ds.dims and ds.sizes[d] == 1]
    if squeeze_dims:
        ds = ds.squeeze(squeeze_dims, drop=True)
    return ds


def _open_and_concat(hourly_nc_paths: list[Path]) -> xr.Dataset:
    datasets = []
    for path in hourly_nc_paths:
        with xr.open_dataset(path, engine="h5netcdf") as raw:
            datasets.append(_normalize_time_coord(raw).load())
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time").sortby("time")


def aggregate_hourly_to_daily(
    hourly_nc_paths: list[Path] | Path, start: dt.date, end: dt.date, output_path: Path
) -> Path:
    paths = [hourly_nc_paths] if isinstance(hourly_nc_paths, Path) else list(hourly_nc_paths)
    ds = _open_and_concat(paths)

    unknown_vars = [str(v) for v in ds.data_vars if str(v) not in _SHORT_NAME_TO_METHOD]
    if unknown_vars:
        raise ValueError(
            f"Variable(s) de ERA5-Land no reconocida(s): {unknown_vars}. Agregar su "
            f"método de agregación ('mean' o 'carryover') a "
            f"ingestion.era5.client.VARIABLE_SPEC antes de agregarla — promediar una "
            f"variable acumulada por error subestimaría su total ~24x."
        )

    mean_vars = [v for v in ds.data_vars if _SHORT_NAME_TO_METHOD[str(v)] == "mean"]
    carryover_vars = [v for v in ds.data_vars if _SHORT_NAME_TO_METHOD[str(v)] == "carryover"]

    target_days: list[dt.date] = []
    current = start
    while current <= end:
        target_days.append(current)
        current += dt.timedelta(days=1)

    daily_mean = None
    if mean_vars:
        clipped = ds[mean_vars].sel(
            time=slice(
                dt.datetime.combine(start, dt.time.min), dt.datetime.combine(end, dt.time.max)
            )
        )
        # .mean() sobre un grupo enteramente NaN (p. ej. una celda oceánica
        # bajo la máscara tierra/mar de ERA5-Land) da NaN por definición
        # (0 muestras válidas => 0/0), nunca 0 — a diferencia de .sum(),
        # que sin min_count trataría "sin muestras" como total 0.
        daily_mean = clipped.resample(time="1D").mean()

    daily_carryover = None
    if carryover_vars:
        carryover_timestamps = [
            dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min) for day in target_days
        ]
        selected = ds[carryover_vars].sel(time=carryover_timestamps)
        # El valor en (día+1) 00:00 ES el acumulado completo del día —
        # se re-etiqueta al día que representa, no al timestamp de origen.
        daily_carryover = selected.assign_coords(
            time=[dt.datetime.combine(day, dt.time.min) for day in target_days]
        )

    if daily_mean is not None and daily_carryover is not None:
        daily = xr.merge([daily_mean, daily_carryover])
    elif daily_mean is not None:
        daily = daily_mean
    elif daily_carryover is not None:
        daily = daily_carryover
    else:
        raise ValueError("El NetCDF horario no tiene ninguna variable reconocida")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_netcdf(output_path, engine="h5netcdf")

    return output_path
