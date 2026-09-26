"""Orquesta: cache -> solicitud+descarga a CDS (una por mes calendario,
ver month_chunks) -> agregación diaria."""
import datetime as dt
from pathlib import Path
from typing import Any

from ingestion.era5.aggregate import aggregate_hourly_to_daily
from ingestion.era5.cache import cache_key_for
from ingestion.era5.client import ERA5_VARIABLES, VARIABLE_SPEC, build_request, month_chunks


def fetch_daily_era5(
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    era5_client: Any,
    raw_dir: Path,
    cache_dir: Path,
    variables: tuple[str, ...] = ERA5_VARIABLES,
    timeout_seconds: float = 3600.0,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(start, end, variables)
    daily_path = cache_dir / f"era5_daily_{key}.nc"
    if daily_path.exists():
        return daily_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Si se pide alguna variable "carryover" (precipitación total), hay
    # que pedir un día extra más allá de `end`: su total real vive en la
    # muestra de (end+1) a las 00:00 — ver ingestion/era5/aggregate.py.
    needs_extra_day = any(VARIABLE_SPEC[v][1] == "carryover" for v in variables)
    request_end = end + dt.timedelta(days=1) if needs_extra_day else end

    hourly_paths = []
    for chunk_start, chunk_end in month_chunks(start, request_end):
        request = build_request(bbox, chunk_start, chunk_end, variables=variables)
        chunk_key = f"{key}_{chunk_start.isoformat()}_{chunk_end.isoformat()}"
        hourly_path = raw_dir / f"era5_hourly_{chunk_key}.nc"
        era5_client.download_hourly(
            "reanalysis-era5-land", request, hourly_path, timeout_seconds=timeout_seconds
        )
        hourly_paths.append(hourly_path)

    return aggregate_hourly_to_daily(hourly_paths, start, end, daily_path)
