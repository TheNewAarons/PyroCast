"""Orquesta: cache -> solicitud+descarga a CDS -> agregación diaria."""
import datetime as dt
from pathlib import Path
from typing import Any

from ingestion.era5.aggregate import aggregate_hourly_to_daily
from ingestion.era5.cache import cache_key_for
from ingestion.era5.client import ERA5_VARIABLES, build_request


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
    hourly_path = raw_dir / f"era5_hourly_{key}.nc"
    request = build_request(bbox, start, end, variables=variables)
    era5_client.download_hourly(
        "reanalysis-era5-land", request, hourly_path, timeout_seconds=timeout_seconds
    )

    return aggregate_hourly_to_daily(hourly_path, start, end, daily_path)
