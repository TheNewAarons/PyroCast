"""Orquesta: cache -> composición mensual Sentinel-2 vía Sentinel2Client."""
from pathlib import Path
from typing import Any

from ingestion.sentinel2.cache import cache_key_for


def fetch_sentinel2(
    bbox: tuple[float, float, float, float],
    year: int,
    month: int,
    client: Any,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(bbox, year, month)
    target = cache_dir / f"sentinel2_{key}.tif"
    if target.exists():
        return target
    result: Path = client.fetch_monthly_composite(bbox, year, month, target)
    return result
