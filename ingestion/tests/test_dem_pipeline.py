"""Tests del pipeline DEM: mosaico + reproyección bilinear + cache, sin red."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from ingestion.dem.pipeline import build_dem
from rasterio.transform import from_origin

BBOX = (-72.6, -37.6, -72.1, -37.1)  # single tile: (-38, -73)


def _write_synthetic_tile(path: Path, sw_lat: int, sw_lon: int, value_at_sw: float) -> None:
    """Tile sintético 4x4 en EPSG:4326, 1 grado de lado (0.25 deg/pixel),
    con un gradiente lineal simple (rampa este-oeste) para poder verificar
    que la reproyección bilineal interpola en vez de escalonar (nearest)."""
    size = 4
    transform = from_origin(sw_lon, sw_lat + 1, 0.25, 0.25)
    data = np.tile(value_at_sw + np.arange(size, dtype="float32") * 10.0, (size, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-32767.0,
    ) as dst:
        dst.write(data, 1)


def test_build_dem_reprojects_to_target_crs_and_resolution(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"

    def fake_download(key: str, dest_path: Path) -> Path:
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-38, sw_lon=-73, value_at_sw=100.0)
        return dest_path

    result_path = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert result_path.exists()
    with rasterio.open(result_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.res == pytest.approx((250.0, 250.0), abs=1.0)
        arr = ds.read(1)
    # Bilinear must produce interpolated (non-integer-multiple, varied)
    # values, not the small set of exact source values nearest-neighbor
    # would just copy verbatim.
    unique_vals = {round(float(v), 3) for v in arr.flatten() if v > 0}
    assert len(unique_vals) > 4  # more distinct values than the 4 source columns


def test_build_dem_cache_hit_skips_download_entirely(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_download(key: str, dest_path: Path) -> Path:
        calls["n"] += 1
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-38, sw_lon=-73, value_at_sw=100.0)
        return dest_path

    first = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    calls_after_first = calls["n"]
    assert calls_after_first > 0

    second = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert second == first
    assert calls["n"] == calls_after_first  # zero additional download calls
