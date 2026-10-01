"""Tests del pipeline DEM: mosaico + reproyección bilinear + cache, sin red."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from ingestion.dem.client import DemDownloadError, TileNotFoundError
from ingestion.dem.pipeline import build_dem
from ingestion.dem.tiles import tile_key
from rasterio.transform import from_origin

BBOX = (-72.6, -37.6, -72.1, -37.1)  # single tile: (-38, -73)


def _write_synthetic_tile(
    path: Path, sw_lat: int, sw_lon: int, value_at_sw: float, nodata: float | None = None
) -> None:
    """Tile sintético 4x4 en EPSG:4326, 1 grado de lado (0.25 deg/pixel),
    con un gradiente lineal simple (rampa este-oeste) para poder verificar
    que la reproyección bilineal interpola en vez de escalonar (nearest).
    nodata=None por defecto, igual que los tiles reales de Copernicus DEM
    (verificado contra el bucket real)."""
    size = 4
    transform = from_origin(sw_lon, sw_lat + 1, 0.25, 0.25)
    data = np.tile(value_at_sw + np.arange(size, dtype="float32") * 10.0, (size, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
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
        assert ds.res == (250.0, 250.0)  # exacto con este fixture — verificado
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


def test_build_dem_output_nodata_is_never_none_even_when_source_tiles_have_none(tmp_path):
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
    with rasterio.open(result_path) as ds:
        assert ds.nodata is not None
        assert ds.nodata != 0.0


def test_build_dem_missing_tile_produces_tagged_hole_not_fabricated_zero(tmp_path):
    # bbox de 2 tiles: (-38,-73) y (-38,-72). Solo el primero "existe".
    two_tile_bbox = (-72.9, -37.6, -72.1, -37.1)
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    missing_key = tile_key(-38, -72)

    def fake_download(key: str, dest_path: Path) -> Path:
        if key == missing_key:
            raise TileNotFoundError(f"404 simulado para {key}")
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-38, sw_lon=-73, value_at_sw=100.0)
        return dest_path

    result_path = build_dem(
        bbox=two_tile_bbox, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    with rasterio.open(result_path) as ds:
        arr = ds.read(1)
        assert ds.nodata is not None
        # La mitad este (tile faltante) debe leerse como nodata, nunca 0.0.
        assert not np.any(arr == 0.0)
        assert np.any(arr == ds.nodata)


def test_build_dem_raises_when_every_requested_tile_fails(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"

    def always_fails(key: str, dest_path: Path) -> Path:
        raise TileNotFoundError(f"404 simulado para {key}")

    with pytest.raises(DemDownloadError, match="Ninguno"):
        build_dem(
            bbox=BBOX, resolution_m=250, crs="EPSG:32719",
            raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=always_fails,
        )
