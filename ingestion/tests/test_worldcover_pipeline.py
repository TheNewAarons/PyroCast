"""Tests del pipeline WorldCover: mosaico + reproyección NEAREST + cache."""
from pathlib import Path

import numpy as np
import rasterio
from ingestion.worldcover.pipeline import build_worldcover
from rasterio.transform import from_origin

BBOX = (-72.9, -38.9, -72.1, -38.1)  # single tile: (-39, -75)


def _write_synthetic_tile(path: Path, sw_lat: int, sw_lon: int) -> None:
    """Tile sintético 4x4 en EPSG:4326, 3 grados de lado, con dos clases
    WorldCover reales en bloques (10=Tree cover, 80=Water) para poder
    verificar que el remuestreo nearest no fabrica clases intermedias."""
    size = 4
    transform = from_origin(sw_lon, sw_lat + 3, 0.75, 0.75)
    data = np.full((size, size), 10, dtype="uint8")
    data[:, size // 2 :] = 80  # mitad este = agua
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0.0,
    ) as dst:
        dst.write(data, 1)


def test_build_worldcover_nearest_resample_never_invents_classes(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"

    def fake_download(key: str, dest_path: Path) -> Path:
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-39, sw_lon=-75)
        return dest_path

    result_path = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    with rasterio.open(result_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        arr = ds.read(1)
    present_values = set(np.unique(arr))
    # Solo deben aparecer los valores de origen (10, 80) y el nodata
    # forzado — nunca un valor fabricado por interpolación entre ambos.
    assert present_values <= {10, 80, 0}


def test_build_worldcover_cache_hit_skips_download_entirely(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_download(key: str, dest_path: Path) -> Path:
        calls["n"] += 1
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-39, sw_lon=-75)
        return dest_path

    first = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    calls_after_first = calls["n"]
    second = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert second == first
    assert calls["n"] == calls_after_first
