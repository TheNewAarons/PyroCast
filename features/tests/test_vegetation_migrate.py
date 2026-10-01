"""Migración del NDVI (offset duplicado): identifica el composite reproduciendo el
NDVI viejo y repara el canal sin adivinar."""
import numpy as np
import pytest
import rasterio
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from features.dataset.resample import resample_to_grid
from features.grid.grid import build_grid
from features.vegetation.migrate import migrate_event_ndvi
from features.vegetation.ndvi import compute_and_save_vegetation
from rasterio.transform import from_origin
from rasterio.warp import Resampling


def _composite(path, red, nir, origin=(-72.2, -37.0)):
    size = red.shape[0]
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=2, dtype="int16",
        crs="EPSG:4326", transform=from_origin(origin[0], origin[1], 0.001, 0.001),
        nodata=-32768,
    ) as dst:
        dst.write(red.astype("int16"), 1)
        dst.write(nir.astype("int16"), 2)
    return path


def _tensor_with_old_ndvi(tmp_path, composite):
    grid = build_grid((-72.19, -37.09, -72.12, -37.02), "EPSG:32719", 250.0)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        old_path = compute_and_save_vegetation(
            composite, Path(tmp), "EPSG:32719", 250, boa_offset=-1000.0, check_range=False,
            mask_dark=False,
        )
        old = resample_to_grid(old_path, grid, Resampling.bilinear)
    data = np.zeros((2, len(CHANNEL_ORDER), grid.height, grid.width), dtype="float32")
    data[:, CHANNEL_ORDER.index("ndvi")] = old
    xs = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    ys = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    return xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-01", "2026-01-02"], "channel": list(CHANNEL_ORDER),
                "y": ys, "x": xs},
        attrs={"crs": grid.crs, "transform": tuple(grid.transform)[:6],
               "resolution_m": 250.0, "event_id": 1},
    )


def test_migration_picks_the_composite_that_reproduces_the_old_ndvi_and_fixes_it(tmp_path):
    size = 120
    rng = np.random.default_rng(0)
    # DN reales de vegetación con el offset YA aplicado: RED ~600, NIR ~2600
    red = rng.normal(600, 40, (size, size))
    nir = rng.normal(2600, 80, (size, size))
    right = _composite(tmp_path / "right.tif", red, nir)
    wrong = _composite(tmp_path / "wrong.tif", red * 2, nir * 0.5)
    tensor = _tensor_with_old_ndvi(tmp_path, right)
    old = tensor.values[0, CHANNEL_ORDER.index("ndvi")]
    assert np.nanmax(old) > 1.0  # el bug: NDVI imposible

    result = migrate_event_ndvi(tensor, [wrong, right])
    assert result.composite == right and result.old_match_max_abs_diff < 1e-3
    new = result.new_ndvi
    assert np.nanmin(new) >= -1.0 and np.nanmax(new) <= 1.0
    assert np.nanmedian(new) == pytest.approx((2600 - 600) / (2600 + 600), abs=0.02)


def test_migration_refuses_when_no_composite_reproduces_the_stored_channel(tmp_path):
    size = 120
    red = np.full((size, size), 600.0)
    nir = np.full((size, size), 2600.0)
    right = _composite(tmp_path / "right.tif", red, nir)
    tensor = _tensor_with_old_ndvi(tmp_path, right)
    other = _composite(tmp_path / "other.tif", red * 3, nir * 0.4)
    with pytest.raises(ValueError, match="no se migra a ciegas"):
        migrate_event_ndvi(tensor, [other])
