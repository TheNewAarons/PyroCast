"""Tests de resampleo genérico de un raster a una WorkGrid arbitraria."""
import numpy as np
import rasterio
from features.dataset.resample import resample_to_grid
from features.grid.grid import build_grid
from rasterio.transform import from_origin
from rasterio.warp import Resampling


def _write_tif(path, data, transform, crs, nodata):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype), crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_resample_to_grid_bilinear_uniform_field_stays_uniform(tmp_path):
    src_path = tmp_path / "src.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 42.0, dtype="float32")
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=-9999.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.bilinear)
    assert result.shape == (grid.height, grid.width)
    finite = result[~np.isnan(result)]
    assert finite.size > 0
    assert np.allclose(finite, 42.0, atol=1e-3)


def test_resample_to_grid_nearest_never_fabricates_class_codes(tmp_path):
    src_path = tmp_path / "classes.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 10, dtype="uint8")
    data[:, 25:] = 80  # dos clases reales en bloques, como en WorldCover
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=0.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.nearest)
    present = set(np.unique(result[~np.isnan(result)]))
    assert present <= {10.0, 80.0}


def test_resample_to_grid_propagates_source_nodata(tmp_path):
    src_path = tmp_path / "with_nodata.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 5.0, dtype="float32")
    # bloque de nodata dentro del área que realmente cubre la grilla de
    # destino (calculado a partir de sus bounds reales, no una esquina
    # arbitraria que podría no solaparse en absoluto).
    data[15:25, 15:25] = -9999.0
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=-9999.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.bilinear)
    assert np.any(np.isnan(result))  # el hueco de origen no se fabrica como 5.0
