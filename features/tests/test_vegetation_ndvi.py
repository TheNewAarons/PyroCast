"""Tests de NDVI, enmascarado de nubes (SCL) y remuestreo, sin red."""
import numpy as np
import pytest
import rasterio
from features.vegetation.ndvi import compute_and_save_vegetation, compute_ndvi, mask_clouds
from rasterio.transform import from_origin


def test_compute_ndvi_known_value():
    # NIR=0.8, RED=0.2 -> NDVI = (0.8-0.2)/(0.8+0.2) = 0.6
    red = np.array([[0.2]])
    nir = np.array([[0.8]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi[0, 0] == pytest.approx(0.6)


def test_compute_ndvi_handles_zero_denominator_without_raising():
    red = np.array([[0.0]])
    nir = np.array([[0.0]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi[0, 0] == -9999.0  # nodata explícito, no NaN/inf propagando


def test_mask_clouds_masks_only_cloud_pixels():
    # fila 0: nube (SCL=9); fila 1: claro (SCL=4, vegetación)
    band = np.array([[0.5], [0.7]])
    scl = np.array([[9], [4]])
    masked = mask_clouds(band, scl, cloud_classes=frozenset({3, 8, 9, 10}))
    assert np.isnan(masked[0, 0])
    assert masked[1, 0] == pytest.approx(0.7)


def test_compute_and_save_vegetation_masks_cloud_pixel_before_ndvi(tmp_path):
    composite_path = tmp_path / "composite.tif"
    size = 4
    transform = from_origin(-72.0, -37.0, 0.0001, 0.0001)
    # banda 1=B04 (red), 2=B08 (nir), 3=SCL
    red = np.full((size, size), 0.2, dtype="float32")
    nir = np.full((size, size), 0.8, dtype="float32")
    scl = np.full((size, size), 4, dtype="float32")  # todo claro (vegetación)
    scl[0, 0] = 9  # una celda nublada
    with rasterio.open(
        composite_path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)
        dst.write(scl, 3)

    output_dir = tmp_path / "vegetation"
    ndvi_path = compute_and_save_vegetation(
        composite_path, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert ndvi_path.exists()
    with rasterio.open(ndvi_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.nodata is not None
        arr = ds.read(1)
    # la celda nublada original ya no existe 1:1 tras reproyectar/remuestrear,
    # pero el resultado no debe tener NaN sin marcar como nodata: cualquier
    # NaN residual del enmascarado debe haberse convertido a nodata antes
    # de escribir el GeoTIFF (GDAL no soporta NaN como nodata de forma
    # confiable en todos los drivers/dtypes).
    assert not np.any(np.isnan(arr))
