"""Tests de NDVI, corrección de offset BOA, nodata y remuestreo, sin red."""
import numpy as np
import pytest
import rasterio
from features.vegetation.ndvi import (
    compute_and_save_vegetation,
    compute_ndvi,
    compute_ndvi_masked,
)
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


def test_compute_ndvi_masked_default_does_not_subtract_a_second_offset():
    # Los composites reales de CDSE/openEO ya vienen con el offset BOA aplicado
    # (DN de RED ~600, NIR ~2500 en vegetación): el NDVI sale directo del cociente.
    # Restar otro -1000 (el bug C2 de docs/review.md) daba NDVI > 1.
    red_dn = np.array([[600.0]])
    nir_dn = np.array([[2600.0]])
    ndvi = compute_ndvi_masked(red_dn, nir_dn, src_nodata=None)
    assert ndvi[0, 0] == pytest.approx((2600 - 600) / (2600 + 600))
    assert 0.0 < ndvi[0, 0] <= 1.0


def test_compute_ndvi_masked_boa_offset_is_still_an_explicit_parameter():
    # fuente con el offset SIN aplicar: RED=1500, NIR=4500 -> 500/3500 -> 0.75
    ndvi = compute_ndvi_masked(np.array([[1500.0]]), np.array([[4500.0]]), None, boa_offset=-1000.0)
    assert ndvi[0, 0] == pytest.approx(0.75)


def test_compute_ndvi_masked_masks_dark_pixels_instead_of_producing_unstable_ratios():
    # agua/sombra con el offset ya aplicado: DN ~0 o negativos por ruido
    red = np.array([[-30.0, 5.0, 600.0]])
    nir = np.array([[31.0, 15.0, 2600.0]])
    ndvi = compute_ndvi_masked(red, nir, None)
    assert ndvi[0, 0] == -9999.0 and ndvi[0, 1] == -9999.0
    assert ndvi[0, 2] == pytest.approx(0.625)


def test_compute_ndvi_masked_rejects_values_outside_the_valid_range():
    # lo que producía el offset duplicado: denominador casi nulo -> NDVI enorme
    with pytest.raises(ValueError, match="fuera de \\[-1, 1\\]"):
        compute_ndvi_masked(
            np.array([[500.0]]), np.array([[1100.0]]), None, boa_offset=-1000.0, mask_dark=False)


def test_compute_ndvi_masked_respects_declared_integer_nodata_sentinel():
    # Un sentinel de nodata entero (p.ej. -32768, convención común en
    # productos Sentinel-2 int16) NO debe leerse como reflectancia: sin
    # esta guarda, (-32768 - -32768)/(-32768 + -32768) = -0.0, un NDVI
    # fabricado que además contaminaría celdas vecinas al reproyectar
    # con bilineal.
    red_dn = np.array([[1500.0, -32768.0]])
    nir_dn = np.array([[4500.0, -32768.0]])
    ndvi = compute_ndvi_masked(red_dn, nir_dn, src_nodata=-32768.0)
    assert ndvi[0, 0] == pytest.approx(0.5)
    assert ndvi[0, 1] == -9999.0


def test_compute_ndvi_masked_respects_nan_input():
    red_dn = np.array([[1500.0, np.nan]])
    nir_dn = np.array([[4500.0, 4500.0]])
    ndvi = compute_ndvi_masked(red_dn, nir_dn, src_nodata=None)
    assert ndvi[0, 0] == pytest.approx(0.5)
    assert ndvi[0, 1] == -9999.0


def test_compute_and_save_vegetation_writes_reprojected_ndvi_with_explicit_nodata(tmp_path):
    composite_path = tmp_path / "composite.tif"
    size = 60
    transform = from_origin(-72.0, -37.0, 0.001, 0.001)
    # banda 1=B04 (red), 2=B08 (nir) -- sin SCL, ver ingestion/sentinel2/client.py
    red = np.full((size, size), 1500.0, dtype="float32")
    nir = np.full((size, size), 4500.0, dtype="float32")
    with rasterio.open(
        composite_path, "w", driver="GTiff", height=size, width=size, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-32768.0,
    ) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)

    output_dir = tmp_path / "vegetation"
    ndvi_path = compute_and_save_vegetation(
        composite_path, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert ndvi_path.exists()
    with rasterio.open(ndvi_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.nodata == -9999.0
        arr = ds.read(1)
    # sin celdas nodata en la entrada, no debe haber NaN sin marcar en la
    # salida (el borde reproyectado puede caer fuera del paralelogramo de
    # datos fuente y quedar en nodata -- eso es correcto, no un NaN suelto).
    # El interior, con solapamiento de datos garantizado, debe reflejar la
    # cociente directo (RED=1500, NIR=4500 -> 0.5, sin offset adicional).
    assert not np.any(np.isnan(arr))
    interior = arr[3:-3, 3:-3]
    assert np.all(interior != -9999.0)
    assert np.allclose(interior, 0.5, atol=1e-4)


def test_compute_and_save_vegetation_propagates_source_nodata(tmp_path):
    composite_path = tmp_path / "composite.tif"
    size = 60
    transform = from_origin(-72.0, -37.0, 0.001, 0.001)
    red = np.full((size, size), 1500.0, dtype="float32")
    nir = np.full((size, size), 4500.0, dtype="float32")
    red[: size // 2, :] = -32768.0
    nir[: size // 2, :] = -32768.0
    with rasterio.open(
        composite_path, "w", driver="GTiff", height=size, width=size, count=2,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-32768.0,
    ) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)

    output_dir = tmp_path / "vegetation"
    ndvi_path = compute_and_save_vegetation(
        composite_path, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )
    with rasterio.open(ndvi_path) as ds:
        arr = ds.read(1)
    # la celda nodata de origen no debe fabricar un NDVI plausible en la
    # salida -- todo lo que sobreviva a la reproyección es -9999 o 0.5,
    # y ambos deben aparecer (la mitad nodata, la mitad con dato real).
    present = set(np.unique(np.round(arr, 4)))
    assert present <= {-9999.0, 0.5}
    assert -9999.0 in present
    assert 0.5 in present
