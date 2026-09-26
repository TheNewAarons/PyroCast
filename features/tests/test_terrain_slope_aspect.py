"""Tests de pendiente/orientación (método de Horn) contra un DEM sintético
con pendiente conocida analíticamente."""
import numpy as np
import pytest
import rasterio
from features.terrain.slope_aspect import compute_and_save_terrain, compute_slope_aspect
from rasterio.transform import from_origin


def test_flat_plane_has_zero_slope_and_sentinel_aspect():
    elevation = np.full((5, 5), 100.0, dtype="float64")
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    assert np.allclose(slope, 0.0)
    assert np.all(aspect == -1.0)


def test_plane_rising_to_the_east_has_45_degree_slope_facing_west():
    # Z = column index (1 m rise per 1 m de cellsize hacia el este) ->
    # analíticamente una pendiente de 45 grados, orientada cuesta abajo
    # hacia el Oeste (270 grados).
    size = 7
    elevation = np.tile(np.arange(size, dtype="float64"), (size, 1))
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    interior = slice(1, -1)
    assert np.allclose(slope[interior, interior], 45.0, atol=1e-6)
    assert np.allclose(aspect[interior, interior], 270.0, atol=1e-6)


def test_plane_rising_to_the_northeast_has_non_cardinal_aspect():
    # Z aumenta con fila (hacia el norte, fila 0 = borde norte bajo una
    # transform north-up) y columna (este) por igual -> cuesta abajo es
    # suroeste = 225 grados. Esto fija el orden/signo de atan2: una
    # fórmula al revés daría 45 o 135, no 225.
    size = 7
    rows = np.arange(size, dtype="float64").reshape(-1, 1)
    cols = np.arange(size, dtype="float64").reshape(1, -1)
    elevation = -rows + cols  # más alto al norte (fila menor) y al este
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    interior = slice(1, -1)
    assert np.allclose(aspect[interior, interior], 225.0, atol=1e-6)


def test_compute_and_save_terrain_writes_two_geotiffs_preserving_crs(tmp_path):
    dem_path = tmp_path / "dem.tif"
    size = 6
    transform = from_origin(500000, 5800000, 250, 250)  # tipo UTM, metros
    elevation = np.tile(np.arange(size, dtype="float32") * 10.0, (size, 1))
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:32719", transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    output_dir = tmp_path / "terrain"
    slope_path, aspect_path = compute_and_save_terrain(dem_path, output_dir)

    assert slope_path.exists()
    assert aspect_path.exists()
    with rasterio.open(slope_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.transform == transform
    with rasterio.open(aspect_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"


def test_compute_and_save_terrain_rejects_geographic_crs(tmp_path):
    # Un DEM sin reproyectar (grados, no metros) haría que cellsize_x/y se
    # interpreten como metros por error, produciendo pendientes fabricadas
    # (p. ej. una pendiente real de 45 grados calculada como ~90 grados).
    dem_path = tmp_path / "dem_wgs84.tif"
    size = 6
    transform = from_origin(-72.0, -37.0, 0.1, 0.1)
    elevation = np.tile(np.arange(size, dtype="float32") * 10.0, (size, 1))
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    with pytest.raises(ValueError, match="geográfic"):
        compute_and_save_terrain(dem_path, tmp_path / "terrain")


def test_compute_and_save_terrain_masks_nodata_cells_in_output(tmp_path):
    dem_path = tmp_path / "dem_with_holes.tif"
    size = 6
    transform = from_origin(500000, 5800000, 250, 250)
    elevation = np.tile(np.arange(size, dtype="float32") * 10.0, (size, 1))
    elevation[0, 0] = -32767.0  # hueco de datos
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:32719", transform=transform, nodata=-32767.0,
    ) as dst:
        dst.write(elevation, 1)

    slope_path, aspect_path = compute_and_save_terrain(dem_path, tmp_path / "terrain")

    with rasterio.open(slope_path) as ds:
        assert ds.nodata is not None
        arr = ds.read(1)
        assert arr[0, 0] == ds.nodata  # celda de origen sin dato -> nodata, no un valor fabricado
