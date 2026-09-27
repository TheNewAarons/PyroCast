"""Tests de la grilla de trabajo canónica: determinismo y snapping."""
import math

import numpy as np
import xarray as xr
from features.grid.grid import build_grid, build_grid_from_settings, grid_template

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north (WGS84)
CRS = "EPSG:32719"
RESOLUTION_M = 250.0


def test_build_grid_is_deterministic_same_inputs_same_grid():
    grid_a = build_grid(BBOX, CRS, RESOLUTION_M)
    grid_b = build_grid(BBOX, CRS, RESOLUTION_M)
    assert grid_a == grid_b


def test_build_grid_produces_whole_pixel_extent():
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    west, south, east, north = grid.bounds
    assert math.isclose((east - west) / RESOLUTION_M, grid.width, rel_tol=0, abs_tol=1e-9)
    assert math.isclose((north - south) / RESOLUTION_M, grid.height, rel_tol=0, abs_tol=1e-9)
    assert (east - west) % RESOLUTION_M == 0.0
    assert (north - south) % RESOLUTION_M == 0.0


def test_build_grid_exact_multiple_bbox_adds_no_spurious_padding():
    # Un bbox cuyo extent reproyectado ya cae en un múltiplo exacto de la
    # resolución no debe ganar una fila/columna extra por floor/ceil.
    resolution_m = 1000.0
    grid_a = build_grid((-72.0, -38.0, -71.0, -37.0), CRS, resolution_m)
    grid_b = build_grid((-72.0, -38.0, -71.0, -37.0), CRS, resolution_m)
    assert grid_a == grid_b
    west, south, east, north = grid_a.bounds
    assert (east - west) % resolution_m == 0.0
    assert (north - south) % resolution_m == 0.0


def test_build_grid_from_settings_matches_build_grid(monkeypatch):
    required_env = {
        "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
        "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
        "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
        "POSTGRES_PASSWORD": "x",
    }
    for key, value in required_env.items():
        monkeypatch.setenv(key, value)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    expected = build_grid(
        settings.study_area_bbox, settings.crs, float(settings.spatial_resolution_m)
    )
    actual = build_grid_from_settings(settings)
    assert actual == expected
    get_settings.cache_clear()


def test_grid_template_has_expected_shape_crs_and_fill_value():
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    template = grid_template(grid, fill_value=7.0, dtype="float32")
    assert isinstance(template, xr.DataArray)
    assert template.shape == (grid.height, grid.width)
    assert template.rio.crs.to_string() == CRS
    assert np.all(template.values == 7.0)
    # las coords x/y deben caer dentro de los bounds de la grilla, no en
    # los bordes exactos (son centros de píxel, no esquinas).
    west, south, east, north = grid.bounds
    assert west < float(template.x.min()) < east
    assert south < float(template.y.min()) < north
