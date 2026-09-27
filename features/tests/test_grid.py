"""Tests de la grilla de trabajo canónica: determinismo y snapping."""
import math

import numpy as np
import xarray as xr
from features.grid.grid import build_grid, build_grid_from_settings, grid_template
from rasterio.warp import transform_bounds

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


def test_build_grid_matches_hand_computed_bounds_for_the_real_study_area():
    # Valor concreto, calculado a mano (floor/ceil sobre
    # transform_bounds("EPSG:4326","EPSG:32719", *BBOX) =
    # (78951.22, 5639387.91, 327543.20, 5958731.91), independientemente
    # verificado en la revisión final de este módulo) -- pin exacto, no
    # solo una propiedad general. Un mutante que usara floor() en vez de
    # ceil() para el borde superior (o viceversa), o que agregara
    # padding espurio, cambiaría estos números.
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    assert grid.bounds == (78750.0, 5639250.0, 327750.0, 5958750.0)
    assert grid.width == 996
    assert grid.height == 1278


def test_build_grid_bounds_always_contain_the_requested_bbox():
    # La invariante que realmente importa: la grilla NUNCA debe recortar
    # el bbox pedido (solo puede sobre-cubrir, nunca sub-cubrir) -- un
    # mutante que usara floor() en los cuatro bordes (en vez de floor
    # oeste/sur + ceil este/norte) violaría esto en el borde este/norte.
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    west, south, east, north = grid.bounds
    req_west, req_south, req_east, req_north = transform_bounds("EPSG:4326", CRS, *BBOX)
    assert west <= req_west
    assert south <= req_south
    assert east >= req_east
    assert north >= req_north


def test_build_grid_south_and_north_already_exact_multiples_get_no_extra_padding():
    # bbox construido (a partir de un rectángulo UTM cuyo sur/norte SON
    # múltiplos exactos de 1000, ida y vuelta a WGS84) para que
    # transform_bounds reproduzca sur=5700000.0 y norte=5800000.0 EXACTOS
    # -- verificado por separado antes de escribir este test (ver
    # docs/superpowers/plans/2026-09-27-features-grid-fire-state.md). El
    # test anterior con este nombre no probaba en realidad un caso de
    # múltiplo exacto (su bbox reproyectaba a valores no-múltiplos); este
    # sí, y confirma que floor/ceil sobre un valor YA entero no le suma
    # una fila/columna de más.
    bbox = (-72.4542643735273, -38.79772154662968, -71.27552823619688, -37.92558174988909)
    resolution_m = 1000.0
    grid = build_grid(bbox, "EPSG:32719", resolution_m)
    west, south, east, north = grid.bounds
    assert south == 5700000.0
    assert north == 5800000.0


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
