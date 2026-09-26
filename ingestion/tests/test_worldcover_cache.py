"""Tests de la clave de cache de WorldCover: bbox+resolución+CRS+versión+año."""
from ingestion.worldcover.cache import cache_key_for

BBOX = (-72.9, -38.9, -72.1, -38.1)


def test_different_year_produces_different_cache_key():
    # Antes del fix, `year` no formaba parte de la clave: dos años
    # distintos colisionaban en el mismo archivo de cache y el segundo
    # `build_worldcover(year=...)` devolvía silenciosamente el raster del
    # primer año.
    key_2020 = cache_key_for(BBOX, 250, "EPSG:32719", "v200", "2020")
    key_2021 = cache_key_for(BBOX, 250, "EPSG:32719", "v200", "2021")
    assert key_2020 != key_2021


def test_same_inputs_produce_same_key():
    key_a = cache_key_for(BBOX, 250, "EPSG:32719", "v200", "2021")
    key_b = cache_key_for(BBOX, 250, "EPSG:32719", "v200", "2021")
    assert key_a == key_b
