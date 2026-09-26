"""Tests de shared.config: valores por defecto de área de estudio y fallo
explícito si faltan credenciales requeridas."""
import pytest
from pydantic import ValidationError

from shared.config import Settings


REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "test-firms-key",
    "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "test-cds-key",
    "COPERNICUS_DATASPACE_CLIENT_ID": "test-client-id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "test-client-secret",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "pyrocast",
    "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "test-password",
}


def test_settings_load_with_all_required_env_vars(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings()
    assert settings.firms_map_key == "test-firms-key"
    assert settings.postgres_port == 5432


def test_settings_default_study_area_covers_biobio_nuble_araucania(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings()
    min_lon, min_lat, max_lon, max_lat = settings.study_area_bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert settings.crs == "EPSG:32719"
    assert settings.spatial_resolution_m == 250
    assert settings.temporal_resolution == "daily"


def test_settings_raises_clear_error_when_firms_key_missing(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        if key != "FIRMS_MAP_KEY":
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(ValidationError, match="firms_map_key"):
        Settings()
