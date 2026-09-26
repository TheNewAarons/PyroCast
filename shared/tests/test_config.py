"""Tests de shared.config: valores por defecto de área de estudio y fallo
explícito si faltan o están vacías las credenciales requeridas.

Todas las llamadas a Settings() aquí usan _env_file=None: si no lo
hiciéramos, un .env real en el directorio de trabajo (p. ej. el que
README.md le pide al desarrollador crear con `cp .env.example .env`)
se leería antes que las variables de entorno que monkeypatch setea,
haciendo que estos tests dependan del estado accidental del CWD.
"""
import pytest
from pydantic import ValidationError
from shared.config import ConfigurationError, Settings, get_settings

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
    settings = Settings(_env_file=None)
    assert settings.firms_map_key == "test-firms-key"
    assert settings.postgres_port == 5432


def test_settings_default_study_area_covers_biobio_nuble_araucania(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings(_env_file=None)
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
        Settings(_env_file=None)


def test_settings_data_dirs_overridable_by_env_for_docker_volume_mount(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATA_RAW_DIR", "/data/raw")
    monkeypatch.setenv("DATA_INTERIM_DIR", "/data/interim")
    monkeypatch.setenv("DATA_PROCESSED_DIR", "/data/processed")
    settings = Settings(_env_file=None)
    assert str(settings.data_raw_dir) == "/data/raw"
    assert str(settings.data_interim_dir) == "/data/interim"
    assert str(settings.data_processed_dir) == "/data/processed"


def test_settings_raises_when_firms_key_is_empty_string(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FIRMS_MAP_KEY", "")
    with pytest.raises(ValidationError, match="firms_map_key"):
        Settings(_env_file=None)


def test_get_settings_raises_configuration_error_naming_missing_env_vars(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # sin .env real en el cwd
    for key, value in REQUIRED_ENV.items():
        if key not in ("FIRMS_MAP_KEY", "POSTGRES_PASSWORD"):
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            get_settings()
    finally:
        get_settings.cache_clear()
    message = str(exc_info.value)
    assert "FIRMS_MAP_KEY" in message
    assert "POSTGRES_PASSWORD" in message
    assert ".env.example" in message
