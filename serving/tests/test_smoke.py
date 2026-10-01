import pytest
from fastapi.testclient import TestClient
from serving.api.main import app
from shared.config import ConfigurationError, get_settings

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


def test_healthz_returns_ok(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    get_settings.cache_clear()


def test_app_startup_fails_clearly_when_config_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigurationError):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()


def test_healthz_does_not_need_model_or_data(settings, model_loader):
    from serving.api.main import create_app

    with TestClient(create_app(settings=settings, model_loader=model_loader)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
