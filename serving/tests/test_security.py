"""Configuración segura por defecto de la API: sin DEBUG, docs solo en
desarrollo, CORS explícito (ninguno por defecto), cabeceras de seguridad."""
import pytest
from fastapi.testclient import TestClient
from serving.api.main import create_app

from .conftest import make_settings


@pytest.fixture
def make_client(processed_dir, model_loader):
    def _make(**overrides):
        settings = make_settings(processed_dir).model_copy(update=overrides)
        return TestClient(create_app(settings=settings, model_loader=model_loader))

    return _make


def test_debug_is_off_and_docs_are_hidden_in_production_by_default(make_client):
    with make_client() as client:
        assert client.app.debug is False
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404


def test_docs_are_available_in_development(make_client):
    with make_client(environment="development") as client:
        assert client.get("/docs").status_code == 200
        assert client.app.debug is False  # nunca debug, ni en desarrollo


def test_no_cors_headers_by_default(make_client):
    with make_client() as client:
        response = client.get("/healthz", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_explicit_origins_and_methods(make_client):
    with make_client(cors_allow_origins=["https://app.example"]) as client:
        ok = client.get("/healthz", headers={"Origin": "https://app.example"})
        assert ok.headers["access-control-allow-origin"] == "https://app.example"
        assert "access-control-allow-credentials" not in ok.headers
        bad = client.get("/healthz", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in bad.headers
        pre = client.options("/predict", headers={
            "Origin": "https://app.example", "Access-Control-Request-Method": "DELETE"})
        assert pre.status_code == 400  # DELETE no permitido


def test_wildcard_cors_origin_is_refused_at_startup(make_client):
    with pytest.raises(ValueError, match="CORS"):
        make_client(cors_allow_origins=["*"])


@pytest.mark.parametrize("path", ["/", "/healthz"])
def test_security_headers_present(make_client, path):
    with make_client() as client:
        headers = client.get(path).headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "permissions-policy" in headers


def test_every_typer_app_hides_locals_in_tracebacks():
    from features.cli import app as features_app
    from ingestion.cli import app as ingestion_app
    from models.cli import app as models_app
    from models.deep.calibration import app as calibrate_app
    from models.deep.train import app as train_app

    for app in (features_app, ingestion_app, models_app, calibrate_app, train_app):
        assert app.pretty_exceptions_show_locals is False
