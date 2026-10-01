"""/predict end-to-end con rasters de fixture (pipeline de features
real, autómata celular real), errores explícitos y cacheo."""
import datetime as dt

import pytest
from fastapi.testclient import TestClient
from serving.api.main import create_app
from serving.api.model_registry import LoadedModel

from .conftest import IGNITION_LAT, IGNITION_LON, SpyModel

BODY = {"lat": IGNITION_LAT, "lon": IGNITION_LON, "date": "2026-01-10", "horizon_days": 3}


@pytest.fixture
def client(settings, model_loader):
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as c:
        yield c


def test_predict_returns_geojson_probabilities_per_day_with_research_notice(client):
    response = client.post("/predict", json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["type"] == "FeatureCollection"
    assert body["research_tool"] is True
    assert "limitations" in body["limitations"]  # referencia a docs/limitations.md
    assert "CONAF/SENAPRED" in body["disclaimer"]
    assert body["model"] == {"name": "cellular_automata", "calibrated": False}
    assert [d["day"] for d in body["days"]] == [1, 2, 3]
    assert [d["date"] for d in body["days"]] == ["2026-01-11", "2026-01-12", "2026-01-13"]
    assert body["cached"] is False

    features = body["features"]
    assert len(features) > 9
    for feature in features:
        assert feature["geometry"]["type"] == "Polygon"
        ring = feature["geometry"]["coordinates"][0]
        assert len(ring) == 5 and ring[0] == ring[-1]
        probs = feature["properties"]["probability_by_day"]
        assert len(probs) == 3
        assert all(0.0 <= p <= 1.0 for p in probs)
        # acumulada: nunca decrece con los días
        assert probs == sorted(probs)
        lon, lat = ring[0]
        assert -73.7 < lon < -71.0 and -39.3 < lat < -36.5

    # la celda de ignición ya ardió desde el día 1; hay propagación a vecinos
    peak_day1 = max(f["properties"]["probability_by_day"][0] for f in features)
    assert peak_day1 >= 0.99
    assert sum(1 for f in features if f["properties"]["probability_by_day"][2] > 0.0) > 1


def test_predict_missing_weather_date_returns_clear_error_not_a_number(client):
    response = client.post("/predict", json={**BODY, "date": "2026-02-20"})
    assert response.status_code == 422
    body = response.json()
    assert "features" not in body
    assert body["error"]["code"] == "weather_unavailable"
    assert "2026-02-20" in body["error"]["message"]
    assert body["error"]["details"]["available_range"] == ["2026-01-10", "2026-01-14"]
    assert "2026-02-20" in body["error"]["details"]["missing_days"]


def test_predict_horizon_beyond_available_weather_names_missing_days(client):
    response = client.post("/predict", json={**BODY, "date": "2026-01-13", "horizon_days": 5})
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert details["missing_days"] == ["2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18"]


def test_predict_outside_study_area_is_rejected(client):
    response = client.post("/predict", json={**BODY, "lat": 40.0, "lon": -3.7})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "outside_study_area"


@pytest.mark.parametrize("patch", [{"horizon_days": 0}, {"horizon_days": 99}, {"lat": 120.0},
                                   {"date": "no-es-fecha"}])
def test_predict_rejects_invalid_request(client, patch):
    assert client.post("/predict", json={**BODY, **patch}).status_code == 422


def test_predict_without_static_layers_returns_503(settings, model_loader, processed_dir):
    for tif in (processed_dir / "dem").glob("*.tif"):
        tif.unlink()
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as client:
        response = client.post("/predict", json=BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "static_layers_unavailable"


def test_predict_ignition_outside_raster_coverage_is_an_error(settings, model_loader):
    # dentro del bbox de estudio, pero fuera de los rasters de fixture
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as client:
        response = client.post("/predict", json={**BODY, "lat": -38.9, "lon": -71.5})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "terrain_coverage_unavailable"


def test_second_identical_request_is_served_from_cache(client, spy_model: SpyModel):
    first = client.post("/predict", json=BODY).json()
    assert spy_model.calls == 1
    second = client.post("/predict", json=BODY).json()
    assert spy_model.calls == 1  # no recalculó
    assert first["cached"] is False and second["cached"] is True
    assert second["features"] == first["features"]
    assert second["research_tool"] is True


def test_nearby_point_in_same_grid_cell_hits_cache(client, spy_model: SpyModel):
    client.post("/predict", json=BODY)
    client.post("/predict", json={**BODY, "lat": IGNITION_LAT + 0.0002, "lon": IGNITION_LON})
    assert spy_model.calls == 1


@pytest.mark.parametrize("patch", [{"horizon_days": 2}, {"date": "2026-01-11"},
                                   {"lat": IGNITION_LAT + 0.05}])
def test_different_date_horizon_or_cell_recomputes(client, spy_model: SpyModel, patch):
    client.post("/predict", json=BODY)
    assert client.post("/predict", json={**BODY, **patch}).status_code == 200
    assert spy_model.calls == 2


def test_failed_requests_are_not_cached(client, spy_model: SpyModel):
    bad = {**BODY, "date": "2026-02-20"}
    client.post("/predict", json=bad)
    client.post("/predict", json=bad)
    assert spy_model.calls == 0


def test_model_is_loaded_once_at_startup_not_per_request(settings, spy_model):
    loads = []

    def loader() -> LoadedModel:
        loads.append(1)
        return LoadedModel(
            name="cellular_automata", model=spy_model, calibrated=False,
            required_channels=("elevation", "wind_u", "wind_v", "fuel_type"),
        )

    with TestClient(create_app(settings=settings, model_loader=loader)) as client:
        assert len(loads) == 1
        for day in ("2026-01-10", "2026-01-11"):
            client.post("/predict", json={**BODY, "date": day})
    assert len(loads) == 1


def test_missing_optional_channel_is_reported_as_warning_not_hidden(
    settings, model_loader, processed_dir
):
    for day in (dt.date(2026, 1, 10) + dt.timedelta(days=i) for i in range(5)):
        (processed_dir / "weather" / f"precipitation_{day.isoformat()}.tif").unlink()
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as client:
        response = client.post("/predict", json=BODY)
    assert response.status_code == 200
    warnings = " ".join(response.json()["warnings"])
    assert "precipitation" in warnings


def test_default_model_is_cellular_automata(settings):
    with TestClient(create_app(settings=settings)) as client:
        body = client.post("/predict", json=BODY).json()
    assert body["model"]["name"] == "cellular_automata"
