"""/active-fires con un cliente FIRMS falso (nunca la red real)."""
import pytest
from fastapi.testclient import TestClient
from ingestion.firms.client import FirmsApiError
from serving.api.main import create_app

CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
    "confidence,version,bright_ti5,frp,daynight\n"
    "-37.5,-72.5,340.1,0.4,0.4,2026-01-12,517,N,VIIRS,n,2.0NRT,300.0,4.2,D\n"
    "-38.0,-72.0,330.0,0.4,0.4,2026-01-12,1340,N,VIIRS,h,2.0NRT,300.0,,N\n"
)


class FakeFirms:
    def __init__(self, body: str = CSV, error: Exception | None = None) -> None:
        self.body, self.error, self.calls = body, error, []

    def fetch_area_csv(self, bbox, sensor, day_range, date=None):
        self.calls.append((bbox, sensor, day_range))
        if self.error:
            raise self.error
        return self.body


def _client(settings, firms):
    return TestClient(create_app(settings=settings, firms_client=firms))


def test_active_fires_returns_point_geojson_inside_study_bbox(settings):
    firms = FakeFirms()
    with _client(settings, firms) as client:
        response = client.get("/active-fires", params={"days": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection" and len(body["features"]) == 2
    first = body["features"][0]
    assert first["geometry"] == {"type": "Point", "coordinates": [-72.5, -37.5]}
    assert first["properties"]["detected_at"] == "2026-01-12T05:17:00Z"
    assert first["properties"]["frp"] == 4.2
    assert body["features"][1]["properties"]["frp"] is None
    assert body["research_tool"] is True
    assert firms.calls == [((-73.7, -39.3, -71.0, -36.5), "VIIRS_SNPP_NRT", 2)]


def test_active_fires_is_cached_to_protect_firms_rate_limit(settings):
    firms = FakeFirms()
    with _client(settings, firms) as client:
        client.get("/active-fires", params={"days": 2})
        second = client.get("/active-fires", params={"days": 2})
        client.get("/active-fires", params={"days": 3})
    assert second.json()["cached"] is True
    assert len(firms.calls) == 2  # days=2 una vez, days=3 otra


def test_active_fires_upstream_failure_is_502_not_empty_list(settings):
    with _client(settings, FakeFirms(error=FirmsApiError("clave inválida"))) as client:
        response = client.get("/active-fires")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "firms_unavailable"
    assert "features" not in response.json()


@pytest.mark.parametrize("days", [0, 6])
def test_active_fires_rejects_days_outside_firms_limit(settings, days):
    with _client(settings, FakeFirms()) as client:
        assert client.get("/active-fires", params={"days": days}).status_code == 422
