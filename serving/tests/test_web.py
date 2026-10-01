"""Interfaz web: la página carga, muestra el aviso permanente, y los
endpoints que el HTML declara son alcanzables y responden (el JS los lee
de atributos `data-*` del HTML servido, no de URLs hardcodeadas)."""
import re
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient
from serving.api.main import create_app
from serving.api.schemas import RESEARCH_DISCLAIMER

from .conftest import IGNITION_LAT, IGNITION_LON


class _Attrs(HTMLParser):
    """Recoge los atributos del elemento con id=\"app\" y los tags script/link."""

    def __init__(self) -> None:
        super().__init__()
        self.app: dict[str, str | None] = {}
        self.scripts: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ids.add(d["id"])
        if d.get("id") == "app":
            self.app = d
        if tag == "script":
            self.scripts.append(d)
        if tag == "link":
            self.links.append(d)


@pytest.fixture
def client(settings, model_loader):
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as c:
        yield c


def _parse(html: str) -> _Attrs:
    parser = _Attrs()
    parser.feed(html)
    return parser


def test_main_page_loads_with_permanent_research_notice(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert RESEARCH_DISCLAIMER in html
    # permanente: no es un diálogo ni un elemento ocultable
    notice = re.search(r'<[^>]*id="research-notice"[^>]*>', html)
    assert notice is not None
    assert "hidden" not in notice.group(0) and "dialog" not in notice.group(0)
    assert "<dialog" not in html
    assert "docs/limitations.md" in html


def test_page_uses_leaflet_from_cdn_and_own_static_js_without_build_step(client):
    parsed = _parse(client.get("/").text)
    srcs = [s.get("src") or "" for s in parsed.scripts]
    assert any(s.startswith("https://unpkg.com/leaflet@") for s in srcs)
    own = [s for s in srcs if s.startswith("/static/")]
    assert own == ["/static/app.js"]
    assert any((link.get("href") or "").startswith("https://unpkg.com/leaflet@")
               for link in parsed.links)
    # los elementos que app.js necesita existen
    for element_id in ("map", "date", "horizon", "predict-button", "day-slider", "error-banner",
                       "legend-list", "status"):
        assert element_id in parsed.ids


def test_static_assets_are_served(client):
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/static/app.css").status_code == 200


def test_page_has_noscript_fallback(client):
    assert "<noscript" in client.get("/").text


def test_html_declares_reachable_endpoints_and_predict_works_from_them(client):
    app = _parse(client.get("/").text).app
    predict_url = app["data-predict-url"]
    fires_url = app["data-active-fires-url"]
    assert predict_url == "/predict" and fires_url == "/active-fires"

    # la misma petición que hace app.js
    response = client.post(
        predict_url,
        json={"lat": IGNITION_LAT, "lon": IGNITION_LON, "date": app["data-default-date"],
              "horizon_days": int(app["data-default-horizon"])},
    )
    assert response.status_code == 200, response.text
    assert response.json()["research_tool"] is True


def test_html_exposes_study_area_and_available_weather_range(client):
    app = _parse(client.get("/").text).app
    assert [float(v) for v in app["data-study-bbox"].split(",")] == [-73.7, -39.3, -71.0, -36.5]
    assert app["data-default-date"] == "2026-01-10"
    assert app["data-weather-min"] == "2026-01-10"
    assert app["data-weather-max"] == "2026-01-14"


def test_page_still_loads_when_no_weather_is_processed(settings, model_loader, processed_dir):
    for tif in (processed_dir / "weather").glob("*.tif"):
        tif.unlink()
    with TestClient(create_app(settings=settings, model_loader=model_loader)) as client:
        response = client.get("/")
    assert response.status_code == 200
    app = _parse(response.text).app
    assert app["data-weather-min"] == "" and app["data-weather-max"] == ""
    assert "No hay clima procesado" in response.text


def test_app_js_reads_urls_from_html_and_renders_errors_safely(client):
    js = client.get("/static/app.js").text
    assert "dataset.predictUrl" in js and "dataset.activeFiresUrl" in js
    assert "innerHTML" not in js  # texto del backend siempre con textContent


def test_demo_dataset_generator_produces_data_the_api_can_serve(tmp_path, model_loader):
    """Lo que `make demo` genera sirve para /predict (el README lo promete)."""
    from serving.demo_data import DEMO_DAYS, DEMO_FIRST_DAY, write_demo_dataset

    from .conftest import make_settings

    processed = write_demo_dataset(tmp_path / "demo" / "processed")
    with TestClient(create_app(settings=make_settings(processed), model_loader=model_loader)) as c:
        body = c.post("/predict", json={
            "lat": IGNITION_LAT, "lon": IGNITION_LON, "date": DEMO_FIRST_DAY.isoformat(),
            "horizon_days": DEMO_DAYS - 1,
        }).json()
        page = c.get("/").text
    assert body["research_tool"] is True and len(body["days"]) == DEMO_DAYS - 1
    assert DEMO_FIRST_DAY.isoformat() in page
