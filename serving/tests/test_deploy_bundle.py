"""Paquete de datos de despliegue: compacto, determinista, con atribuciones, y
servible por la API tal cual (lo que Vercel descarga en el build)."""
import json

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from serving.api.main import create_app
from serving.deploy_bundle import WEATHER_FIELDS_SERVED, build_bundle, extract_bundle

from .conftest import IGNITION_LAT, IGNITION_LON, make_settings


def test_bundle_keeps_only_what_predict_reads_with_coarser_weather(processed_dir, tmp_path):
    # campos que /predict no usa no deben viajar
    with rasterio.open(next((processed_dir / "weather").glob("wind_u_*.tif"))) as src:
        profile = src.profile
    zeros = np.zeros((1, profile["height"], profile["width"]), dtype="float32")
    for extra in ("wind_speed", "wind_direction"):
        path = processed_dir / "weather" / f"{extra}_2026-01-10.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(zeros)
    info = build_bundle(processed_dir, tmp_path / "bundle.tar.gz")
    fields = {f.split("_20")[0].removeprefix("weather/")
              for f in info.files if f.startswith("weather/")}
    assert fields == set(WEATHER_FIELDS_SERVED)
    assert "ATTRIBUTION.txt" in info.files and "MANIFEST.json" in info.files
    out = extract_bundle(info.path, tmp_path / "x")
    with rasterio.open(next((out / "weather").glob("wind_u_*.tif"))) as w:
        assert abs(w.res[0] - 2000.0) < 1.0
    with rasterio.open(next((out / "dem").glob("*.tif"))) as dem:
        assert dem.res[0] == 250.0  # estáticas sin cambiar de resolución
    assert "Copernicus" in (out / "ATTRIBUTION.txt").read_text()
    assert json.loads((out / "MANIFEST.json").read_text())["weather_resolution_m"] == 2000.0


def test_bundle_is_deterministic(processed_dir, tmp_path):
    a = build_bundle(processed_dir, tmp_path / "a.tar.gz")
    b = build_bundle(processed_dir, tmp_path / "b.tar.gz")
    assert a.sha256 == b.sha256


def test_extracted_bundle_serves_predictions_and_health(processed_dir, tmp_path, model_loader):
    info = build_bundle(processed_dir, tmp_path / "bundle.tar.gz")
    served = extract_bundle(info.path, tmp_path / "deploy")
    app = create_app(settings=make_settings(served), model_loader=model_loader)
    with TestClient(app) as client:
        assert client.get("/api/health/").status_code == 200
        response = client.post("/predict", json={
            "lat": IGNITION_LAT, "lon": IGNITION_LON, "date": "2026-01-10", "horizon_days": 3})
    assert response.status_code == 200, response.text
    assert response.json()["research_tool"] is True
