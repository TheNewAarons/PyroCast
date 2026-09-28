"""Test de humo del CLI `pyrocast-models backtest`: sin red, sin
Postgres real, sin Zarr real -- eventos de fixture inyectados, pero el
backtest y la escritura de bench/results corren de verdad."""
import json

import numpy as np
import xarray as xr
from models.cli import app
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}

_CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


def _fixture_event(event_id: int) -> xr.DataArray:
    size, n_days = 11, 3
    data = np.zeros((n_days, len(_CHANNEL_ORDER), size, size), dtype="float32")
    data[:, _CHANNEL_ORDER.index("fuel_type"), :, :] = 1.0
    data[0, _CHANNEL_ORDER.index("fire_mask"), size // 2, size // 2] = 1.0
    return xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(_CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_backtest_cli_writes_bench_results_json(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    fixture_events = [_fixture_event(1), _fixture_event(2)]
    monkeypatch.setattr(
        "models.cli.load_test_events", lambda dataset_dir: fixture_events
    )
    monkeypatch.setattr("models.cli.persist_backtest_run", lambda **kwargs: 1)

    result = runner.invoke(app, ["backtest", "--n-bootstrap", "50"])
    assert result.exit_code == 0, result.output

    results_path = tmp_path / "bench" / "results" / "baseline.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert payload["model_name"] == "cellular_automata"
    assert len(payload["per_event"]) == 2
    assert set(payload["aggregate"]) == {"iou", "dice", "brier", "ece"}
    get_settings.cache_clear()
