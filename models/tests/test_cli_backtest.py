"""Test de humo del CLI `pyrocast-models backtest`: sin red, sin
Postgres real, sin Zarr real -- eventos de fixture inyectados, pero el
backtest y la escritura de bench/results corren de verdad."""
import json
from pathlib import Path

import numpy as np
import pytest
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
    persisted_calls = []
    monkeypatch.setattr(
        "models.cli.persist_backtest_run",
        lambda **kwargs: persisted_calls.append(kwargs) or [1, 2],
    )

    result = runner.invoke(app, ["backtest", "--n-bootstrap", "50"])
    assert result.exit_code == 0, result.output

    results_path = tmp_path / "bench" / "results" / "baseline.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert payload["model_name"] == "cellular_automata"
    assert len(payload["per_event"]) == 2
    assert set(payload["aggregate"]) == {"iou", "dice", "brier", "ece"}

    # el registro debe describir la corrida que produjo estos números,
    # no un dict copiado a mano que puede desincronizarse de los
    # defaults reales de SpreadParameters (encontrado en la revisión
    # final del 2026-09-28).
    assert payload["config"]["base_spread_prob"] == pytest.approx(0.3)
    assert "fuel_flammability" in payload["config"]
    assert payload["seed"] == 42
    assert payload["n_bootstrap"] == 50
    assert "ece_bins" in payload
    assert "confidence" in payload

    # una sola llamada, atómica, con los 2 eventos juntos -- no una
    # llamada por evento.
    assert len(persisted_calls) == 1
    assert len(persisted_calls[0]["per_event"]) == 2

    get_settings.cache_clear()


def test_load_test_events_trims_leading_padding_days_with_no_fire(tmp_path, monkeypatch):
    # padded_days_for_event (features/dataset/pipeline.py) antepone
    # DEFAULT_PRE_EVENT_PADDING_DAYS=5 días SIN fuego antes del primer
    # día real del evento -- el día 0 del tensor resultante NO tiene
    # ninguna celda en llamas. Pero tanto CalibratedUNet.predict como
    # CellularAutomatonModel.predict asumen que el día 0 es "el ancla
    # conocida" (el estado ACTUAL del fuego del que hay que propagar):
    # sin recortar el padding, ambos modelos parten de "nada ardiendo"
    # y no tienen forma de anticipar la ignición real que recién ocurre
    # más adelante -- un desajuste del PROTOCOLO de evaluación, no una
    # limitación de ningún modelo en particular. Verificado contra
    # datos reales de docs/backtest-2026.md (event_2582836092: 5 días
    # de padding sin fuego, IoU=Dice=0.0 para AMBOS modelos).
    from models.cli import load_test_events

    monkeypatch.chdir(tmp_path)
    size, n_days = 6, 7
    data = np.zeros((n_days, len(_CHANNEL_ORDER), size, size), dtype="float32")
    data[:, _CHANNEL_ORDER.index("fuel_type"), :, :] = 1.0
    fire_idx = _CHANNEL_ORDER.index("fire_mask")
    # días 0-1: padding, sin fuego. día 2 en adelante: fuego real.
    data[2:, fire_idx, size // 2, size // 2] = 1.0
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(_CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"event_id": 1, "resolution_m": 100.0},
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    event.to_dataset().to_zarr(dataset_dir / "event_0001.zarr", mode="w")
    (dataset_dir / "splits.json").write_text(json.dumps({"test": [1]}))

    events = load_test_events(dataset_dir)
    assert len(events) == 1
    trimmed = events[0]
    assert trimmed.sizes["day"] == 5  # 7 días - 2 de padding sin fuego
    assert trimmed.values[0, fire_idx].sum() > 0  # el nuevo día 0 SÍ tiene fuego


def test_backtest_cli_help_does_not_claim_the_default_ca_model_is_calibrated(monkeypatch):
    # el default (cellular_automata) sigue sin calibrar contra
    # incendios reales -- ver docs/limitations.md. --model unet SÍ es
    # calibrado (P11), y el help lo dice; esta aserción es específica
    # a la frase que describe el autómata celular, no al texto entero.
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0, result.output
    assert "sin calibrar" in result.output.lower()


def test_backtest_cli_records_the_exact_command_and_git_commit(tmp_path, monkeypatch):
    # docs/backtest-2026.md exige que los resultados en bench/results/
    # queden reproducibles por otra persona: el comando exacto usado y
    # el commit de git con el que se generaron.
    import subprocess

    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    fixture_events = [_fixture_event(1), _fixture_event(2)]
    monkeypatch.setattr("models.cli.load_test_events", lambda dataset_dir: fixture_events)
    monkeypatch.setattr("models.cli.persist_backtest_run", lambda **kwargs: [1, 2])

    result = runner.invoke(app, ["backtest", "--n-bootstrap", "50"])
    assert result.exit_code == 0, result.output

    payload = json.loads((tmp_path / "bench" / "results" / "baseline.json").read_text())
    assert "backtest" in payload["command"]
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert payload["git_commit"] == expected_commit
    get_settings.cache_clear()


def test_backtest_cli_supports_the_unet_model(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    fixture_events = [_fixture_event(1), _fixture_event(2)]
    monkeypatch.setattr("models.cli.load_test_events", lambda dataset_dir: fixture_events)
    monkeypatch.setattr("models.cli.persist_backtest_run", lambda **kwargs: [1, 2])

    class _FakeCalibratedUNet:
        def __init__(self, checkpoint_path, calibration_path=None):
            self.checkpoint_path = checkpoint_path
            self.calibration_path = calibration_path

        def predict(self, event):
            return np.zeros(
                (event.sizes["day"], event.sizes["y"], event.sizes["x"]), dtype="float64"
            )

    monkeypatch.setattr("models.cli.CalibratedUNet", _FakeCalibratedUNet)

    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"fake")

    result = runner.invoke(
        app,
        [
            "backtest", "--model", "unet", "--checkpoint", str(checkpoint_path),
            "--n-bootstrap", "50",
        ],
    )
    assert result.exit_code == 0, result.output

    results_path = tmp_path / "bench" / "results" / "unet.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert payload["model_name"] == "unet"
    assert len(payload["per_event"]) == 2
    get_settings.cache_clear()


def test_backtest_cli_unet_model_requires_a_checkpoint(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    result = runner.invoke(app, ["backtest", "--model", "unet"])
    assert result.exit_code == 1
    assert "checkpoint" in result.output.lower()


def test_backtest_survives_an_unavailable_database_and_says_so(tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError

    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    from shared.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(
        "models.cli.load_test_events", lambda dataset_dir: [_fixture_event(1), _fixture_event(2)]
    )

    def _down(**kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr("models.cli.persist_backtest_run", _down)
    result = runner.invoke(app, ["backtest", "--n-bootstrap", "10"])
    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "no se pudo persistir en PostGIS" in result.output
    assert (tmp_path / "bench" / "results" / "baseline.json").exists()
