"""Test de humo del CLI de ingesta FIRMS: cero red real."""
import re
from pathlib import Path

import responses
from ingestion.cli import app
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "test-key",
    "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x",
    "COPERNICUS_DATASPACE_CLIENT_ID": "x",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "x",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "pyrocast",
    "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def _set_required_env(monkeypatch, tmp_path) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)


@responses.activate
def test_firms_cli_downloads_and_prints_summary(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)

    responses.add(
        responses.GET,
        re.compile(r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*"),
        body=OK_CSV,
        status=200,
    )

    result = runner.invoke(
        app,
        [
            "firms",
            "--start",
            "2026-01-15",
            "--end",
            "2026-01-15",
            "--bbox",
            "-73.7,-39.3,-71.0,-36.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "3 detecciones" in result.output  # 3 detections in the fixture
    saved = list((tmp_path / "data" / "raw" / "firms").glob("**/*.parquet"))
    assert len(saved) == 1


@responses.activate
def test_firms_cli_multi_chunk_range_writes_one_parquet_per_chunk(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)

    # 2026-01-01..2026-01-07 (7 days) chunks into 5+2 -> two requests, two
    # Parquet files — exercises the CLI's own orchestration loop, not just
    # the client's chunking math (already covered in test_firms_client.py).
    responses.add(
        responses.GET,
        re.compile(r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*/5/2026-01-01"),
        body=OK_CSV,
        status=200,
    )
    responses.add(
        responses.GET,
        re.compile(r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*/2/2026-01-06"),
        body=OK_CSV,
        status=200,
    )

    result = runner.invoke(
        app,
        [
            "firms",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-07",
            "--bbox",
            "-73.7,-39.3,-71.0,-36.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "6 detecciones" in result.output  # 3 per chunk x 2 chunks
    saved = list((tmp_path / "data" / "raw" / "firms").glob("**/*.parquet"))
    assert len(saved) == 2


def test_firms_cli_rejects_end_before_start(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)
    result = runner.invoke(app, ["firms", "--start", "2026-01-15", "--end", "2026-01-10"])
    assert result.exit_code != 0
    assert "es anterior" in result.output


def test_firms_cli_rejects_malformed_bbox_number(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        ["firms", "--start", "2026-01-15", "--end", "2026-01-15", "--bbox", "a,b,c,d"],
    )
    assert result.exit_code != 0
    assert "número" in result.output


def test_firms_cli_rejects_unknown_sensor(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        ["firms", "--start", "2026-01-15", "--end", "2026-01-15", "--sensor", "NOT_A_SENSOR"],
    )
    assert result.exit_code != 0
    assert "sensor desconocido" in result.output


@responses.activate
def test_firms_cli_handles_genuinely_empty_response(tmp_path, monkeypatch):
    _set_required_env(monkeypatch, tmp_path)
    empty_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
    )
    responses.add(
        responses.GET,
        re.compile(r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*"),
        body=empty_csv,
        status=200,
    )
    result = runner.invoke(
        app,
        [
            "firms",
            "--start",
            "2026-01-15",
            "--end",
            "2026-01-15",
            "--bbox",
            "-73.7,-39.3,-71.0,-36.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Total: 0 detecciones" in result.output
