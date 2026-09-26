"""Test de humo del CLI de ingesta FIRMS: cero red real."""
import re
from pathlib import Path

import responses
from ingestion.cli import app
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()

runner = CliRunner()


@responses.activate
def test_firms_cli_downloads_and_prints_summary(tmp_path, monkeypatch):
    for key, value in {
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
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

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
    assert "3" in result.output  # 3 detections in the fixture
    saved = list((tmp_path / "data" / "raw" / "firms").glob("**/*.parquet"))
    assert len(saved) == 1
