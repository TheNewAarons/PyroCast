"""Test de humo del CLI de ingesta ERA5: sin red, sin descargas reales."""
from pathlib import Path

from ingestion.cli import app
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x",
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


def test_era5_cli_wires_settings_into_fetch_and_derive(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    calls: dict[str, object] = {}

    monkeypatch.setattr("ingestion.era5.cli.Era5Client", lambda **kwargs: object())

    def fake_fetch_daily_era5(bbox, start, end, era5_client, raw_dir, cache_dir, timeout_seconds):
        calls["fetch_daily_era5"] = (bbox, start, end, raw_dir, cache_dir, timeout_seconds)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "daily.nc"
        path.write_bytes(b"fake")
        return path

    def fake_compute_and_save_weather(daily_nc_path, output_dir, target_crs, target_resolution_m):
        calls["compute_and_save_weather"] = (
            daily_nc_path, output_dir, target_crs, target_resolution_m
        )
        return {"wind_speed": {"2026-01-15": Path("/tmp/fake.tif")}}

    monkeypatch.setattr("ingestion.era5.cli.fetch_daily_era5", fake_fetch_daily_era5)
    monkeypatch.setattr(
        "ingestion.era5.cli.compute_and_save_weather", fake_compute_and_save_weather
    )

    result = runner.invoke(app, ["era5", "--start", "2026-01-15", "--end", "2026-01-16"])

    assert result.exit_code == 0, result.output
    bbox, start, end, raw_dir, cache_dir, timeout_seconds = calls["fetch_daily_era5"]
    assert bbox == settings.study_area_bbox
    assert raw_dir == settings.data_raw_dir / "era5"
    assert cache_dir == settings.data_processed_dir / "era5"

    daily_nc_path, output_dir, target_crs, target_resolution_m = calls["compute_and_save_weather"]
    assert output_dir == settings.data_processed_dir / "weather"
    assert target_crs == settings.crs
    assert target_resolution_m == settings.spatial_resolution_m
    get_settings.cache_clear()
