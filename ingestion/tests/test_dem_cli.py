"""Test de humo del CLI de ingesta DEM: sin red, sin descargas reales."""
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


def test_dem_cli_wires_settings_into_build_dem_and_compute_and_save_terrain(
    tmp_path, monkeypatch
):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    calls: dict[str, object] = {}

    def fake_build_dem(bbox, resolution_m, crs, raw_tiles_dir, cache_dir):
        calls["build_dem"] = (bbox, resolution_m, crs, raw_tiles_dir, cache_dir)
        dem_path = cache_dir / "dem.tif"
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        dem_path.write_bytes(b"fake-dem")
        return dem_path

    def fake_compute_and_save_terrain(dem_path, output_dir):
        calls["compute_and_save_terrain"] = (dem_path, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slope_path = output_dir / "slope_deg.tif"
        aspect_path = output_dir / "aspect_deg.tif"
        slope_path.write_bytes(b"fake-slope")
        aspect_path.write_bytes(b"fake-aspect")
        return slope_path, aspect_path

    monkeypatch.setattr("ingestion.dem.cli.build_dem", fake_build_dem)
    monkeypatch.setattr(
        "ingestion.dem.cli.compute_and_save_terrain", fake_compute_and_save_terrain
    )

    result = runner.invoke(app, ["dem"])

    assert result.exit_code == 0, result.output
    bbox, resolution_m, crs, raw_tiles_dir, cache_dir = calls["build_dem"]
    assert bbox == settings.study_area_bbox
    assert resolution_m == settings.spatial_resolution_m
    assert crs == settings.crs
    assert raw_tiles_dir == settings.data_raw_dir / "dem"
    assert cache_dir == settings.data_processed_dir / "dem"

    dem_path, output_dir = calls["compute_and_save_terrain"]
    assert output_dir == settings.data_processed_dir / "terrain"
    assert isinstance(dem_path, Path)
    get_settings.cache_clear()
