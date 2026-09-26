"""Test de humo del CLI de ingesta WorldCover: sin red, sin descargas reales."""
import numpy as np
import rasterio
from ingestion.cli import app
from rasterio.transform import from_origin
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def test_worldcover_cli_wires_settings_and_produces_fuel_type(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    def fake_build_worldcover(bbox, resolution_m, crs, raw_tiles_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "worldcover.tif"
        transform = from_origin(500000, 5800000, 250, 250)
        data = np.full((4, 4), 10, dtype="uint8")
        with rasterio.open(
            path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="uint8", crs=crs, transform=transform, nodata=0.0,
        ) as dst:
            dst.write(data, 1)
        return path

    monkeypatch.setattr("ingestion.worldcover.cli.build_worldcover", fake_build_worldcover)

    result = runner.invoke(app, ["worldcover"])
    assert result.exit_code == 0, result.output
    assert "Tipo de combustible" in result.output
    get_settings.cache_clear()
