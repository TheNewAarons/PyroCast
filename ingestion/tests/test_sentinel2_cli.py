"""Test de humo del CLI de ingesta Sentinel-2: sin red, sin descargas reales."""
from ingestion.cli import app
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def test_sentinel2_cli_wires_settings_and_produces_ndvi(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setattr("ingestion.sentinel2.cli.Sentinel2Client", lambda **kwargs: object())

    def fake_fetch_sentinel2(bbox, year, month, client, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "composite.tif"
        path.write_bytes(b"fake")
        return path

    def fake_compute_and_save_vegetation(
        composite_path, output_dir, target_crs, target_resolution_m
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ndvi.tif"
        path.write_bytes(b"fake-ndvi")
        return path

    monkeypatch.setattr("ingestion.sentinel2.cli.fetch_sentinel2", fake_fetch_sentinel2)
    monkeypatch.setattr(
        "ingestion.sentinel2.cli.compute_and_save_vegetation", fake_compute_and_save_vegetation
    )

    result = runner.invoke(app, ["sentinel2", "--year", "2026", "--month", "1"])
    assert result.exit_code == 0, result.output
    assert "NDVI" in result.output

    from shared.config import get_settings as reread_settings

    settings = reread_settings()
    month_stamped = settings.data_processed_dir / "vegetation" / "ndvi_2026-01.tif"
    assert month_stamped.exists()
    assert not (settings.data_processed_dir / "vegetation" / "ndvi.tif").exists()
    get_settings.cache_clear()


def test_sentinel2_cli_bbox_option_overrides_the_default_study_area(tmp_path, monkeypatch):
    # encontrado contra la API real de Copernicus Data Space (revisión
    # final del 2026-09-29): el bbox completo de la zona de estudio
    # (Biobío+Ñuble+Araucanía) excede el límite de píxeles de un job
    # síncrono de openEO (20000x20000) a la resolución nativa de 10 m de
    # Sentinel-2 -- "ProcessGraphComplexity: Requested spatial extent is
    # too large for a sync job". Sin --bbox, no hay forma de pedir un
    # área más chica (p. ej. el bbox de un evento real) y este comando
    # queda inutilizable para la zona de estudio completa del proyecto.
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setattr("ingestion.sentinel2.cli.Sentinel2Client", lambda **kwargs: object())

    captured_bbox = {}

    def fake_fetch_sentinel2(bbox, year, month, client, cache_dir):
        captured_bbox["value"] = bbox
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "composite.tif"
        path.write_bytes(b"fake")
        return path

    def fake_compute_and_save_vegetation(
        composite_path, output_dir, target_crs, target_resolution_m
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ndvi.tif"
        path.write_bytes(b"fake-ndvi")
        return path

    monkeypatch.setattr("ingestion.sentinel2.cli.fetch_sentinel2", fake_fetch_sentinel2)
    monkeypatch.setattr(
        "ingestion.sentinel2.cli.compute_and_save_vegetation", fake_compute_and_save_vegetation
    )

    result = runner.invoke(
        app,
        [
            "sentinel2", "--year", "2026", "--month", "1",
            "--bbox", "-73.1,-37.0,-72.9,-36.8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_bbox["value"] == (-73.1, -37.0, -72.9, -36.8)
    get_settings.cache_clear()
