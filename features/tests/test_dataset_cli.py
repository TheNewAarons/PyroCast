"""Test de humo del CLI `pyrocast-features build-dataset`: sin red, sin
Postgres real -- cada colaborador externo está monkeypatcheado con datos
de fixture, pero el ensamblado, el Zarr y el split corren de verdad."""
import datetime as dt
import json

import numpy as np
import rasterio
import xarray as xr
from features.cli import app
from features.dataset.assemble import CHANNEL_ORDER
from rasterio.transform import from_origin
from shared.schemas import FireDetection
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_build_dataset_cli_produces_at_least_one_zarr_event(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fixture_detections = [_det(-38.0, -72.5, at)]
    monkeypatch.setattr(
        "features.cli.load_firms_detections", lambda base_dir, start, end: fixture_detections
    )

    def fake_resolve_event_sources(days, settings_arg):
        processed = settings_arg.data_processed_dir
        size = 20
        transform = from_origin(190000, 5791000, 250, 250)
        for subdir, name in [
            ("dem", "dem_x.tif"), ("terrain", "slope_deg.tif"), ("terrain", "aspect_deg.tif"),
            ("vegetation", "fuel_type.tif"), ("vegetation", "ndvi_2026-01.tif"),
        ]:
            path = processed / subdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                path, "w", driver="GTiff", height=size, width=size, count=1,
                dtype="float32", crs="EPSG:32719", transform=transform, nodata=-9999.0,
            ) as dst:
                dst.write(np.ones((size, size), dtype="float32"), 1)

        from features.dataset.pipeline import EventSources

        return EventSources(
            elevation_path=processed / "dem" / "dem_x.tif",
            slope_path=processed / "terrain" / "slope_deg.tif",
            aspect_path=processed / "terrain" / "aspect_deg.tif",
            fuel_type_path=processed / "vegetation" / "fuel_type.tif",
            ndvi_paths_by_month={"2026-01": processed / "vegetation" / "ndvi_2026-01.tif"},
            weather_paths_by_day={},
        )

    monkeypatch.setattr("features.cli.resolve_event_sources", fake_resolve_event_sources)
    persisted_calls: list[dict] = []
    monkeypatch.setattr(
        "features.cli.persist_fire_event_metadata",
        lambda **kwargs: persisted_calls.append(kwargs) or 1,
    )

    result = runner.invoke(app, ["build-dataset", "--start", "2026-01-10", "--end", "2026-01-15"])
    assert result.exit_code == 0, result.output

    dataset_dir = settings.data_processed_dir / "dataset"
    zarr_dirs = list(dataset_dir.glob("event_*.zarr"))
    assert len(zarr_dirs) >= 1

    # abrir el Zarr real y verificar su forma/orden de canales -- antes,
    # el test solo comprobaba que EXISTIERA algún directorio "event_*.zarr",
    # lo que un `split_events` roto o un tensor vacío también satisfaría
    # (encontrado en la revisión final del 2026-09-27).
    reopened = xr.open_zarr(zarr_dirs[0])
    tensor = reopened["fire_event_tensor"]
    assert tensor.dims == ("day", "channel", "y", "x")
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert tensor.shape[0] > 0

    # la metadata persistida debe llevar el mismo event_id que nombra el
    # Zarr.
    assert len(persisted_calls) == len(zarr_dirs)
    zarr_event_id = int(zarr_dirs[0].stem.removeprefix("event_"))
    assert any(call["event_id"] == zarr_event_id for call in persisted_calls)

    splits_path = dataset_dir / "splits.json"
    assert splits_path.exists()
    splits = json.loads(splits_path.read_text())
    assert set(splits) == {"train", "val", "test"}
    all_split_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_split_ids) == sorted(call["event_id"] for call in persisted_calls)
    get_settings.cache_clear()
