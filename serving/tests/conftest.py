"""Fixtures compartidas de serving/: un directorio `data/processed`
sintético (rasters chicos, EPSG:32719) con la misma convención de nombres
que produce la ingesta real -- ninguna llamada a red."""
import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from models.cellular_automata.model import CellularAutomatonModel
from pyproj import Transformer
from rasterio.transform import from_origin
from serving.api.model_registry import LoadedModel
from shared.config import Settings

IGNITION_LAT = -37.5
IGNITION_LON = -72.5
WEATHER_DAYS = [dt.date(2026, 1, 10) + dt.timedelta(days=i) for i in range(5)]
_SIZE = 80  # celdas de 250 m -> 20 km de lado, centrado en la ignición
_WEATHER_FIELDS = ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation")


def _write_tif(path: Path, value: float, transform, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((_SIZE, _SIZE), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=_SIZE, width=_SIZE, count=1,
        dtype="float32", crs="EPSG:32719", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def make_settings(processed: Path) -> Settings:
    return Settings(
        firms_map_key="test-firms-key", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(
        IGNITION_LON, IGNITION_LAT
    )
    half = _SIZE * 250 / 2
    transform = from_origin(x - half, y + half, 250, 250)
    _write_tif(processed / "dem" / "dem_fixture.tif", 100.0, transform)
    _write_tif(processed / "terrain" / "slope_deg.tif", 0.0, transform)
    _write_tif(processed / "terrain" / "aspect_deg.tif", -1.0, transform)
    _write_tif(processed / "vegetation" / "fuel_type.tif", 1.0, transform)  # 1 = pastizal
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, transform)
    values = {"wind_u": 3.0, "wind_v": 0.0, "temperature": 25.0,
              "relative_humidity": 30.0, "precipitation": 0.0}
    for day in WEATHER_DAYS:
        for field in _WEATHER_FIELDS:
            _write_tif(
                processed / "weather" / f"{field}_{day.isoformat()}.tif", values[field], transform
            )
    return processed


@pytest.fixture
def settings(processed_dir: Path) -> Settings:
    return make_settings(processed_dir)


class SpyModel:
    """Envuelve el autómata celular real y cuenta las llamadas a
    `predict` -- para verificar el cacheo sin mockear el cálculo."""

    def __init__(self) -> None:
        self._inner = CellularAutomatonModel(seed=42)
        self.calls = 0

    def predict(self, event: xr.DataArray) -> np.ndarray:
        self.calls += 1
        return self._inner.predict(event)


@pytest.fixture
def spy_model() -> SpyModel:
    return SpyModel()


@pytest.fixture
def model_loader(spy_model: SpyModel) -> Callable[[], LoadedModel]:
    def _load() -> LoadedModel:
        return LoadedModel(
            name="cellular_automata", model=spy_model, calibrated=False,
            required_channels=("elevation", "wind_u", "wind_v", "fuel_type"),
        )
    return _load
