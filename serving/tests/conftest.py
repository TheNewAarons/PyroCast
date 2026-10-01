"""Fixtures compartidas de serving/: un directorio `data/processed`
sintético (rasters chicos, EPSG:32719) con la misma convención de nombres
que produce la ingesta real -- ninguna llamada a red."""
import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from models.cellular_automata.model import CellularAutomatonModel
from serving.api.model_registry import LoadedModel
from serving.demo_data import (
    DEMO_DAYS,
    DEMO_FIRST_DAY,
    DEMO_LAT,
    DEMO_LON,
    write_demo_dataset,
)
from shared.config import Settings

IGNITION_LAT = DEMO_LAT
IGNITION_LON = DEMO_LON
WEATHER_DAYS = [DEMO_FIRST_DAY + dt.timedelta(days=i) for i in range(DEMO_DAYS)]


def make_settings(processed: Path) -> Settings:
    return Settings(
        firms_map_key="test-firms-key", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    return write_demo_dataset(tmp_path / "processed")


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
