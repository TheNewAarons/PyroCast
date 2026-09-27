"""Tests de ensamblado del tensor (día, canal, alto, ancho) y su
persistencia en Zarr -- canales sintéticos de fixture."""
import datetime as dt

import numpy as np
import xarray as xr
from features.dataset.assemble import (
    CHANNEL_ORDER,
    EventChannels,
    assemble_event_tensor,
    save_event_to_zarr,
)
from features.grid.grid import WorkGrid
from rasterio.transform import from_origin

_GRID = WorkGrid(
    crs="EPSG:32719", transform=from_origin(500000, 6000000, 250, 250),
    width=4, height=4, resolution_m=250.0,
)


def _fixture_channels() -> EventChannels:
    height, width = 4, 4
    days = (dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 3))
    static = {
        "elevation": np.full((height, width), 100.0, dtype="float32"),
        "slope_deg": np.full((height, width), 5.0, dtype="float32"),
        "aspect_deg": np.full((height, width), 180.0, dtype="float32"),
        "fuel_type": np.full((height, width), 3.0, dtype="float32"),
    }
    dynamic_names = (
        "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation", "ndvi"
    )
    dynamic = {
        name: {
            day: np.full((height, width), float(i), dtype="float32")
            for i, day in enumerate(days)
        }
        for name in dynamic_names
    }
    dynamic["fire_mask"] = {
        days[0]: np.zeros((height, width), dtype="float32"),
        days[1]: np.zeros((height, width), dtype="float32"),
        days[2]: np.ones((height, width), dtype="float32"),
    }
    return EventChannels(days=days, static=static, dynamic=dynamic)


def test_assemble_event_tensor_has_expected_shape_and_channel_order():
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=42)
    assert isinstance(tensor, xr.DataArray)
    assert tensor.dims == ("day", "channel", "y", "x")
    assert tensor.shape == (3, len(CHANNEL_ORDER), 4, 4)
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert list(tensor.coords["day"].values) == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_assemble_event_tensor_static_channels_repeat_identically_across_days():
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=42)
    elevation_idx = CHANNEL_ORDER.index("elevation")
    elevation_across_days = tensor.values[:, elevation_idx, :, :]
    assert np.all(elevation_across_days == 100.0)


def test_assemble_event_tensor_dynamic_channels_vary_by_day():
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=42)
    wind_u_idx = CHANNEL_ORDER.index("wind_u")
    assert np.all(tensor.values[0, wind_u_idx, :, :] == 0.0)
    assert np.all(tensor.values[1, wind_u_idx, :, :] == 1.0)
    assert np.all(tensor.values[2, wind_u_idx, :, :] == 2.0)


def test_assemble_event_tensor_fire_mask_channel_matches_input():
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=42)
    fire_mask_idx = CHANNEL_ORDER.index("fire_mask")
    assert np.all(tensor.values[0, fire_mask_idx, :, :] == 0.0)
    assert np.all(tensor.values[2, fire_mask_idx, :, :] == 1.0)


def test_assemble_event_tensor_is_georeferenced():
    # Antes del fix, el tensor no tenía coords y/x ni CRS/transform/id de
    # evento -- no había forma de recuperar la ubicación real de un
    # píxel, ni de qué evento era, a partir solo del Zarr. Verificado en
    # la revisión final del 2026-09-27.
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=42)
    west, south, east, north = _GRID.bounds
    assert west < float(tensor.coords["x"].min()) < east
    assert south < float(tensor.coords["y"].min()) < north
    assert tensor.attrs["crs"] == "EPSG:32719"
    assert tensor.attrs["resolution_m"] == 250.0
    assert tensor.attrs["event_id"] == 42
    assert tuple(tensor.attrs["transform"]) == tuple(_GRID.transform)[:6]


def test_save_event_to_zarr_roundtrip_preserves_values(tmp_path):
    tensor = assemble_event_tensor(_fixture_channels(), _GRID, event_id=7)
    zarr_path = save_event_to_zarr(tensor, tmp_path, event_id=7)
    assert zarr_path.exists()
    assert zarr_path.name == "event_0007.zarr"

    reopened = xr.open_zarr(zarr_path)
    reopened_tensor = reopened["fire_event_tensor"]
    assert reopened_tensor.shape == tensor.shape
    assert np.allclose(reopened_tensor.values, tensor.values)
    assert list(reopened_tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert reopened_tensor.attrs["event_id"] == 7
    assert reopened_tensor.attrs["crs"] == "EPSG:32719"
