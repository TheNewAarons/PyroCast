"""Tests del split train/val/test reproducible POR EVENTO."""
import datetime as dt

import numpy as np
import pytest
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from features.dataset.split import (
    EventFootprint,
    find_split_leakage,
    footprint_from_tensor,
    group_events,
    split_events,
    split_events_grouped,
)
from pyproj import Transformer


def test_split_events_no_event_appears_in_two_splits():
    event_ids = list(range(20))
    splits = split_events(event_ids, seed=1)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(event_ids)


def test_split_events_is_reproducible_with_the_same_seed():
    event_ids = list(range(50))
    first = split_events(event_ids, seed=7)
    second = split_events(event_ids, seed=7)
    assert first == second


def test_split_events_handles_small_event_count_without_crashing():
    event_ids = [101, 102, 103]
    splits = split_events(event_ids, seed=1)
    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == event_ids


def test_split_events_guarantees_at_least_one_per_split_when_n_is_at_least_three():
    # Antes del fix, n=3..5 podian dejar val o test completamente vacios
    # (contradiciendo el 70/15/15 documentado) -- verificado en la
    # revision final del 2026-09-27. A partir de 3 eventos, cada split
    # debe tener al menos 1.
    for n in range(3, 12):
        splits = split_events(list(range(n)), seed=1)
        assert len(splits["train"]) >= 1
        assert len(splits["val"]) >= 1
        assert len(splits["test"]) >= 1
        assert sum(len(v) for v in splits.values()) == n


def test_split_events_with_fewer_than_three_events_puts_everything_in_train():
    # Sin suficientes eventos para un split con sentido, todo va a train
    # -- documentado explicitamente, no un accidente silencioso.
    assert split_events([1], seed=1) == {"train": [1], "val": [], "test": []}
    assert sorted(split_events([1, 2], seed=1)["train"]) == [1, 2]


def test_split_events_input_order_does_not_change_the_result():
    # el split depende del contenido del conjunto de ids, no del orden en
    # que la lista de entrada los trae (los ids se ordenan antes de
    # mezclar con la semilla) -- dos llamadas con los mismos ids en
    # distinto orden de entrada deben dar el mismo split.
    forward = split_events([1, 2, 3, 4, 5], seed=3)
    shuffled_input = split_events([5, 3, 1, 4, 2], seed=3)
    assert forward == shuffled_input


# ---- split agrupado espaciotemporalmente (revisión independiente: fuga entre splits) ----
def _fp(west, south, day0, days=3, size=0.02):
    start = dt.date(2026, 1, day0)
    return EventFootprint(west, south, west + size, south + size, start,
                          start + dt.timedelta(days=days - 1))


def test_group_events_links_nearby_events_that_overlap_in_time_and_keeps_far_ones_apart():
    fps = {
        1: _fp(-72.60, -36.60, 15),
        2: _fp(-72.58, -36.58, 16),      # contiguo a 1, mismos días
        3: _fp(-72.60, -36.60, 25),      # mismo lugar, 10 días después -> independiente
        4: _fp(-71.00, -38.00, 15),      # mismos días, 200 km
    }
    groups = group_events(fps, max_gap_km=10.0, max_gap_days=3)
    assert sorted(map(sorted, groups)) == [[1, 2], [3], [4]]


def test_group_events_chains_transitively():
    fps = {i: _fp(-72.6 + i * 0.07, -36.6, 15) for i in range(4)}  # ~6 km entre vecinos
    assert group_events(fps, max_gap_km=10.0, max_gap_days=3) == [[0, 1, 2, 3]]


def test_grouped_split_never_separates_a_group_and_covers_every_event_once():
    fps = {i: _fp(-72.6 + (i // 2) * 1.0, -36.6, 15) for i in range(12)}  # 6 pares acoplados
    splits = split_events_grouped(fps, seed=3, max_gap_km=10.0, max_gap_days=3)
    assert sorted(sum(splits.values(), [])) == list(range(12))
    where = {e: s for s, ids in splits.items() for e in ids}
    for pair in ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]):
        assert where[pair[0]] == where[pair[1]]
    assert all(splits[s] for s in ("train", "val", "test"))
    assert find_split_leakage(splits, fps, 10.0, 3) == []


def test_grouped_split_is_reproducible_and_order_independent():
    fps = {i: _fp(-72.6 + i * 0.5, -36.6, 15) for i in range(10)}
    a = split_events_grouped(fps, seed=7)
    b = split_events_grouped(dict(reversed(list(fps.items()))), seed=7)
    assert a == b


def test_grouped_split_refuses_when_fewer_than_three_independent_groups():
    fps = {i: _fp(-72.6 + i * 0.03, -36.6, 15) for i in range(6)}  # todo un solo grupo
    with pytest.raises(ValueError, match="grupos independientes"):
        split_events_grouped(fps, seed=1)


def test_find_split_leakage_flags_adjacent_events_in_different_splits():
    fps = {1: _fp(-72.60, -36.60, 15), 2: _fp(-72.58, -36.58, 16), 3: _fp(-71.0, -38.0, 15)}
    leaks = find_split_leakage({"train": [3], "val": [2], "test": [1]}, fps, 10.0, 3)
    assert len(leaks) == 1
    leak = leaks[0]
    assert {leak["event_a"], leak["event_b"]} == {1, 2}
    assert {leak["split_a"], leak["split_b"]} == {"val", "test"}
    assert leak["gap_km"] < 10.0 and leak["gap_days"] == 0


def test_footprint_from_tensor_uses_fire_extent_and_days_with_fire():
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
    x0, y0 = to_utm.transform(-72.5, -37.5)
    size, res = 10, 250.0
    xs = x0 + res * (np.arange(size) + 0.5)
    ys = y0 - res * (np.arange(size) + 0.5)
    data = np.zeros((4, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire = CHANNEL_ORDER.index("fire_mask")
    data[1:3, fire, 2:4, 5:7] = 1.0  # fuego solo en los días 1 y 2
    tensor = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13"],
                "channel": list(CHANNEL_ORDER), "y": ys, "x": xs},
        attrs={"event_id": 1},
    )
    fp = footprint_from_tensor(tensor)
    assert (fp.first_day, fp.last_day) == (dt.date(2026, 1, 11), dt.date(2026, 1, 12))
    assert -72.5 < fp.west < fp.east < -72.4 and -37.6 < fp.south < fp.north < -37.5


def test_resplit_cli_rewrites_splits_without_leakage_and_keeps_the_previous_one(
    tmp_path, monkeypatch
):
    import json

    from features.cli import app
    from shared.config import get_settings
    from typer.testing import CliRunner

    for key, value in {
        "FIRMS_MAP_KEY": "x", "CDS_API_URL": "x", "CDS_API_KEY": "x",
        "COPERNICUS_DATASPACE_CLIENT_ID": "x", "COPERNICUS_DATASPACE_CLIENT_SECRET": "x",
        "POSTGRES_HOST": "x", "POSTGRES_PORT": "1", "POSTGRES_DB": "x", "POSTGRES_USER": "x",
        "POSTGRES_PASSWORD": "x", "DATA_PROCESSED_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
    ids = list(range(1, 8))
    for i in ids:
        # eventos 1 y 2 contiguos (mismo complejo); el resto, lejos entre sí
        lon, lat = (-72.60 + (0.02 if i == 2 else 0.0), -36.60) if i <= 2 else (
            -72.6 + (i - 3) * 1.0, -38.0)
        x0, y0 = to_utm.transform(lon, lat)
        size = 6
        data = np.zeros((3, len(CHANNEL_ORDER), size, size), dtype="float32")
        data[:, CHANNEL_ORDER.index("fire_mask"), 2:4, 2:4] = 1.0
        xr.DataArray(
            data, dims=("day", "channel", "y", "x"),
            coords={"day": ["2026-01-15", "2026-01-16", "2026-01-17"],
                    "channel": list(CHANNEL_ORDER),
                    "y": y0 - 250.0 * (np.arange(size) + 0.5),
                    "x": x0 + 250.0 * (np.arange(size) + 0.5)},
            name="fire_event_tensor", attrs={"event_id": i, "crs": "EPSG:32719"},
        ).to_dataset().to_zarr(dataset / f"event_{i:04d}.zarr", mode="w")
    # split viejo con fuga: 1 y 2 en splits distintos
    (dataset / "splits.json").write_text(json.dumps(
        {"train": [3, 4, 5], "val": [2], "test": [1, 6, 7]}))

    result = CliRunner().invoke(app, ["resplit"])
    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "fugas en el split anterior: 1" in result.output
    new = json.loads((dataset / "splits.json").read_text())
    assert sorted(sum(new.values(), [])) == ids
    where = {e: s for s, v in new.items() for e, s in [(e, s) for e in v]}
    assert where[1] == where[2]
    assert json.loads((dataset / "splits.previous.json").read_text())["val"] == [2]
    assert json.loads((dataset / "split_groups.json").read_text())["leakage_in_previous_split"]
