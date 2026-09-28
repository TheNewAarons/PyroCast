"""Tests del adaptador NDWS -> esquema de tensor de PyroCast. El
fixture simula el formato del dataset público (un registro TFRecord de
verdad, escrito con el mismo encoder de test_tfrecord_reader.py) con
los 13 nombres de feature reales de NDWS, en una grilla pequeña (4x4,
no 64x64 -- las dimensiones no importan para probar el MAPEO de
canales, solo que se preserven)."""
from pathlib import Path

import numpy as np
import pytest
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.public_dataset import (
    PublicDatasetSample,
    load_public_dataset_samples,
    split_public_dataset,
    transform_ndws_record,
)
from tfrecord_fixtures import write_tfrecord as _write_tfrecord

_SIZE = 4


def _make_ndws_record(
    th_deg: float = 90.0,
    vs: float = 10.0,
    tmmn: float = 280.0,
    tmmx: float = 300.0,
    sph: float = 0.0,
    pr_mm: float = 5.0,
    ndvi_raw: float = 5000.0,
    prev_fire: float = 1.0,
    next_fire: float = 0.0,
) -> dict[str, np.ndarray]:
    def const(value: float) -> np.ndarray:
        return np.full((_SIZE, _SIZE), value, dtype="float32")

    return {
        "elevation": np.arange(_SIZE * _SIZE, dtype="float32").reshape(_SIZE, _SIZE),
        "pdsi": const(1.0),
        "NDVI": const(ndvi_raw),
        "pr": const(pr_mm),
        "sph": const(sph),
        "th": const(th_deg),
        "tmmn": const(tmmn),
        "tmmx": const(tmmx),
        "vs": const(vs),
        "erc": const(30.0),
        "population": const(5.0),
        "PrevFireMask": const(prev_fire),
        "FireMask": const(next_fire),
    }


def test_transform_ndws_record_produces_the_pyrocast_channel_order():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    assert isinstance(sample, PublicDatasetSample)
    assert list(sample.tensor.coords["channel"].values) == list(CHANNEL_ORDER)


def test_transform_ndws_record_produces_the_expected_shape():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    # (day=1, channel=11, y=4, x=4)
    assert sample.tensor.shape == (1, len(CHANNEL_ORDER), _SIZE, _SIZE)
    assert sample.next_day_fire_mask.shape == (_SIZE, _SIZE)


def test_transform_ndws_record_copies_elevation_directly():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    elevation = sample.tensor.values[0, channels.index("elevation")]
    np.testing.assert_array_equal(elevation, record["elevation"])


def test_transform_ndws_record_converts_precipitation_mm_to_meters():
    record = _make_ndws_record(pr_mm=5.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    precip = sample.tensor.values[0, channels.index("precipitation")]
    assert precip[0, 0] == pytest.approx(0.005)


def test_transform_ndws_record_converts_temperature_to_mean_of_min_max():
    record = _make_ndws_record(tmmn=280.0, tmmx=300.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    temperature = sample.tensor.values[0, channels.index("temperature")]
    assert temperature[0, 0] == pytest.approx(290.0)


def test_transform_ndws_record_converts_ndvi_by_dividing_by_ten_thousand():
    record = _make_ndws_record(ndvi_raw=5000.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    ndvi = sample.tensor.values[0, channels.index("ndvi")]
    assert ndvi[0, 0] == pytest.approx(0.5)


def test_transform_ndws_record_converts_wind_direction_and_speed_to_components():
    # th=90 (viento SOPLA DESDE el este) -> avanza hacia el oeste ->
    # componente u (este) NEGATIVA, v (norte) CERO.
    record = _make_ndws_record(th_deg=90.0, vs=10.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    wind_u = sample.tensor.values[0, channels.index("wind_u")]
    wind_v = sample.tensor.values[0, channels.index("wind_v")]
    assert wind_u[0, 0] == pytest.approx(-10.0, abs=1e-6)
    assert wind_v[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_transform_ndws_record_fills_fuel_type_with_the_unknown_code():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    fuel_type = sample.tensor.values[0, channels.index("fuel_type")]
    assert np.all(fuel_type == 99.0)


def test_transform_ndws_record_clips_uncertain_fire_mask_to_not_fire():
    record = _make_ndws_record(prev_fire=-1.0, next_fire=-1.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    fire_mask = sample.tensor.values[0, channels.index("fire_mask")]
    assert np.all(fire_mask == 0.0)
    assert np.all(sample.next_day_fire_mask == 0.0)


def test_transform_ndws_record_relative_humidity_is_zero_when_bone_dry():
    record = _make_ndws_record(sph=0.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    rh = sample.tensor.values[0, channels.index("relative_humidity")]
    assert rh[0, 0] == pytest.approx(0.0)


def test_transform_ndws_record_relative_humidity_is_self_consistent_at_saturation():
    # construye sph EXACTAMENTE en el punto de saturación a la misma T
    # que usa la conversión (mean(tmmn,tmmx)) -- debe dar ~100% RH,
    # verificando la fórmula contra sí misma en vez de un número
    # calculado a mano (evita un error de aritmética en el test).
    from models.deep.public_dataset import _saturation_specific_humidity

    temp_k = 290.0
    q_sat = _saturation_specific_humidity(temp_k)
    record = _make_ndws_record(tmmn=temp_k, tmmx=temp_k, sph=q_sat)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    rh = sample.tensor.values[0, channels.index("relative_humidity")]
    assert rh[0, 0] == pytest.approx(100.0, abs=0.01)


def test_transform_ndws_record_rejects_a_record_missing_a_required_feature():
    record = _make_ndws_record()
    del record["th"]
    with pytest.raises(ValueError, match="th"):
        transform_ndws_record(record, sample_id=1)


def test_load_public_dataset_samples_reads_a_real_fixture_file(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [_make_ndws_record(), _make_ndws_record()])

    samples = list(load_public_dataset_samples([path]))
    assert len(samples) == 2
    assert all(isinstance(s, PublicDatasetSample) for s in samples)


def test_split_public_dataset_is_deterministic_and_covers_every_shard():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(10)]
    first = split_public_dataset(shards, seed=1)
    second = split_public_dataset(shards, seed=1)
    assert first == second
    assert set(first["train"]) | set(first["val"]) == set(shards)
    assert set(first["train"]).isdisjoint(first["val"])
    assert "test" not in first


def test_split_public_dataset_never_leaves_val_empty_with_enough_shards():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(5)]
    result = split_public_dataset(shards, seed=1)
    assert len(result["val"]) >= 1


def test_split_public_dataset_is_disjoint_from_a_different_seed():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(20)]
    a = split_public_dataset(shards, seed=1)
    b = split_public_dataset(shards, seed=2)
    assert a != b  # distinto seed -> partición distinta (con alta probabilidad, N=20)
