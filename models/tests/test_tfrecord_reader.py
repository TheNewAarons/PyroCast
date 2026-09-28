"""Tests del lector TFRecord + tf.Example sin dependencias (sin
tensorflow, sin el paquete `tfrecord` de PyPI -- ver docs/decisions.md).
Verificado por round-trip contra un ENCODER escrito a mano en
tfrecord_fixtures.py (no hay acceso de red en este entorno para probar
contra un archivo real descargado de Kaggle -- ver
docs/public-dataset.md, sección de incertidumbre)."""
import numpy as np
import pytest
from models.deep.tfrecord_reader import CorruptTFRecordError, read_tf_examples
from tfrecord_fixtures import write_tfrecord as _write_tfrecord


def test_read_tf_examples_round_trips_a_single_record(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    original = {
        "elevation": np.arange(16, dtype="float32").reshape(4, 4),
        "NDVI": np.full((4, 4), 5000.0, dtype="float32"),
    }
    _write_tfrecord(path, [original])

    records = list(read_tf_examples(path))
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["elevation"], original["elevation"])
    np.testing.assert_array_equal(records[0]["NDVI"], original["NDVI"])


def test_read_tf_examples_round_trips_multiple_records(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    records_in = [
        {"elevation": np.zeros((2, 2), dtype="float32")},
        {"elevation": np.ones((2, 2), dtype="float32")},
        {"elevation": np.full((2, 2), 7.0, dtype="float32")},
    ]
    _write_tfrecord(path, records_in)

    records_out = list(read_tf_examples(path))
    assert len(records_out) == 3
    for expected, actual in zip(records_in, records_out, strict=True):
        np.testing.assert_array_equal(actual["elevation"], expected["elevation"])


def test_read_tf_examples_rejects_a_non_square_feature(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    # 6 valores -- no es un cuadrado perfecto (sqrt(6) no es entero).
    _write_tfrecord(path, [{"elevation": np.zeros(6, dtype="float32")}])
    with pytest.raises(CorruptTFRecordError, match="elevation"):
        list(read_tf_examples(path))


def test_read_tf_examples_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [{"elevation": np.zeros((2, 2), dtype="float32")}])
    truncated = path.read_bytes()[:-5]  # corta a mitad del último CRC
    path.write_bytes(truncated)
    with pytest.raises(CorruptTFRecordError):
        list(read_tf_examples(path))


def test_read_tf_examples_rejects_a_corrupted_length_crc(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [{"elevation": np.zeros((2, 2), dtype="float32")}])
    raw = bytearray(path.read_bytes())
    raw[8] ^= 0xFF  # corrompe un byte del CRC de longitud (offset 8-11)
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptTFRecordError, match="CRC"):
        list(read_tf_examples(path))


def test_read_tf_examples_on_an_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.tfrecord"
    path.write_bytes(b"")
    assert list(read_tf_examples(path)) == []
