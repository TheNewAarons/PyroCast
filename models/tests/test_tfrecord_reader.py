"""Tests del lector TFRecord + tf.Example sin dependencias (sin
tensorflow, sin el paquete `tfrecord` de PyPI -- ver docs/decisions.md).
Verificado por round-trip contra un ENCODER escrito a mano en este mismo
archivo de test (no hay acceso de red en este entorno para probar
contra un archivo real descargado de Kaggle -- ver
docs/public-dataset.md, sección de incertidumbre)."""
import struct
from pathlib import Path

import numpy as np
import pytest
from models.deep.tfrecord_reader import CorruptTFRecordError, read_tf_examples

_CRC32C_POLY = 0x82F63B78  # polinomio Castagnoli reflejado


def _crc32c_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _mask_crc(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_float_list_feature(values: np.ndarray) -> bytes:
    packed = struct.pack(f"<{values.size}f", *values.ravel().tolist())
    float_list = _encode_length_delimited(1, packed)  # FloatList.value (packed)
    return _encode_length_delimited(2, float_list)  # Feature.float_list


def _encode_example(features: dict[str, np.ndarray]) -> bytes:
    feature_entries = b""
    for name, values in features.items():
        key_bytes = _encode_length_delimited(1, name.encode("utf-8"))
        value_bytes = _encode_length_delimited(2, _encode_float_list_feature(values))
        entry = key_bytes + value_bytes
        feature_entries += _encode_length_delimited(1, entry)  # Features.feature (map entry)
    features_message = feature_entries
    example = _encode_length_delimited(1, features_message)  # Example.features
    return example


def _write_tfrecord(path: Path, records: list[dict[str, np.ndarray]]) -> None:
    with open(path, "wb") as f:
        for record in records:
            data = _encode_example(record)
            length_bytes = struct.pack("<Q", len(data))
            length_crc = struct.pack("<I", _mask_crc(_crc32c(length_bytes)))
            data_crc = struct.pack("<I", _mask_crc(_crc32c(data)))
            f.write(length_bytes + length_crc + data + data_crc)


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
