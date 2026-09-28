"""Encoder TFRecord + tf.Example escrito a mano, usado SOLO por los
tests (`test_tfrecord_reader.py`, `test_public_dataset.py`) para
construir archivos de fixture reales sin depender de `tensorflow`. No
es parte del código de producción -- ver
`models/src/models/deep/tfrecord_reader.py` para el lector real."""
import struct
from pathlib import Path

import numpy as np

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


def write_tfrecord(path: Path, records: list[dict[str, np.ndarray]]) -> None:
    with open(path, "wb") as f:
        for record in records:
            data = _encode_example(record)
            length_bytes = struct.pack("<Q", len(data))
            length_crc = struct.pack("<I", _mask_crc(_crc32c(length_bytes)))
            data_crc = struct.pack("<I", _mask_crc(_crc32c(data)))
            f.write(length_bytes + length_crc + data + data_crc)
