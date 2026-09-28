"""Lector del formato contenedor TFRecord + el esquema protobuf
`tf.train.Example`, sin depender de `tensorflow` ni del paquete
`tfrecord` de PyPI (este último tiene un conflicto de versión de
protobuf conocido) -- ver docs/decisions.md. Domain-agnostic: no sabe
nada de NDWS ni de PyroCast, solo sabe leer el formato binario. Ver
docs/public-dataset.md para el contexto de qué dataset lo usa.

Formato TFRecord (contenedor, por registro):
    uint64 longitud                              (little-endian)
    uint32 CRC32C enmascarado de los 8 bytes de longitud
    byte   datos[longitud]                       (un Example serializado)
    uint32 CRC32C enmascarado de datos

CRC32C (Castagnoli) se implementa y se VALIDA (no se omite) -- un
archivo grande descargado manualmente que llega truncado o corrupto
debe fallar ruidosamente en el primer registro malo, no devolver datos
parciales en silencio. La fórmula de "máscara" es la estándar de
TFRecord: `mask(crc) = ((crc >> 15) | (crc << 17)) + 0xa282ead8 (mod
2^32)`.

`tf.train.Example` (protobuf, solo los wire types que este esquema usa
-- varint y length-delimited, nunca fixed32/fixed64 sueltos, porque los
floats van "packed" dentro de un length-delimited):
    Example { Features features = 1; }
    Features { map<string, Feature> feature = 1; }   -- un map en wire
               format es un repeated de mensajes {key=1, value=2}
    Feature { oneof { FloatList float_list = 2; ... } }  -- solo
               float_list está implementado: es lo único que NDWS usa.
    FloatList { repeated float value = 1 [packed=true]; }  -- packed:
               los floats van concatenados como bytes crudos dentro de
               UN campo length-delimited, no como floats sueltos.

*** LIMITACIÓN DE VERIFICACIÓN, DOCUMENTADA A PROPÓSITO: este lector se
verificó por round-trip contra un encoder escrito a mano en
models/tests/test_tfrecord_reader.py, siguiendo la especificación
pública del formato. Este entorno no tiene acceso de red para
descargar un archivo real de Kaggle y probar compatibilidad byte a
byte contra él -- ver docs/public-dataset.md. ***
"""
import gzip
import math
import struct
from collections.abc import Iterator
from gzip import GzipFile
from io import BufferedReader
from pathlib import Path

import numpy as np

_GZIP_MAGIC = b"\x1f\x8b"

_CRC32C_POLY = 0x82F63B78  # polinomio Castagnoli reflejado
_CRC_MASK_DELTA = 0xA282EAD8


class CorruptTFRecordError(ValueError):
    """El archivo TFRecord está truncado, tiene un CRC inválido, o un
    Example con una feature que no se puede interpretar como una
    grilla cuadrada."""


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
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF


def _read_exact(f: BufferedReader | GzipFile, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise CorruptTFRecordError(
            f"archivo TFRecord truncado -- se esperaban {n} bytes, se leyeron {len(data)}."
        )
    return data


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise CorruptTFRecordError("varint truncado dentro de un Example.")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_top_level_fields(data: bytes) -> Iterator[tuple[int, bytes | int]]:
    """Itera los campos de nivel superior de un mensaje protobuf,
    soportando solo wire type 0 (varint) y 2 (length-delimited) -- los
    únicos que el esquema de tf.train.Example usa."""
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
            yield field_number, value
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            if pos + length > len(data):
                raise CorruptTFRecordError("submensaje length-delimited truncado.")
            yield field_number, data[pos : pos + length]
            pos += length
        else:
            raise CorruptTFRecordError(f"wire type {wire_type} no soportado.")


def _parse_float_list(feature_bytes: bytes) -> np.ndarray | None:
    for field_number, value in _iter_top_level_fields(feature_bytes):
        if field_number == 2 and isinstance(value, bytes):  # Feature.float_list
            for inner_field, inner_value in _iter_top_level_fields(value):
                if inner_field == 1 and isinstance(inner_value, bytes):  # FloatList.value
                    count = len(inner_value) // 4
                    return np.frombuffer(inner_value, dtype="<f4", count=count).astype("float32")
    return None


def _parse_example(data: bytes) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field_number, features_bytes in _iter_top_level_fields(data):
        if field_number != 1 or not isinstance(features_bytes, bytes):
            continue  # Example.features
        for entry_field, entry_bytes in _iter_top_level_fields(features_bytes):
            if entry_field != 1 or not isinstance(entry_bytes, bytes):
                continue  # Features.feature map entry
            name = None
            flat: np.ndarray | None = None
            for inner_field, inner_value in _iter_top_level_fields(entry_bytes):
                if inner_field == 1 and isinstance(inner_value, bytes):
                    name = inner_value.decode("utf-8")
                elif inner_field == 2 and isinstance(inner_value, bytes):
                    flat = _parse_float_list(inner_value)
            if name is not None and flat is not None:
                side = math.isqrt(flat.size)
                if side * side != flat.size:
                    raise CorruptTFRecordError(
                        f"feature '{name}' tiene {flat.size} valores, que no es un "
                        f"cuadrado perfecto -- no se puede reformar a una grilla."
                    )
                result[name] = flat.reshape(side, side)
    return result


def _open_maybe_gzip(path: Path) -> BufferedReader | GzipFile:
    # NDWS se distribuye como shards *.tfrecord.gz (ver
    # docs/public-dataset.md) -- se detecta por los magic bytes (1f 8b),
    # NUNCA por la extensión del archivo: el plan deliberadamente nunca
    # asume nombres de archivo de Kaggle, y sniffear el contenido real
    # funciona sin importar cómo el usuario haya nombrado el archivo
    # descargado.
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == _GZIP_MAGIC:
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_tf_examples(path: Path) -> Iterator[dict[str, np.ndarray]]:
    with _open_maybe_gzip(path) as f:
        while True:
            length_bytes = f.read(8)
            if length_bytes == b"":
                return
            if len(length_bytes) != 8:
                raise CorruptTFRecordError("longitud de registro truncada.")
            (length,) = struct.unpack("<Q", length_bytes)

            length_crc_bytes = _read_exact(f, 4)
            (length_crc,) = struct.unpack("<I", length_crc_bytes)
            if _mask_crc(_crc32c(length_bytes)) != length_crc:
                raise CorruptTFRecordError("CRC de longitud inválido -- archivo corrupto.")

            data = _read_exact(f, length)

            data_crc_bytes = _read_exact(f, 4)
            (data_crc,) = struct.unpack("<I", data_crc_bytes)
            if _mask_crc(_crc32c(data)) != data_crc:
                raise CorruptTFRecordError("CRC de datos inválido -- archivo corrupto.")

            yield _parse_example(data)
