"""Tests del cliente Sentinel-2: openeo.Connection mockeado por completo."""
from pathlib import Path

import pytest
from ingestion.sentinel2.client import CLOUD_SCL_CLASSES, SENTINEL2_BANDS, Sentinel2Client

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


class _FakeDataCube:
    def __init__(self, log: list, name: str = "load_collection"):
        self._log = log
        self._name = name

    def band(self, name: str) -> "_FakeDataCube":
        self._log.append(("band", name))
        return _FakeDataCube(self._log, f"band:{name}")

    def __eq__(self, other) -> "_FakeDataCube":  # type: ignore[override]
        self._log.append(("eq", other))
        return _FakeDataCube(self._log, "mask")

    def __or__(self, other) -> "_FakeDataCube":
        self._log.append(("or", "combine"))
        return _FakeDataCube(self._log, "mask")

    def resample_cube_spatial(self, target: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("resample_cube_spatial",))
        return self

    def mask(self, mask_cube: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("mask",))
        return self

    def filter_bands(self, bands: list) -> "_FakeDataCube":
        self._log.append(("filter_bands", tuple(bands)))
        return self

    def reduce_dimension(self, dimension: str, reducer: str) -> "_FakeDataCube":
        self._log.append(("reduce_dimension", dimension, reducer))
        return self

    def download(self, target: str, format: str | None = None) -> None:  # noqa: A002
        self._log.append(("download", target, format))
        Path(target).write_bytes(b"fake-composite-bytes")


class _FakeConnection:
    def __init__(self):
        self.log: list = []
        self.authenticated_with: tuple | None = None

    def authenticate_oidc_client_credentials(self, client_id: str, client_secret: str) -> None:
        self.authenticated_with = (client_id, client_secret)

    def load_collection(
        self, collection_id, spatial_extent, temporal_extent, bands, max_cloud_cover
    ):
        self.log.append(
            (
                "load_collection", collection_id, spatial_extent, temporal_extent, bands,
                max_cloud_cover,
            )
        )
        return _FakeDataCube(self.log)


def test_fetch_monthly_composite_authenticates_with_client_credentials(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="my-id", client_secret="my-secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    assert fake_connection.authenticated_with == ("my-id", "my-secret")


def test_fetch_monthly_composite_uses_named_spatial_extent_no_axis_reorder(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    _, collection_id, spatial_extent, temporal_extent, bands, _ = call
    assert collection_id == "SENTINEL2_L2A"
    assert spatial_extent == {"west": -73.7, "south": -39.3, "east": -71.0, "north": -36.5}
    assert temporal_extent == ["2026-01-01", "2026-02-01"]
    assert bands == list(SENTINEL2_BANDS)


def test_fetch_monthly_composite_downloads_to_target(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    target = tmp_path / "out.tif"
    result = client.fetch_monthly_composite(BBOX, year=2026, month=2, target=target)
    assert result == target
    assert target.read_bytes() == b"fake-composite-bytes"


def test_temporal_extent_end_is_first_of_next_month_not_last_day_of_this_one(tmp_path):
    # openEO's load_collection trata el limite superior de temporal_extent
    # como EXCLUSIVO ("excluded from the interval", spec oficial de
    # `load_collection`) -- usar el ultimo dia del mes como limite
    # perderia ese dia completo de observaciones.
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2025, month=2, target=tmp_path / "unused.tif")
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    assert call[3] == ["2025-02-01", "2025-03-01"]


def test_temporal_extent_rolls_over_year_for_december(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2025, month=12, target=tmp_path / "unused.tif")
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    assert call[3] == ["2025-12-01", "2026-01-01"]


def test_cloud_scl_classes_are_the_documented_set():
    assert CLOUD_SCL_CLASSES == frozenset({3, 8, 9, 10})


def test_cloud_mask_is_built_true_where_cloudy_using_eq_and_or(tmp_path):
    # openEO `mask(mask_cube)` reemplaza por nodata donde mask_cube es
    # True/no-cero (process spec oficial de `mask`) -- la mascara debe
    # construirse en sentido "true = es nube/sombra/cirros -> descartar",
    # nunca "true = esta claro" (eso enmascararia lo claro y conservaria
    # la nube, exactamente al reves de lo pedido).
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    eq_values = {c[1] for c in fake_connection.log if c[0] == "eq"}
    assert eq_values == set(CLOUD_SCL_CLASSES)
    or_count = sum(1 for c in fake_connection.log if c[0] == "or")
    assert or_count == len(CLOUD_SCL_CLASSES) - 1
    assert not any(c[0] == "ne" for c in fake_connection.log)
    assert not any(c[0] == "and" for c in fake_connection.log)


def test_scl_is_filtered_out_before_temporal_reduce(tmp_path):
    # SCL es un codigo categorico: reducirlo con mediana temporal
    # fabricaria clases inexistentes (mediana de [4, 8] = 6.0, "Bare
    # soil", que no ocurrio en ninguna observacion real). Debe
    # descartarse del cubo antes de `reduce_dimension`, usarse solo para
    # construir la mascara de nubes server-side.
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    filter_call = next(c for c in fake_connection.log if c[0] == "filter_bands")
    assert filter_call[1] == ("B04", "B08")
    filter_idx = fake_connection.log.index(filter_call)
    reduce_idx = next(
        i for i, c in enumerate(fake_connection.log) if c[0] == "reduce_dimension"
    )
    assert filter_idx < reduce_idx


def test_download_is_atomic_no_stray_part_file_after_success(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    target = tmp_path / "out.tif"
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=target)
    assert target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_download_passes_an_explicit_format_not_guessed_from_the_part_suffix(tmp_path):
    # Encontrado contra una descarga REAL (revisión final del
    # 2026-09-29): sin `format=` explícito, openEO adivina el formato de
    # salida a partir de la EXTENSIÓN del archivo de destino -- pero el
    # destino real que se le pasa a `download()` es la ruta temporal
    # ".part" (`target.with_suffix(target.suffix + ".part")`, para el
    # rename atómico), no el ".tif" final. Adivinar desde ".tif.part" da
    # `ValueError: Invalid format 'PART'. Should be one of {...}` --
    # reproducido contra la API real de Copernicus Data Space. Pasar
    # `format="GTiff"` explícito evita depender de la extensión del
    # archivo temporal en absoluto.
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    download_call = next(c for c in fake_connection.log if c[0] == "download")
    assert download_call[2] == "GTiff"


def test_interrupted_download_never_produces_final_target(tmp_path):
    class _CrashingDataCube(_FakeDataCube):
        def download(self, target: str, format: str | None = None) -> None:  # noqa: A002
            Path(target).write_bytes(b"TRUNCATED-partial-download")
            raise RuntimeError("network interrupted")

    class _CrashingConnection(_FakeConnection):
        def load_collection(self, *args, **kwargs):
            self.log.append(("load_collection", *args))
            return _CrashingDataCube(self.log)

    fake_connection = _CrashingConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    target = tmp_path / "out.tif"
    with pytest.raises(RuntimeError):
        client.fetch_monthly_composite(BBOX, year=2026, month=1, target=target)
    assert not target.exists()  # el .part truncado nunca se promovio a final
