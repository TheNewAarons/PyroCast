"""Tests del cliente Sentinel-2: openeo.Connection mockeado por completo."""
from pathlib import Path

from ingestion.sentinel2.client import CLOUD_SCL_CLASSES, SENTINEL2_BANDS, Sentinel2Client

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


class _FakeDataCube:
    def __init__(self, log: list, name: str = "load_collection"):
        self._log = log
        self._name = name

    def band(self, name: str) -> "_FakeDataCube":
        self._log.append(("band", name))
        return _FakeDataCube(self._log, f"band:{name}")

    def __ne__(self, other) -> "_FakeDataCube":  # scl_band != cloud_class
        self._log.append(("ne", other))
        return _FakeDataCube(self._log, "mask")

    def __and__(self, other) -> "_FakeDataCube":
        self._log.append(("and", "combine"))
        return _FakeDataCube(self._log, "mask")

    def resample_cube_spatial(self, target: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("resample_cube_spatial",))
        return self

    def mask(self, mask_cube: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("mask",))
        return self

    def reduce_dimension(self, dimension: str, reducer: str) -> "_FakeDataCube":
        self._log.append(("reduce_dimension", dimension, reducer))
        return self

    def download(self, target: str) -> None:
        self._log.append(("download", target))
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
    assert temporal_extent == ["2026-01-01", "2026-01-31"]
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


def test_february_month_end_is_28_not_30(tmp_path):
    # un mes de 28/29/30/31 días construido con calendar, no un +30 fijo
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2025, month=2, target=tmp_path / "unused.tif")
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    assert call[3] == ["2025-02-01", "2025-02-28"]  # 2025 no es bisiesto


def test_cloud_scl_classes_are_the_documented_set():
    assert CLOUD_SCL_CLASSES == frozenset({3, 8, 9, 10})
