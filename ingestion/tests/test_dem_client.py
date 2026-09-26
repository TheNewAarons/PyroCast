"""Tests del cliente de descarga DEM: cero red real (usa `responses`)."""
import pytest
import responses
from ingestion.dem.client import DemDownloadError, download_tile
from ingestion.dem.tiles import tile_url

KEY = "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"


@responses.activate
def test_download_tile_writes_bytes_to_dest_path(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"fake-tif-bytes"


@responses.activate
def test_download_tile_raises_dem_download_error_on_404(tmp_path):
    responses.add(responses.GET, tile_url(KEY), status=404)
    with pytest.raises(DemDownloadError, match="404"):
        download_tile(KEY, tmp_path / f"{KEY}.tif")


@responses.activate
def test_download_tile_skips_request_when_dest_already_exists(tmp_path):
    dest = tmp_path / f"{KEY}.tif"
    dest.write_bytes(b"already-here")
    # deliberadamente no se registra ningún response — una petición real
    # levantaría responses.exceptions.ConnectionError, probando que el
    # cache hit se saltó la red
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"already-here"


@responses.activate
def test_download_tile_leaves_no_half_written_dest_on_a_write_failure(tmp_path, monkeypatch):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"

    from pathlib import Path

    real_write_bytes = Path.write_bytes

    def _failing_write_bytes(self: Path, data: bytes) -> int:
        if self.name.endswith(".part"):
            real_write_bytes(self, data)
            raise OSError("simulated interrupted write")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", _failing_write_bytes)

    with pytest.raises(OSError, match="simulated interrupted write"):
        download_tile(KEY, dest)

    assert not dest.exists()  # nunca se llegó a os.replace()


@responses.activate
def test_download_tile_leaves_no_stray_part_file_after_success(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    download_tile(KEY, dest)
    assert not dest.with_suffix(dest.suffix + ".part").exists()
