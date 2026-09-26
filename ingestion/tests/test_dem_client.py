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
