"""Tests del cliente de descarga WorldCover: cero red real (usa `responses`)."""
import pytest
import responses
from ingestion.worldcover.client import WorldCoverDownloadError, download_tile
from ingestion.worldcover.tiles import tile_url

KEY = "ESA_WorldCover_10m_2021_v200_S39W072_Map"


@responses.activate
def test_download_tile_writes_bytes_to_dest_path(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"fake-tif-bytes"


@responses.activate
def test_download_tile_raises_on_404(tmp_path):
    responses.add(responses.GET, tile_url(KEY), status=404)
    with pytest.raises(WorldCoverDownloadError, match="404"):
        download_tile(KEY, tmp_path / f"{KEY}.tif")


@responses.activate
def test_download_tile_skips_request_when_dest_already_exists(tmp_path):
    dest = tmp_path / f"{KEY}.tif"
    dest.write_bytes(b"already-here")
    result = download_tile(KEY, dest)  # sin response registrado -> probaría red real si se llamara
    assert result == dest
    assert dest.read_bytes() == b"already-here"


@responses.activate
def test_download_tile_leaves_no_stray_part_file_after_success(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    download_tile(KEY, dest)
    assert not dest.with_suffix(dest.suffix + ".part").exists()


@responses.activate
def test_download_tile_uses_caller_provided_version_and_year_in_url(tmp_path):
    # Antes del fix, download_tile llamaba a tile_url(key) con los
    # defaults ("v200"/"2021") sin importar qué version/year recibiera
    # el llamador -- build_worldcover(year="2020") pedía siempre la URL
    # de 2021 y obtenía 404 en todos los tiles.
    other_key = "ESA_WorldCover_10m_2020_v100_S39W072_Map"
    expected_url = tile_url(other_key, version="v100", year="2020")
    responses.add(responses.GET, expected_url, body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{other_key}.tif"
    result = download_tile(other_key, dest, version="v100", year="2020")
    assert result == dest
    assert dest.read_bytes() == b"fake-tif-bytes"
