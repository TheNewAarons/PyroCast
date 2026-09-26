"""Tests de la grilla de tiles WorldCover: solo matemática, sin red."""
from ingestion.worldcover.tiles import tile_key, tile_url, tiles_for_bbox


def test_tile_key_matches_real_bucket_naming():
    assert tile_key(-39, -72) == "ESA_WorldCover_10m_2021_v200_S39W072_Map"


def test_tile_key_north_east_hemisphere():
    assert tile_key(0, 6) == "ESA_WorldCover_10m_2021_v200_N00E006_Map"


def test_tile_url_matches_real_bucket_path():
    url = tile_url("ESA_WorldCover_10m_2021_v200_S39W072_Map")
    assert url == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif"
    )


def test_tiles_for_bbox_study_area_uses_3_degree_multiples():
    bbox = (-73.7, -39.3, -71.0, -36.5)
    tiles = tiles_for_bbox(bbox)
    assert (-42, -75) in tiles  # SW-most 3-degree tile
    assert (-39, -72) in tiles  # NE-most 3-degree tile
    for lat, lon in tiles:
        assert lat % 3 == 0
        assert lon % 3 == 0


def test_tiles_for_bbox_excludes_tile_when_edge_lands_exactly_on_its_start():
    # borde exactamente en un múltiplo de 3: el tile que arranca ahí no
    # se solapa (cubre [N, N+3)) — mismo caso que ingestion/dem, corregido
    # de entrada aquí.
    bbox = (-72.0, -39.0, -69.0, -36.0)
    assert tiles_for_bbox(bbox) == [(-39, -72)]


def test_tiles_for_bbox_single_tile():
    bbox = (-72.9, -38.9, -72.1, -38.1)
    assert tiles_for_bbox(bbox) == [(-39, -75)]
