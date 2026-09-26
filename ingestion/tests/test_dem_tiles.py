"""Tests de la grilla de tiles Copernicus DEM: solo matemática, sin red."""
from ingestion.dem.tiles import tile_key, tile_url, tiles_for_bbox


def test_tile_key_south_west_hemisphere():
    assert tile_key(-37, -72) == "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"


def test_tile_key_north_east_hemisphere():
    assert tile_key(5, 10) == "Copernicus_DSM_COG_10_N05_00_E010_00_DEM"


def test_tile_url_builds_full_tif_path():
    url = tile_url("Copernicus_DSM_COG_10_S37_00_W072_00_DEM")
    assert url == (
        "https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_S37_00_W072_00_DEM/"
        "Copernicus_DSM_COG_10_S37_00_W072_00_DEM.tif"
    )


def test_tiles_for_bbox_covers_study_area_example():
    # west, south, east, north — Biobío/Ñuble/Araucanía-shaped bbox
    bbox = (-73.7, -39.3, -71.0, -36.5)
    tiles = tiles_for_bbox(bbox)
    assert (-40, -74) in tiles  # SW-most tile
    assert (-37, -71) in tiles  # NE-most tile
    assert len(tiles) == (40 - 37 + 1) * (74 - 71 + 1)  # 4 lat bands x 4 lon bands = 16


def test_tiles_for_bbox_includes_tile_when_edge_lands_on_integer_degree():
    # north edge exactly on -36.0: must still include the S36 tile whose
    # southern edge IS that boundary, and must NOT silently stop at S37.
    bbox = (-72.0, -37.0, -71.0, -36.0)
    tiles = tiles_for_bbox(bbox)
    assert (-37, -72) in tiles
    assert (-36, -72) in tiles


def test_tiles_for_bbox_single_tile():
    bbox = (-72.6, -37.6, -72.1, -37.1)
    assert tiles_for_bbox(bbox) == [(-38, -73)]
