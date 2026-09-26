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
    # west, south, east, north — Biobío/Ñuble/Araucanía-shaped bbox. east
    # lands exactly on -71.0: the tile starting at -71 covers [-71,-70)
    # and has zero overlap with this bbox, so the NE-most tile is -72,
    # not -71 (see test below for the half-open-interval reasoning).
    bbox = (-73.7, -39.3, -71.0, -36.5)
    tiles = tiles_for_bbox(bbox)
    assert (-40, -74) in tiles  # SW-most tile
    assert (-37, -72) in tiles  # NE-most tile
    assert (-37, -71) not in tiles  # zero-overlap tile, must be excluded
    assert len(tiles) == (40 - 37 + 1) * (74 - 72 + 1)  # 4 lat bands x 3 lon bands = 12


def test_tiles_for_bbox_excludes_tile_when_edge_lands_exactly_on_its_start():
    # Cada tile cubre [lat, lat+1) x [lon, lon+1) -- semiabierto. Un borde
    # north/east que cae EXACTAMENTE en un entero N no se solapa con el
    # tile que arranca en N (ese tile es [N, N+1)) -- incluirlo pediría
    # un tile de cobertura cero, y en la práctica falla con 404 cuando
    # ese tile cae en el océano (bboxes de números redondos son comunes).
    bbox = (-72.0, -37.0, -71.0, -36.0)
    tiles = tiles_for_bbox(bbox)
    assert tiles == [(-37, -72)]


def test_tiles_for_bbox_single_tile():
    bbox = (-72.6, -37.6, -72.1, -37.1)
    assert tiles_for_bbox(bbox) == [(-38, -73)]
