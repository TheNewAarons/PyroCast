"""Grilla de tiles de ESA WorldCover (bucket público de AWS).

Cada tile cubre 3°x3°, nombrado por su esquina suroeste — mismo esquema
semiabierto que Copernicus DEM (ver ingestion/dem/tiles.py), verificado
contra el bucket real (2026-09-26): S39W072 cubre [-39,-36) x [-72,-69).
Usa ceil(x)-1 para los límites superiores desde el principio — la
revisión final de ingestion/dem encontró y corrigió este mismo bug
usando floor() para ambos límites; se aplica la lección aquí de entrada.
"""
import math

_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
_TILE_SIZE_DEG = 3


def tile_key(lat: int, lon: int, version: str = "v200", year: str = "2021") -> str:
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return f"ESA_WorldCover_10m_{year}_{version}_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map"


def tile_url(key: str, version: str = "v200", year: str = "2021") -> str:
    return f"{_BASE_URL}/{version}/{year}/map/{key}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """bbox = (west, south, east, north). Tiles son múltiplos de 3°."""
    west, south, east, north = bbox
    lat_start = math.floor(south / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    lat_end = (math.ceil(north / _TILE_SIZE_DEG) - 1) * _TILE_SIZE_DEG
    lon_start = math.floor(west / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    lon_end = (math.ceil(east / _TILE_SIZE_DEG) - 1) * _TILE_SIZE_DEG
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + 1, _TILE_SIZE_DEG)
        for lon in range(lon_start, lon_end + 1, _TILE_SIZE_DEG)
    ]
