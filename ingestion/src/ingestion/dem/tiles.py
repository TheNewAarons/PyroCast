"""Grilla de tiles de Copernicus DEM GLO-30 (bucket público de AWS).

Cada tile cubre exactamente 1°x1°, nombrado por su esquina suroeste.
Verificado contra el bucket real (2026-09-26): ver docs/data-sources.md
y el encabezado de docs/superpowers/plans/2026-09-26-ingestion-dem.md
para la evidencia (curl contra tileList.txt y una URL de tile real).
"""
import math

_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
_TILE_FORMAT_CODE = "10"  # 1.0 arco-segundo == GLO-30 (~30 m)


def tile_key(lat: int, lon: int) -> str:
    """lat/lon son la esquina SO del tile, en grados enteros."""
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return (
        f"Copernicus_DSM_COG_{_TILE_FORMAT_CODE}_{ns}{abs(lat):02d}_00_"
        f"{ew}{abs(lon):03d}_00_DEM"
    )


def tile_url(key: str) -> str:
    return f"{_BASE_URL}/{key}/{key}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """bbox = (west, south, east, north). Devuelve pares (lat, lon) de la
    esquina SO de cada tile 1x1 que intersecta el bbox.

    Cada tile cubre [lat, lat+1) x [lon, lon+1) — semiabierto. Por eso el
    límite superior usa ceil(x)-1, no floor(x): si north (o east) cae
    justo en un entero N, el tile que empieza en N cubre [N, N+1) y no
    se solapa con el bbox en absoluto (solapamiento de ancho cero en su
    borde sur/oeste) — incluirlo pediría un tile de más sin ninguna
    cobertura real, y en la práctica falla con 404 cuando ese tile cae
    en el océano (frecuente con bboxes de números redondos)."""
    west, south, east, north = bbox
    lat_start = math.floor(south)
    lat_end = math.ceil(north) - 1
    lon_start = math.floor(west)
    lon_end = math.ceil(east) - 1
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + 1)
        for lon in range(lon_start, lon_end + 1)
    ]
