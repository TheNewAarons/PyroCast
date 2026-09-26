"""Clave de cache para un mosaico WorldCover procesado: hash de (bbox,
resolución, CRS, versión, año)."""
import hashlib


def cache_key_for(
    bbox: tuple[float, float, float, float], resolution_m: int, crs: str, version: str, year: str
) -> str:
    payload = f"{bbox}|{resolution_m}|{crs}|{version}|{year}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
