"""Clave de cache para un DEM procesado: hash de (bbox, resolución, CRS)."""
import hashlib


def cache_key_for(bbox: tuple[float, float, float, float], resolution_m: int, crs: str) -> str:
    payload = f"{bbox}|{resolution_m}|{crs}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
