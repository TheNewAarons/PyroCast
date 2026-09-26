"""Clave de cache para una composición mensual de Sentinel-2: hash de
(bbox, año, mes)."""
import hashlib


def cache_key_for(bbox: tuple[float, float, float, float], year: int, month: int) -> str:
    payload = f"{bbox}|{year:04d}-{month:02d}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
