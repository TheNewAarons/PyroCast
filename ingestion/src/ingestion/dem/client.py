"""Descarga de un tile de Copernicus DEM desde el bucket público de AWS.

Bucket público sobre HTTPS plano — sin credenciales AWS, sin boto3. Sin
límite de tasa documentado, pero la red falla: timeouts (conexión/lectura),
reintentos con backoff ante 5xx/429/cortes (`ingestion.resilience`), y
verificación de que la descarga no llegó truncada.

Taxonomía: un 404 es `TileNotFoundError` (hueco de cobertura u océano:
tolerable, ver `ingestion/dem/pipeline.py`); una falla de red agotada es
`SourceUnavailableError` (NO tolerable: dejaría un hueco silencioso en el
mosaico).
"""
import os
from pathlib import Path

import requests

from ingestion.dem.tiles import tile_url
from ingestion.resilience import IngestionError, get_with_retry

SOURCE = "Copernicus DEM"


class DemDownloadError(IngestionError):
    """Fallo al descargar un tile de Copernicus DEM."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(SOURCE, message, hint)


class TileNotFoundError(DemDownloadError):
    """El tile no existe en el bucket (HTTP 404)."""


def download_tile(key: str, dest_path: Path, session: requests.Session | None = None) -> Path:
    if dest_path.exists():
        return dest_path

    sess = session or requests.Session()
    url = tile_url(key)
    response = get_with_retry(sess, url, source=SOURCE)
    if response.status_code == 404:
        raise TileNotFoundError(f"Error 404 descargando tile {key}: {url}")
    if response.status_code != 200:
        raise DemDownloadError(
            f"Error {response.status_code} descargando tile {key}: {url}",
            hint="Estado HTTP inesperado; reintenta o revisa docs/data-sources.md.",
        )
    expected = response.headers.get("Content-Length")
    if expected is not None and expected.isdigit() and int(expected) != len(response.content):
        raise DemDownloadError(
            f"Descarga truncada del tile {key}: {len(response.content)} de {expected} bytes.",
            hint="Reintenta; el tile truncado no se guardó en caché.",
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica: si el proceso se interrumpe a mitad de la
    # descarga (los tiles pesan ~40 MB), un archivo .tif truncado se
    # trataría como cache-hit válido para siempre (download_tile arriba
    # solo chequea existencia). Se escribe a un .part y se reemplaza solo
    # al completar.
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, dest_path)
    return dest_path
