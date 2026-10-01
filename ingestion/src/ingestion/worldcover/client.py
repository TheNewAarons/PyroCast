"""Descarga de un tile de ESA WorldCover desde el bucket público de AWS.

Mismo patrón que ingestion/dem/client.py: bucket público sobre HTTPS
plano, timeouts + reintentos con backoff (`ingestion.resilience`),
verificación de descarga truncada y escritura atómica (.part +
os.replace). 404 -> `WorldCoverTileNotFoundError` (tolerable); falla de
red agotada -> `SourceUnavailableError` (no tolerable).
"""
import os
from pathlib import Path

import requests

from ingestion.resilience import IngestionError, get_with_retry
from ingestion.worldcover.tiles import tile_url

SOURCE = "ESA WorldCover"


class WorldCoverDownloadError(IngestionError):
    """Fallo al descargar un tile de ESA WorldCover."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(SOURCE, message, hint)


class WorldCoverTileNotFoundError(WorldCoverDownloadError):
    """El tile no existe en el bucket (HTTP 404)."""


def download_tile(
    key: str,
    dest_path: Path,
    version: str = "v200",
    year: str = "2021",
    session: requests.Session | None = None,
) -> Path:
    if dest_path.exists():
        return dest_path

    sess = session or requests.Session()
    url = tile_url(key, version=version, year=year)
    response = get_with_retry(sess, url, source=SOURCE)
    if response.status_code == 404:
        raise WorldCoverTileNotFoundError(f"Error 404 descargando tile {key}: {url}")
    if response.status_code != 200:
        raise WorldCoverDownloadError(
            f"Error {response.status_code} descargando tile {key}: {url}",
            hint="Estado HTTP inesperado; reintenta o revisa docs/data-sources.md.",
        )
    expected = response.headers.get("Content-Length")
    if expected is not None and expected.isdigit() and int(expected) != len(response.content):
        raise WorldCoverDownloadError(
            f"Descarga truncada del tile {key}: {len(response.content)} de {expected} bytes.",
            hint="Reintenta; el tile truncado no se guardó en caché.",
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, dest_path)
    return dest_path
