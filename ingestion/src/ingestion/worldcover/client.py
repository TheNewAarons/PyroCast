"""Descarga de un tile de ESA WorldCover desde el bucket público de AWS.

Mismo patrón que ingestion/dem/client.py: bucket público sobre HTTPS
plano, sin retry/backoff (no hay límite de tasa documentado para
archivos estáticos), escritura atómica (.part + os.replace) desde el
principio — la revisión final de ingestion/dem encontró y corrigió la
ausencia de esto; se aplica la lección aquí de entrada.
"""
import os
from pathlib import Path

import requests

from ingestion.worldcover.tiles import tile_url


class WorldCoverDownloadError(RuntimeError):
    """Fallo al descargar un tile de ESA WorldCover."""


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
    response = sess.get(url, timeout=60)
    if response.status_code != 200:
        raise WorldCoverDownloadError(f"Error {response.status_code} descargando tile {key}: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, dest_path)
    return dest_path
