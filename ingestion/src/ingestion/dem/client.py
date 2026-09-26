"""Descarga de un tile de Copernicus DEM desde el bucket público de AWS.

Bucket público sobre HTTPS plano — sin credenciales AWS, sin boto3. A
diferencia de FIRMS, esta fuente no tiene un límite de tasa documentado
para descargas de archivos estáticos, así que no se implementa
retry/backoff aquí (YAGNI); un fallo se reporta de inmediato.
"""
import os
from pathlib import Path

import requests

from ingestion.dem.tiles import tile_url


class DemDownloadError(RuntimeError):
    """Fallo al descargar un tile de Copernicus DEM."""


def download_tile(key: str, dest_path: Path, session: requests.Session | None = None) -> Path:
    if dest_path.exists():
        return dest_path

    sess = session or requests.Session()
    response = sess.get(tile_url(key), timeout=60)
    if response.status_code != 200:
        raise DemDownloadError(
            f"Error {response.status_code} descargando tile {key}: {tile_url(key)}"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica: si el proceso se interrumpe a mitad de la
    # descarga (los tiles pesan ~40 MB), un archivo .tif truncado se
    # trataría como cache-hit válido para siempre (ver download_tile
    # arriba: solo chequea existencia). Se escribe a un .part y se
    # reemplaza solo al completar.
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, dest_path)
    return dest_path
