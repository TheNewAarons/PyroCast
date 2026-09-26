"""Cliente Sentinel-2 L2A vía openEO (Copernicus Data Space Ecosystem).

Verificado contra la documentación vigente (2026-09-26):
https://documentation.dataspace.copernicus.eu/APIs/openEO/ — elegido
sobre sentinelhub-py porque su autenticación por client credentials usa
solo client_id/client_secret (ya en shared/config.py), sin campos de
configuración adicionales (sentinelhub-py necesita además sh_base_url y
sh_token_url). Ver docs/decisions.md.

Composición mensual de menor nubosidad: se carga SENTINEL2_L2A con las
bandas B04 (rojo), B08 (NIR) y SCL (Scene Classification), se enmascaran
las clases de nube/sombra de SCL, y se reduce sobre el tiempo con la
mediana — el resultado se descarga como GeoTIFF de 3 bandas en ESE ORDEN
(B04, B08, SCL): es un contrato implícito con
features/vegetation/ndvi.py, que lee las bandas por posición.
"""
import calendar
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openeo

SENTINEL2_BANDS: tuple[str, ...] = ("B04", "B08", "SCL")
# Sombra de nube, nube prob. media, nube prob. alta, cirros delgados.
CLOUD_SCL_CLASSES: frozenset[int] = frozenset({3, 8, 9, 10})


class Sentinel2Client:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        connect_fn: Callable[[str], Any] = openeo.connect,
        backend_url: str = "openeofed.dataspace.copernicus.eu",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._connection = connect_fn(backend_url)
        self._connection.authenticate_oidc_client_credentials(
            client_id=client_id, client_secret=client_secret
        )

    def fetch_monthly_composite(
        self,
        bbox: tuple[float, float, float, float],
        year: int,
        month: int,
        target: Path,
        max_cloud_cover: int = 85,
    ) -> Path:
        west, south, east, north = bbox
        last_day = calendar.monthrange(year, month)[1]
        start = dt.date(year, month, 1).isoformat()
        end = dt.date(year, month, last_day).isoformat()

        datacube = self._connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=[start, end],
            bands=list(SENTINEL2_BANDS),
            max_cloud_cover=max_cloud_cover,
        )

        scl_band = datacube.band("SCL")
        cloud_classes = list(CLOUD_SCL_CLASSES)
        cloud_mask = scl_band != cloud_classes[0]
        for cloud_class in cloud_classes[1:]:
            cloud_mask = cloud_mask & (scl_band != cloud_class)
        mask_resampled = cloud_mask.resample_cube_spatial(datacube)
        masked = datacube.mask(mask_resampled)

        composite = masked.reduce_dimension(dimension="t", reducer="median")
        composite.download(str(target))
        return target
