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
mediana. SCL se usa SOLO para construir la máscara server-side y se
descarta antes de la reducción temporal — es un código categórico, y
una mediana temporal sobre él fabricaría clases inexistentes (p. ej.
mediana de observaciones [4, 8] = 6.0, "Bare soil", que no ocurrió en
ninguna observación real). El GeoTIFF descargado tiene 2 bandas, en ESE
ORDEN (B04, B08): es un contrato implícito con
features/vegetation/ndvi.py, que las lee por posición.

Dirección de la máscara: openEO `mask(mask_cube)` reemplaza por nodata
los píxeles donde `mask_cube` es `true`/no-cero (process spec oficial de
`mask`) — así que `cloud_mask` se construye "true = ES nube/sombra/
cirros", nunca "true = está claro" (esto último enmascararía lo claro y
dejaría pasar la nube, exactamente al revés).
"""
import datetime as dt
import os
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
        start = dt.date(year, month, 1).isoformat()
        # temporal_extent es EXCLUSIVO en su límite superior (spec oficial
        # de load_collection: "the specified time instant is excluded from
        # the interval") -- el límite es el primer día del mes SIGUIENTE,
        # no el último día de este mes (eso perdería ese día completo).
        end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
        end = dt.date(end_year, end_month, 1).isoformat()

        datacube = self._connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=[start, end],
            bands=list(SENTINEL2_BANDS),
            max_cloud_cover=max_cloud_cover,
        )

        scl_band = datacube.band("SCL")
        cloud_classes = list(CLOUD_SCL_CLASSES)
        cloud_mask = scl_band == cloud_classes[0]
        for cloud_class in cloud_classes[1:]:
            cloud_mask = cloud_mask | (scl_band == cloud_class)
        mask_resampled = cloud_mask.resample_cube_spatial(datacube)
        masked = datacube.mask(mask_resampled)

        reflectance = masked.filter_bands(["B04", "B08"])
        composite = reflectance.reduce_dimension(dimension="t", reducer="median")

        tmp_target = target.with_suffix(target.suffix + ".part")
        # format="GTiff" explícito -- sin esto, openEO adivina el formato
        # de salida desde la EXTENSIÓN del archivo de destino, y el
        # destino real acá es la ruta temporal ".part" (para el rename
        # atómico de abajo), no ".tif". Adivinar desde ".tif.part" falla
        # con "Invalid format 'PART'" -- reproducido contra la API real
        # de Copernicus Data Space (revisión final del 2026-09-29).
        composite.download(str(tmp_target), format="GTiff")
        os.replace(tmp_target, target)
        return target
