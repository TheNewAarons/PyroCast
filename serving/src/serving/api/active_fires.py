"""Detecciones activas recientes de FIRMS dentro del bbox de trabajo,
solo como contexto para el mapa (no alimentan /predict)."""
from typing import Protocol

from ingestion.firms.client import FirmsApiError
from ingestion.firms.parser import parse_csv_to_detections
from shared.config import Settings

from serving.api.cache import LRUCache
from serving.api.errors import FirmsUnavailableError
from serving.api.schemas import ActiveFiresResponse

# NRT: el producto de tiempo casi real (el SP/archivo científico no cubre
# los últimos meses -- ver docs/backtest-2026.md sección 2).
ACTIVE_FIRES_SENSOR = "VIIRS_SNPP_NRT"
CACHE_TTL_SECONDS = 600.0  # FIRMS limita 5000 transacciones / 10 min por clave


class FirmsClientLike(Protocol):
    def fetch_area_csv(
        self, bbox: tuple[float, float, float, float], sensor: str, day_range: int,
    ) -> str: ...


class ActiveFiresService:
    def __init__(
        self, settings: Settings, client: FirmsClientLike, cache: LRUCache[ActiveFiresResponse]
    ) -> None:
        self._bbox = settings.study_area_bbox
        self._client = client
        self._cache = cache

    def get(self, days: int) -> ActiveFiresResponse:
        cached = self._cache.get(days)
        if cached is not None:
            return cached.model_copy(update={"cached": True})
        try:
            raw = self._client.fetch_area_csv(self._bbox, ACTIVE_FIRES_SENSOR, days)
            detections = parse_csv_to_detections(raw)
        except (FirmsApiError, ValueError) as exc:
            raise FirmsUnavailableError(
                f"No se pudo obtener detecciones de FIRMS: {exc}. No se devuelve una lista "
                f"vacía porque sería indistinguible de 'no hay incendios activos'."
            ) from exc
        response = ActiveFiresResponse(
            features=[
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [d.longitude, d.latitude]},
                    "properties": {
                        "detected_at": d.detected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "frp": d.frp,
                        "confidence": d.confidence,
                        "satellite": d.satellite,
                        "instrument": d.instrument,
                    },
                }
                for d in detections
            ],
            source=f"NASA FIRMS {ACTIVE_FIRES_SENSOR}",
            bbox=self._bbox, days=days,
        )
        self._cache.put(days, response)
        return response
