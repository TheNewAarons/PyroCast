"""Cliente para el Area API de NASA FIRMS (detecciones activas de fuego).

Referencia verificada contra la documentación vigente (2026-09-26):
https://firms.modaps.eosdis.nasa.gov/api/area/ — no asumida de memoria.

URL: https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/
     [west,south,east,north]/[DAY_RANGE]/[DATE]
- DAY_RANGE: entero 1..5 (límite duro de la API).
- DATE (opcional): primer día del rango devuelto; el rango es
  [DATE, DATE + DAY_RANGE - 1].
- Límite de la API: 5000 transacciones / ventana de 10 minutos por
  MAP_KEY.

El comportamiento documentado ante errores es débil: no hay códigos de
estado documentados oficialmente para un MAP_KEY inválido o una
solicitud malformada. Este cliente no confía en suposiciones no
verificadas: trata 429/5xx como reintentables, y valida que un cuerpo
200 sea CSV real (encabezado esperado o vacío) antes de intentar
parsearlo — cualquier otra cosa (texto de error de una línea, HTML,
etc.) se levanta de inmediato como FirmsApiError, sin reintentar.
"""
import datetime as dt
import time
from collections.abc import Callable, Iterator

import requests

_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_EXPECTED_HEADER_PREFIX = "latitude,longitude"
_MIN_DAY_RANGE = 1
_MAX_DAY_RANGE = 5
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class FirmsApiError(RuntimeError):
    """Error no reintentable del Area API de FIRMS (clave inválida,
    cuerpo de respuesta que no es CSV, o reintentos agotados)."""


class FirmsClient:
    def __init__(
        self,
        map_key: str,
        session: requests.Session | None = None,
        max_retries: int = 4,
        backoff_base_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.25,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._map_key = map_key
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._min_request_interval_seconds = min_request_interval_seconds
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._last_request_at: float | None = None

    def _redact(self, message: str) -> str:
        return message.replace(self._map_key, "<MAP_KEY>")

    def _rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._monotonic_fn() - self._last_request_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            self._sleep_fn(remaining)

    def fetch_area_csv(
        self,
        bbox: tuple[float, float, float, float],
        sensor: str,
        day_range: int,
        date: dt.date | None = None,
    ) -> str:
        if not (_MIN_DAY_RANGE <= day_range <= _MAX_DAY_RANGE):
            raise ValueError(
                f"day_range debe estar entre {_MIN_DAY_RANGE} y {_MAX_DAY_RANGE} "
                f"(límite del Area API de FIRMS), recibido: {day_range}"
            )
        west, south, east, north = bbox
        coords = f"{west},{south},{east},{north}"
        url = f"{_BASE_URL}/{self._map_key}/{sensor}/{coords}/{day_range}"
        if date is not None:
            url = f"{url}/{date.isoformat()}"

        attempt = 0
        while True:
            self._rate_limit()
            self._last_request_at = self._monotonic_fn()
            try:
                response = self._session.get(url, timeout=30)
            except requests.RequestException as exc:
                if attempt < self._max_retries:
                    self._sleep_fn(self._backoff_base_seconds * (2**attempt))
                    attempt += 1
                    continue
                # from None: requests' own exception message embeds the
                # full URL (MAP_KEY included) — a chained traceback would
                # re-leak it even though this message is redacted.
                raise FirmsApiError(
                    f"Fallo de transporte tras {attempt} reintento(s): "
                    f"{self._redact(str(exc))}"
                ) from None

            if response.status_code == 200:
                body = response.text
                if body == "" or body.lstrip().lower().startswith(_EXPECTED_HEADER_PREFIX):
                    return body
                raise FirmsApiError(
                    f"Respuesta 200 de FIRMS no parece CSV válido: {body[:200]!r}"
                )

            if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_retries:
                self._sleep_fn(self._backoff_base_seconds * (2**attempt))
                attempt += 1
                continue

            raise FirmsApiError(
                f"FIRMS Area API devolvió estado {response.status_code} tras "
                f"{attempt} reintento(s): {self._redact(response.text[:200])!r}"
            )

    def _chunk_date_range(self, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
        if end < start:
            raise ValueError(f"end ({end}) es anterior a start ({start})")
        chunks: list[tuple[dt.date, dt.date]] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + dt.timedelta(days=_MAX_DAY_RANGE - 1), end)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + dt.timedelta(days=1)
        return chunks

    def fetch_range(
        self,
        bbox: tuple[float, float, float, float],
        sensor: str,
        start: dt.date,
        end: dt.date,
    ) -> Iterator[tuple[dt.date, dt.date, str]]:
        # Generador, no lista: si un chunk falla a mitad de un rango largo,
        # los chunks ya descargados deben quedar persistidos por el
        # llamador (ver ingestion/firms/cli.py) en vez de perderse junto
        # con el resto de la respuesta.
        for chunk_start, chunk_end in self._chunk_date_range(start, end):
            day_range = (chunk_end - chunk_start).days + 1
            raw = self.fetch_area_csv(bbox, sensor=sensor, day_range=day_range, date=chunk_start)
            yield (chunk_start, chunk_end, raw)
