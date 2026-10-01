"""Cliente ERA5-Land: construcción de la solicitud a cdsapi y manejo propio
de la cola asíncrona de CDS (envío, polling con timeout configurable,
descarga cuando está lista).

Verificado contra el código fuente real de cdsapi (2026-09-26): con
wait_until_complete=True (el default de la librería), el polling interno
de cdsapi NO tiene límite de tiempo total — solo bounded por
timeout/retry_max de cada petición HTTP individual, en un `while True`
sin salida por tiempo. Por eso este cliente usa wait_until_complete=False
y su propio bucle acotado.

Puede tardar minutos u horas según la carga del servicio CDS — el
timeout_seconds de download_hourly() es la única protección contra una
espera indefinida en un pipeline automatizado.

Ambas familias de objeto "remote" que cdsapi puede devolver (según el
formato del token del usuario) se soportan por duck typing:
- ecmwf.datastores Remote (moderno): remote.status en
  {accepted,running,successful,failed,rejected,dismissed,deleted} —
  verificado contra el código fuente instalado: es una property real,
  no deprecada. Se chequea PRIMERO.
- cdsapi.api.Result (clásico): remote.reply["state"] en
  {queued,running,completed,failed} — se usa solo si el objeto no tiene
  .status (el moderno Remote SÍ tiene .reply, pero es un alias
  retrocompatible deprecado que emite DeprecationWarning; chequear
  .status primero evita dispararlo innecesariamente).
"""
import calendar
import datetime as dt
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cdsapi
import requests

from ingestion.resilience import IngestionError, QuotaExceededError, redact

# request_name (usado por build_request/CDS) -> (nombre corto en el NetCDF
# resultante, método de agregación diaria). "carryover" es el caso especial
# de total_precipitation — ver ingestion/era5/aggregate.py: NO es una suma
# de las 24 muestras horarias (ERA5-Land acumula desde las 00 UTC de cada
# día; sumar las 24 muestras cuenta ~11-12x de más y mezcla el total del
# día anterior). Tabla explícita y cerrada: una variable no listada aquí
# hace que aggregate.py falle en vez de promediarla en silencio.
VARIABLE_SPEC: dict[str, tuple[str, str]] = {
    "2m_temperature": ("t2m", "mean"),
    "2m_dewpoint_temperature": ("d2m", "mean"),
    "10m_u_component_of_wind": ("u10", "mean"),
    "10m_v_component_of_wind": ("v10", "mean"),
    "total_precipitation": ("tp", "carryover"),
}
ERA5_VARIABLES: tuple[str, ...] = tuple(VARIABLE_SPEC.keys())

_SUCCESS_STATES = {"completed", "successful"}
_FAILURE_STATES = {"failed", "rejected", "dismissed", "deleted"}


SOURCE = "ERA5-Land (CDS)"
_QUOTA_PATTERN = re.compile(r"quota|too many|limit|429|cost", re.IGNORECASE)
_TRANSIENT_STATUS = re.compile(r"\b(500|502|503|504)\b")


class Era5RequestFailedError(IngestionError):
    """La solicitud a CDS terminó en un estado de fallo."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(SOURCE, message, hint)


class Era5RequestTimeoutError(IngestionError):
    """La solicitud a CDS no completó dentro del timeout configurado."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(
            SOURCE, message,
            hint or "CDS está encolado o saturado: reintenta más tarde o sube --timeout-seconds; "
                    "los tramos mensuales ya descargados se reutilizan.",
        )


class Era5QuotaExceededError(Era5RequestFailedError, QuotaExceededError):
    """CDS rechazó la solicitud por límite de cuota / cola."""


def _default_cds_client(url: str, key: str, wait_until_complete: bool) -> Any:
    # timeout/retry_max de cdsapi acotan CADA petición HTTP individual (el
    # polling total lo acota `timeout_seconds` de download_hourly).
    return cdsapi.Client(
        url=url, key=key, wait_until_complete=wait_until_complete,
        timeout=60, retry_max=3, quiet=True,
    )


def month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Divide [start, end] en tramos que no cruzan un límite de mes.

    CDS acepta year/month/day como listas independientes — un tramo que
    cruza un límite de mes/año produce el PRODUCTO CARTESIANO de esas
    listas (p. ej. year=[2025,2026] x month=[01,12] x day=[28..03]
    incluiría 2026-12-28, una fecha futura inexistente). Dividir por mes
    calendario elimina el problema en la fuente, en vez de confiar en el
    recorte posterior de aggregate.py (que protege el resultado agregado,
    pero no evita la sobre-solicitud ni un posible rechazo de CDS)."""
    if end < start:
        raise ValueError(f"end ({end}) es anterior a start ({start})")
    chunks: list[tuple[dt.date, dt.date]] = []
    chunk_start = start
    while chunk_start <= end:
        last_day_of_month = calendar.monthrange(chunk_start.year, chunk_start.month)[1]
        month_end = dt.date(chunk_start.year, chunk_start.month, last_day_of_month)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + dt.timedelta(days=1)
    return chunks


def build_request(
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    variables: tuple[str, ...] = ERA5_VARIABLES,
) -> dict[str, Any]:
    """Construye una única solicitud CDS. `start`/`end` deben caer dentro
    del mismo mes calendario — usar month_chunks() para partir un rango
    más largo antes de llamar a esta función una vez por tramo."""
    if end < start:
        raise ValueError(f"end ({end}) es anterior a start ({start})")
    if (start.year, start.month) != (end.year, end.month):
        raise ValueError(
            f"build_request requiere que start ({start}) y end ({end}) caigan "
            f"en el mismo mes calendario — usar month_chunks() primero."
        )
    west, south, east, north = bbox
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += dt.timedelta(days=1)
    days = sorted({d.strftime("%d") for d in dates})
    return {
        "variable": list(variables),
        "year": [start.strftime("%Y")],
        "month": [start.strftime("%m")],
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        # CDS usa [Norte, Oeste, Sur, Este] — distinto del orden
        # west,south,east,north usado en ingestion/firms y ingestion/dem.
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _state_of(remote: Any) -> str:
    status = getattr(remote, "status", None)
    if status is not None:
        return str(status)
    return str(remote.reply["state"])


class Era5Client:
    def __init__(
        self,
        url: str,
        key: str,
        client_factory: Callable[..., Any] = _default_cds_client,
        max_retries: int = 3,
        backoff_base_seconds: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = key
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._retry_sleep = sleep_fn
        self._client = client_factory(url=url, key=key, wait_until_complete=False)

    def _call(self, what: str, fn: Callable[[], Any]) -> Any:
        """Ejecuta una operación de cdsapi: reintenta fallas transitorias de
        red/5xx con backoff, y envuelve TODA excepción cruda en un error
        tipado (cuota / falla) sin la clave."""
        attempt = 0
        while True:
            try:
                return fn()
            except IngestionError:
                raise
            except Exception as exc:
                text = redact(f"{type(exc).__name__}: {exc}", [self._key])
                if _QUOTA_PATTERN.search(text):
                    raise Era5QuotaExceededError(
                        f"CDS rechazó {what} por cuota/límite de solicitudes: {text}",
                        hint="Espera a que terminen tus solicitudes en cola en "
                             "https://cds.climate.copernicus.eu/requests y reintenta.",
                    ) from None
                transient = isinstance(exc, requests.ConnectionError | requests.Timeout) or bool(
                    _TRANSIENT_STATUS.search(text)
                )
                if transient and attempt < self._max_retries:
                    self._retry_sleep(self._backoff_base * (2**attempt))
                    attempt += 1
                    continue
                raise Era5RequestFailedError(
                    f"Falló {what} tras {attempt} reintento(s): {text}",
                    hint="Revisa CDS_API_URL / CDS_API_KEY y que hayas aceptado la licencia "
                         "del dataset ERA5-Land en el sitio de CDS (docs/data-sources.md).",
                ) from None

    def download_hourly(
        self,
        dataset: str,
        request: dict[str, Any],
        target: Path,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 3600.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> Path:
        remote = self._call(
            "el envío de la solicitud", lambda: self._client.retrieve(dataset, request)
        )
        start = monotonic_fn()

        while True:
            state = _state_of(remote)

            if state in _SUCCESS_STATES:
                self._download_atomically(remote, target)
                return target

            if state in _FAILURE_STATES:
                raise Era5RequestFailedError(
                    f"Solicitud CDS terminó en estado {state!r}"
                )

            if monotonic_fn() - start > timeout_seconds:
                raise Era5RequestTimeoutError(
                    f"Solicitud CDS no completó en {timeout_seconds}s "
                    f"(último estado: {state!r})"
                )

            sleep_fn(poll_interval_seconds)
            self._call("la consulta de estado", remote.update)

    def _download_atomically(self, remote: Any, target: Path) -> None:
        # a .part y os.replace: una descarga cortada nunca deja un .nc
        # parcial que el pipeline tome por un tramo ya descargado.
        tmp = target.with_suffix(target.suffix + ".part")

        def _download() -> None:
            try:
                remote.download(str(tmp))
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

        self._call("la descarga del archivo", _download)
        os.replace(tmp, target)
