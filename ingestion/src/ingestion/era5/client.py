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
- cdsapi.api.Result (clásico): remote.reply["state"] en
  {queued,running,completed,failed}
- ecmwf.datastores Remote (moderno): remote.status en
  {accepted,running,successful,failed,rejected,dismissed,deleted}
"""
import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cdsapi

ERA5_VARIABLES: tuple[str, ...] = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
)

_SUCCESS_STATES = {"completed", "successful"}
_FAILURE_STATES = {"failed", "rejected", "dismissed", "deleted"}


class Era5RequestFailedError(RuntimeError):
    """La solicitud a CDS terminó en un estado de fallo."""


class Era5RequestTimeoutError(RuntimeError):
    """La solicitud a CDS no completó dentro del timeout configurado."""


def build_request(
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    variables: tuple[str, ...] = ERA5_VARIABLES,
) -> dict[str, Any]:
    if end < start:
        raise ValueError(f"end ({end}) es anterior a start ({start})")
    west, south, east, north = bbox
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += dt.timedelta(days=1)
    years = sorted({d.strftime("%Y") for d in dates})
    months = sorted({d.strftime("%m") for d in dates})
    days = sorted({d.strftime("%d") for d in dates})
    return {
        "variable": list(variables),
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        # CDS usa [Norte, Oeste, Sur, Este] — distinto del orden
        # west,south,east,north usado en ingestion/firms y ingestion/dem.
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _state_of(remote: Any) -> str:
    reply = getattr(remote, "reply", None)
    if reply is not None:
        return str(reply["state"])
    return str(remote.status)


class Era5Client:
    def __init__(
        self,
        url: str,
        key: str,
        client_factory: Callable[..., Any] = cdsapi.Client,
    ) -> None:
        self._client = client_factory(url=url, key=key, wait_until_complete=False)

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
        remote = self._client.retrieve(dataset, request)
        start = monotonic_fn()

        while True:
            state = _state_of(remote)

            if state in _SUCCESS_STATES:
                remote.download(str(target))
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
            remote.update()
