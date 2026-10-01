"""Robustez común de la ingesta: taxonomía de errores, descarga HTTP con
timeouts + reintentos con backoff + manejo explícito de cuota, y un
decorador de CLI que convierte cualquier falla de una fuente externa en un
mensaje claro (qué fuente, qué pasó, qué hacer) en vez de un traceback.

Principios (docs/decisions.md, "Endurecimiento de la ingesta"):
- Una falla transitoria (timeout, corte de conexión, 5xx, 429) se reintenta
  con backoff exponencial acotado; si se agota, se levanta un error TIPADO
  (`SourceUnavailableError` / `QuotaExceededError`) que nombra la fuente.
- Una excepción cruda de `requests`/`cdsapi`/`openeo` nunca llega al
  usuario: siempre se envuelve.
- Los secretos nunca aparecen en un mensaje (`redact`).
- Un archivo parcial nunca queda en el caché (los clientes escriben a
  `.part` y reemplazan al completar).
"""
import functools
import time
from collections.abc import Callable, Iterable
from typing import Any

import requests
import typer

DEFAULT_TIMEOUT: tuple[float, float] = (10.0, 60.0)  # (conexión, lectura) en segundos
RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})
EXIT_FAILURE = 1
EXIT_QUOTA = 3


class IngestionError(RuntimeError):
    """Falla de una fuente externa con un mensaje accionable."""

    def __init__(self, source: str, message: str, hint: str = "") -> None:
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message
        self.hint = hint


class SourceUnavailableError(IngestionError):
    """La fuente no respondió (red, timeout, 5xx) tras agotar los reintentos."""


class QuotaExceededError(IngestionError):
    """La fuente rechazó la solicitud por cuota / límite de uso agotado."""


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<REDACTED>")
    return text


def _retry_after_seconds(response: requests.Response, cap: float) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return min(max(float(raw), 0.0), cap)
    except ValueError:  # formato fecha HTTP: no se interpreta, se usa el backoff
        return None


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    source: str,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = 4,
    backoff_base: float = 1.0,
    retry_after_cap: float = 60.0,
    sleep_fn: Callable[[float], None] | None = None,
    secrets: Iterable[str] = (),
) -> requests.Response:
    """GET con reintentos. Devuelve la respuesta para CUALQUIER estado no
    reintentable (200, 404, 403...): el llamador decide qué significa. Se
    reintenta 429, 5xx y errores de transporte; agotados los reintentos se
    levanta `QuotaExceededError` (429) o `SourceUnavailableError`."""
    sleeper = sleep_fn if sleep_fn is not None else time.sleep
    secret_list = list(secrets)
    attempt = 0
    while True:
        wait = backoff_base * (2**attempt)
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise SourceUnavailableError(
                    source,
                    f"sin respuesta tras {attempt} reintento(s): "
                    f"{redact(str(exc), secret_list)}",
                    hint="Revisa tu conexión y reintenta; lo ya descargado quedó en caché.",
                ) from None
        else:
            status = response.status_code
            if status != 429 and status not in RETRYABLE_STATUSES:
                return response
            if attempt >= max_retries:
                if status == 429:
                    raise QuotaExceededError(
                        source,
                        f"cuota o límite de uso agotado (HTTP 429) tras {attempt} reintento(s).",
                        hint="Espera a que se restablezca la cuota (suele ser minutos) y "
                             "reintenta; lo ya descargado quedó en caché.",
                    )
                raise SourceUnavailableError(
                    source,
                    f"el servicio respondió HTTP {status} tras {attempt} reintento(s).",
                    hint="El servicio externo falla; reintenta más tarde.",
                )
            if status == 429:
                retry_after = _retry_after_seconds(response, retry_after_cap)
                wait = retry_after if retry_after is not None else wait
        sleeper(min(wait, retry_after_cap))
        attempt += 1


def handle_errors[F: Callable[..., Any]](func: F) -> F:
    """Decorador de comandos Typer: una `IngestionError` (o una excepción de
    red que se haya escapado) se imprime como mensaje claro y termina con
    código 1 (3 si fue por cuota), sin traceback ni variables locales."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except IngestionError as exc:
            typer.secho(f"ERROR {exc}", err=True, fg=typer.colors.RED)
            if exc.hint:
                typer.echo(f"Qué hacer: {exc.hint}", err=True)
            raise typer.Exit(
                code=EXIT_QUOTA if isinstance(exc, QuotaExceededError) else EXIT_FAILURE
            ) from None
        except requests.RequestException as exc:
            typer.secho(
                f"ERROR [red] {type(exc).__name__}: se escapó un error de red sin envolver "
                f"(es un bug de PyroCast, reportarlo).",
                err=True, fg=typer.colors.RED,
            )
            raise typer.Exit(code=EXIT_FAILURE) from None

    return wrapper  # type: ignore[return-value]
