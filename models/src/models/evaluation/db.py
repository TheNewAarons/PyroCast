"""Persiste un resultado de backtest en `model_run` + `evaluation_result`
(PostGIS). `model_run.event_id` es el `fire_event.id` serial (la PK real
de la tabla), NO el `firms_event_id` (hash de contenido) que trae el
tensor -- hay que resolverlo primero. Cada corrida de backtest es un
registro de experimento nuevo (a diferencia del upsert de
`features/dataset/db.py::persist_fire_event_metadata`): re-correr un
backtest no "corrige" el anterior, agrega otro.

Todo el backtest (todos los eventos) se persiste en UNA transacción: si
un `firms_event_id` no existe en `fire_event` (p. ej. los Zarr
sobrevivieron a una base de datos recreada), `UnknownFireEventError` se
levanta ANTES de escribir cualquier fila, no a mitad de camino --
encontrado en la revisión final del 2026-09-28: la versión anterior
abría una transacción por evento, dejando `model_run` parcialmente
poblado si un evento a mitad de la lista fallaba."""
from typing import Any

from shared.db.schema import evaluation_result, fire_event, model_run
from sqlalchemy import Connection, Engine, insert, select

from models.evaluation.backtest import EventMetrics

_METRIC_NAMES = ("iou", "dice", "brier", "ece")


class UnknownFireEventError(RuntimeError):
    """`firms_event_id` no existe en `fire_event` -- probablemente el
    Zarr sobrevivió a una base de datos recreada, o `pyrocast-features
    build-dataset` nunca corrió para ese evento. Correr `pyrocast-features
    build-dataset` para repoblar `fire_event` antes de reintentar."""


def _resolve_fire_event_id(conn: Any, firms_event_id: int) -> int:
    row_id = conn.execute(
        select(fire_event.c.id).where(fire_event.c.firms_event_id == firms_event_id)
    ).scalar_one_or_none()
    if row_id is None:
        raise UnknownFireEventError(
            f"firms_event_id={firms_event_id} no existe en fire_event -- "
            "correr `pyrocast-features build-dataset` para repoblarlo."
        )
    result: int = row_id
    return result


def persist_backtest_run(
    engine: Engine | Connection,
    per_event: list[EventMetrics],
    model_name: str,
    config: dict[str, object],
    split: str,
) -> list[int]:
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            return _persist_all(conn, per_event, model_name, config, split)
    return _persist_all(engine, per_event, model_name, config, split)


def _persist_all(
    conn: Any,
    per_event: list[EventMetrics],
    model_name: str,
    config: dict[str, object],
    split: str,
) -> list[int]:
    run_ids = []
    for metrics in per_event:
        resolved_event_id = _resolve_fire_event_id(conn, metrics.event_id)
        run_id = conn.execute(
            insert(model_run)
            .values(event_id=resolved_event_id, model_name=model_name, config=config)
            .returning(model_run.c.id)
        ).scalar_one()
        for metric_name in _METRIC_NAMES:
            conn.execute(
                insert(evaluation_result).values(
                    run_id=run_id,
                    metric_name=metric_name,
                    value=getattr(metrics, metric_name),
                    split=split,
                )
            )
        run_ids.append(run_id)
    return run_ids
