"""Persiste un resultado de backtest en `model_run` + `evaluation_result`
(PostGIS). `model_run.event_id` es el `fire_event.id` serial (la PK real
de la tabla), NO el `firms_event_id` (hash de contenido) que trae el
tensor -- hay que resolverlo primero. Cada corrida de backtest es un
registro de experimento nuevo (a diferencia del upsert de
`features/dataset/db.py::persist_fire_event_metadata`): re-correr un
backtest no "corrige" el anterior, agrega otro."""
from typing import Any

from shared.db.schema import evaluation_result, fire_event, model_run
from sqlalchemy import Connection, Engine, insert, select

from models.evaluation.backtest import EventMetrics


def _resolve_fire_event_id(conn: Any, firms_event_id: int) -> int:
    row_id = conn.execute(
        select(fire_event.c.id).where(fire_event.c.firms_event_id == firms_event_id)
    ).scalar_one()
    result: int = row_id
    return result


def persist_backtest_run(
    engine: Engine | Connection,
    firms_event_id: int,
    model_name: str,
    config: dict[str, float],
    split: str,
    metrics: EventMetrics,
) -> int:
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            return _persist(conn, firms_event_id, model_name, config, split, metrics)
    return _persist(engine, firms_event_id, model_name, config, split, metrics)


def _persist(
    conn: Any,
    firms_event_id: int,
    model_name: str,
    config: dict[str, float],
    split: str,
    metrics: EventMetrics,
) -> int:
    resolved_event_id = _resolve_fire_event_id(conn, firms_event_id)
    run_id = conn.execute(
        insert(model_run)
        .values(event_id=resolved_event_id, model_name=model_name, config=config)
        .returning(model_run.c.id)
    ).scalar_one()
    for metric_name in ("iou", "dice", "brier", "ece"):
        conn.execute(
            insert(evaluation_result).values(
                run_id=run_id,
                metric_name=metric_name,
                value=getattr(metrics, metric_name),
                split=split,
            )
        )
    result: int = run_id
    return result
