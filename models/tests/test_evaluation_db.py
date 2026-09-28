"""Tests de persistencia de resultados de backtest en `model_run` +
`evaluation_result` (PostGIS). El roundtrip completo (con `fire_event`,
que usa una columna `Geometry` que sqlite no soporta) solo se ejercita
contra un Postgres real (guardado con skipif, mismo patrón que el resto
de este proyecto) -- pero la lógica de `persist_backtest_run` en sí
(resolución de firms_event_id, atomicidad, el error de evento
desconocido) SÍ corre de verdad contra sqlite en memoria, sin red/DB
real: `model_run`/`evaluation_result` no usan `Geometry`, y una tabla
`fire_event` mínima (solo las columnas `id`/`firms_event_id` que
`persist_backtest_run` realmente consulta) es suficiente para ejercitar
el código real sin necesitar PostGIS. Encontrado en la revisión final
del 2026-09-28: antes de este test, `persist_backtest_run` no tenía
NINGUNA cobertura ejecutada en la suite por defecto."""
import os

import pytest
from models.evaluation.backtest import EventMetrics
from models.evaluation.db import UnknownFireEventError, persist_backtest_run
from shared.config import Settings
from shared.db.schema import evaluation_result, fire_event, metadata, model_run
from sqlalchemy import Engine, create_engine, insert, select, text


def _live_settings() -> Settings | None:
    required = [
        "FIRMS_MAP_KEY", "CDS_API_URL", "CDS_API_KEY",
        "COPERNICUS_DATASPACE_CLIENT_ID", "COPERNICUS_DATASPACE_CLIENT_SECRET",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD",
    ]
    if not all(os.getenv(k) for k in required):
        return None
    return Settings()


@pytest.mark.skipif(_live_settings() is None, reason="requiere POSTGRES_* de un contenedor real")
def test_persist_backtest_run_roundtrip_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)

        fire_event_id = conn.execute(
            insert(fire_event).values(
                bbox="-73.0,-38.0,-72.0,-37.0", start_date="2026-01-10",
                end_date="2026-01-15", source="test", firms_event_id=999,
                geom="SRID=4326;POLYGON((-73 -38, -73 -37, -72 -37, -72 -38, -73 -38))",
            ).returning(fire_event.c.id)
        ).scalar_one()

        metrics = [EventMetrics(event_id=999, iou=0.5, dice=0.6, brier=0.1, ece=0.05)]
        run_ids = persist_backtest_run(
            engine=conn, per_event=metrics, model_name="cellular_automata",
            config={"base_spread_prob": 0.3}, split="test",
        )
        run_id = run_ids[0]

        stored_model_name = conn.execute(
            select(model_run.c.model_name).where(model_run.c.id == run_id)
        ).scalar_one()
        assert stored_model_name == "cellular_automata"
        stored_event_id = conn.execute(
            select(model_run.c.event_id).where(model_run.c.id == run_id)
        ).scalar_one()
        assert stored_event_id == fire_event_id

        stored_metrics = conn.execute(
            select(evaluation_result.c.metric_name, evaluation_result.c.value)
            .where(evaluation_result.c.run_id == run_id)
        ).all()
        stored = {row.metric_name: row.value for row in stored_metrics}
        assert stored == {"iou": 0.5, "dice": 0.6, "brier": 0.1, "ece": 0.05}
    finally:
        trans.rollback()
        conn.close()


def _sqlite_engine_with_fire_events(firms_event_ids: list[int]) -> Engine:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE fire_event "
                "(id INTEGER PRIMARY KEY, firms_event_id INTEGER UNIQUE)"
            )
        )
        for i, firms_event_id in enumerate(firms_event_ids, start=1):
            conn.execute(
                text("INSERT INTO fire_event (id, firms_event_id) VALUES (:id, :firms_event_id)"),
                {"id": i, "firms_event_id": firms_event_id},
            )
    metadata.create_all(engine, tables=[model_run, evaluation_result])
    return engine


def test_persist_backtest_run_writes_one_model_run_and_four_evaluation_results_per_event():
    engine = _sqlite_engine_with_fire_events([100, 200])
    per_event = [
        EventMetrics(event_id=100, iou=0.5, dice=0.6, brier=0.1, ece=0.05),
        EventMetrics(event_id=200, iou=0.7, dice=0.8, brier=0.2, ece=0.15),
    ]
    run_ids = persist_backtest_run(
        engine=engine, per_event=per_event, model_name="cellular_automata",
        config={"base_spread_prob": 0.3}, split="test",
    )
    assert len(run_ids) == 2

    with engine.connect() as conn:
        model_names = conn.execute(select(model_run.c.model_name)).scalars().all()
        assert model_names == ["cellular_automata", "cellular_automata"]
        metric_rows = conn.execute(select(evaluation_result.c.run_id)).scalars().all()
        assert len(metric_rows) == 8  # 4 métricas x 2 eventos


def test_persist_backtest_run_raises_a_named_error_for_an_unknown_firms_event_id():
    engine = _sqlite_engine_with_fire_events([100])
    per_event = [EventMetrics(event_id=999, iou=0.5, dice=0.6, brier=0.1, ece=0.05)]
    with pytest.raises(UnknownFireEventError, match="999"):
        persist_backtest_run(
            engine=engine, per_event=per_event, model_name="cellular_automata",
            config={}, split="test",
        )


def test_persist_backtest_run_is_atomic_one_bad_event_writes_nothing():
    engine = _sqlite_engine_with_fire_events([100])
    per_event = [
        EventMetrics(event_id=100, iou=0.5, dice=0.6, brier=0.1, ece=0.05),
        EventMetrics(event_id=999, iou=0.7, dice=0.8, brier=0.2, ece=0.15),  # no existe
    ]
    with pytest.raises(UnknownFireEventError):
        persist_backtest_run(
            engine=engine, per_event=per_event, model_name="cellular_automata",
            config={}, split="test",
        )

    with engine.connect() as conn:
        count = conn.execute(select(model_run.c.id)).scalars().all()
        assert count == []  # nada se escribió -- ni siquiera el evento 100 válido
