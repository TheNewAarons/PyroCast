"""Tests de persistencia de resultados de backtest en `model_run` +
`evaluation_result` (PostGIS). El insert real solo se ejercita contra un
Postgres real (guardado con skipif, mismo patrón que el resto de este
proyecto) -- lo demás se verifica sin red/DB."""
import os

import pytest
from models.evaluation.backtest import EventMetrics
from models.evaluation.db import persist_backtest_run
from shared.config import Settings
from shared.db.schema import evaluation_result, fire_event, metadata, model_run
from sqlalchemy import create_engine, insert, select, text


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

        metrics = EventMetrics(event_id=999, iou=0.5, dice=0.6, brier=0.1, ece=0.05)
        run_id = persist_backtest_run(
            engine=conn, firms_event_id=999, model_name="cellular_automata",
            config={"base_spread_prob": 0.3}, split="test", metrics=metrics,
        )

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
