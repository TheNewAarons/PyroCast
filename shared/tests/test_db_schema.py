"""Verifica la forma de las tablas Core (sin conexión real a Postgres)."""
import os

import pytest
from shared.config import Settings
from shared.db.schema import evaluation_result, fire_event, metadata, model_run
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.exc import IntegrityError


def test_metadata_has_expected_tables():
    assert set(metadata.tables) == {
        "fire_event",
        "model_run",
        "evaluation_result",
    }


def test_fire_event_columns():
    cols = set(fire_event.columns.keys())
    assert {"id", "bbox", "start_date", "end_date", "source", "geom"} <= cols


def test_model_run_has_fk_to_fire_event():
    fk_targets = {fk.column.table.name for fk in model_run.foreign_keys}
    assert fk_targets == {"fire_event"}


def test_evaluation_result_has_fk_to_model_run():
    fk_targets = {fk.column.table.name for fk in evaluation_result.foreign_keys}
    assert fk_targets == {"model_run"}


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
def test_fire_event_model_run_evaluation_result_roundtrip_against_real_postgis():
    """Inserta las tres tablas encadenadas (FK + columna JSON) contra un
    Postgres real y confirma que el tipo JSON hace round-trip como dict y
    que una FK inválida efectivamente falla. Todo dentro de una transacción
    que se revierte al final: no deja filas persistidas, así el test es
    repetible sin acumular datos entre corridas."""
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)

        event_id = conn.execute(
            insert(fire_event)
            .values(
                bbox="-73.7,-39.3,-71.0,-36.5",
                start_date="2026-01-15",
                end_date="2026-01-20",
                source="test",
                geom="SRID=4326;POLYGON((-73 -39, -73 -38, -72 -38, -72 -39, -73 -39))",
            )
            .returning(fire_event.c.id)
        ).scalar_one()

        run_id = conn.execute(
            insert(model_run)
            .values(
                event_id=event_id,
                model_name="cellular_automata",
                config={"cell_size_m": 250, "wind_weight": 0.6},
            )
            .returning(model_run.c.id)
        ).scalar_one()

        conn.execute(
            insert(evaluation_result).values(
                run_id=run_id,
                metric_name="iou",
                value=0.42,
                split="val",
            )
        )

        stored_config = conn.execute(
            select(model_run.c.config).where(model_run.c.id == run_id)
        ).scalar_one()
        assert stored_config == {"cell_size_m": 250, "wind_weight": 0.6}

        stored_value = conn.execute(
            select(evaluation_result.c.value).where(evaluation_result.c.run_id == run_id)
        ).scalar_one()
        assert stored_value == pytest.approx(0.42)

        nonexistent_event_id = event_id + 1_000_000
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(model_run).values(
                    event_id=nonexistent_event_id,
                    model_name="orphan",
                    config={},
                )
            )
    finally:
        trans.rollback()
        conn.close()
