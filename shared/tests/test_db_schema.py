"""Verifica la forma de las tablas Core (sin conexión real a Postgres)."""
import os

import pytest
from sqlalchemy import create_engine, insert, select, text

from shared.config import Settings
from shared.db.schema import evaluation_result, fire_event, metadata, model_run


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
def test_can_insert_and_read_fire_event_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)
        conn.execute(
            insert(fire_event).values(
                bbox="-73.7,-39.3,-71.0,-36.5",
                start_date="2026-01-15",
                end_date="2026-01-20",
                source="test",
                geom="SRID=4326;POLYGON((-73 -39, -73 -38, -72 -38, -72 -39, -73 -39))",
            )
        )
        row = conn.execute(select(fire_event.c.source)).first()
    assert row is not None
    assert row.source == "test"
