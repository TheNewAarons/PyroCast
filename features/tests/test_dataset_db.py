"""Tests de persistencia de metadatos de evento en `fire_event`
(PostGIS). El insert real solo se ejercita contra un Postgres real
(guardado con skipif, mismo patrón que shared/tests/test_db_schema.py) --
lo demás se verifica sin red/DB."""
import datetime as dt
import os

import pytest
from features.dataset.db import persist_fire_event_metadata
from shared.config import Settings
from shared.db.schema import fire_event, metadata
from sqlalchemy import create_engine, select, text


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
def test_persist_fire_event_metadata_roundtrip_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)

        event_id = persist_fire_event_metadata(
            engine=conn,
            event_id=1,
            bbox_cut=(-72.9, -38.9, -72.1, -38.1),
            start_date=dt.date(2026, 1, 15),
            end_date=dt.date(2026, 1, 20),
        )
        stored_bbox = conn.execute(
            select(fire_event.c.bbox).where(fire_event.c.id == event_id)
        ).scalar_one()
        assert stored_bbox == "-72.9,-38.9,-72.1,-38.1"
        stored_firms_event_id = conn.execute(
            select(fire_event.c.firms_event_id).where(fire_event.c.id == event_id)
        ).scalar_one()
        assert stored_firms_event_id == 1

        # re-correr con el MISMO firms_event_id pero un bbox distinto
        # debe ACTUALIZAR la fila existente (upsert), no insertar una
        # segunda -- simula re-correr build-dataset para el mismo evento
        # (encontrado en la revisión final del 2026-09-27).
        same_row_id = persist_fire_event_metadata(
            engine=conn,
            event_id=1,
            bbox_cut=(-73.0, -39.0, -72.0, -38.0),
            start_date=dt.date(2026, 1, 15),
            end_date=dt.date(2026, 1, 21),
        )
        assert same_row_id == event_id
        updated_bbox = conn.execute(
            select(fire_event.c.bbox).where(fire_event.c.id == event_id)
        ).scalar_one()
        assert updated_bbox == "-73.0,-39.0,-72.0,-38.0"
        matching_rows = conn.execute(
            select(fire_event.c.id).where(fire_event.c.firms_event_id == 1)
        ).all()
        assert len(matching_rows) == 1
    finally:
        trans.rollback()
        conn.close()
