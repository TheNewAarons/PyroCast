"""Persiste metadatos de un evento de incendio en la tabla `fire_event`
de PostGIS (`shared.db.schema`) -- bbox recortado del evento, fechas, y
un `source` fijo identificando que viene del clustering de FIRMS (no de
un incendio catalogado por CONAF/SENAPRED, que usaría otro `source`).

Upsert por `firms_event_id` (no un `insert` liso): re-correr
`pyrocast-features build-dataset` para el mismo rango de fechas produce
los MISMOS `event_id` (hash de contenido, ver
`features/fire_state/clustering.py`) -- sin upsert, cada corrida
insertaría una fila nueva con el mismo `firms_event_id`, violando la
restricción UNIQUE o (peor, si no existiera) duplicando filas
indefinidamente. Encontrado en la revisión final del 2026-09-27."""
import datetime as dt

from shared.db.schema import fire_event
from sqlalchemy import Connection, Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert


def persist_fire_event_metadata(
    engine: Engine | Connection,
    event_id: int,
    bbox_cut: tuple[float, float, float, float],
    start_date: dt.date,
    end_date: dt.date,
    source: str = "firms_cluster",
) -> int:
    west, south, east, north = bbox_cut
    wkt = (
        f"SRID=4326;POLYGON(({west} {south}, {west} {north}, "
        f"{east} {north}, {east} {south}, {west} {south}))"
    )
    values = {
        "firms_event_id": event_id,
        "bbox": f"{west},{south},{east},{north}",
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
        "geom": wkt,
    }
    insert_stmt = pg_insert(fire_event).values(**values)
    stmt = insert_stmt.on_conflict_do_update(
        index_elements=[fire_event.c.firms_event_id],
        set_={
            "bbox": insert_stmt.excluded.bbox,
            "start_date": insert_stmt.excluded.start_date,
            "end_date": insert_stmt.excluded.end_date,
            "source": insert_stmt.excluded.source,
            "geom": insert_stmt.excluded.geom,
        },
    ).returning(fire_event.c.id)
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            row_id: int = conn.execute(stmt).scalar_one()
            return row_id
    row_id = engine.execute(stmt).scalar_one()
    return row_id
