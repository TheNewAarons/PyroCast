"""Persiste metadatos de un evento de incendio en la tabla `fire_event`
de PostGIS (`shared.db.schema`) -- bbox recortado del evento, fechas, y
un `source` fijo identificando que viene del clustering de FIRMS (no de
un incendio catalogado por CONAF/SENAPRED, que usaría otro `source`)."""
import datetime as dt

from shared.db.schema import fire_event
from sqlalchemy import Connection, Engine, insert


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
        "bbox": f"{west},{south},{east},{north}",
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
        "geom": wkt,
    }
    stmt = insert(fire_event).values(**values).returning(fire_event.c.id)
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            row_id: int = conn.execute(stmt).scalar_one()
            return row_id
    row_id = engine.execute(stmt).scalar_one()
    return row_id
