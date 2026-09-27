"""Modelo de datos inicial en PostGIS, vía SQLAlchemy Core (sin ORM).

Tres tablas: fire_event (incendios reales usados para calibración y
backtesting), model_run (una ejecución de un modelo sobre un evento) y
evaluation_result (métricas de esa ejecución, por split train/val/test).
"""
from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    func,
)

metadata = MetaData()

fire_event = Table(
    "fire_event",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("bbox", String, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=True),
    Column("source", String, nullable=False),
    Column("geom", Geometry(geometry_type="POLYGON", srid=4326), nullable=False),
    # enlaza esta fila con el event_id (hash de contenido derivado de
    # features/fire_state/clustering.py, potencialmente > 2^31 -- de ahí
    # BigInteger, no Integer) que features/dataset/ usa para nombrar el
    # Zarr y para las claves de splits.json. nullable=True: un evento
    # catalogado por CONAF/SENAPRED, no derivado de clustering de FIRMS,
    # no tiene este id. Encontrado en la revisión final del 2026-09-27:
    # sin esta columna, nada conectaba un archivo Zarr con su fila de
    # PostGIS.
    Column("firms_event_id", BigInteger, nullable=True, unique=True),
)

model_run = Table(
    "model_run",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey("fire_event.id"), nullable=False),
    Column("model_name", String, nullable=False),
    Column("config", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

evaluation_result = Table(
    "evaluation_result",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", Integer, ForeignKey("model_run.id"), nullable=False),
    Column("metric_name", String, nullable=False),
    Column("value", Float, nullable=False),
    Column("split", String, nullable=False),
)
