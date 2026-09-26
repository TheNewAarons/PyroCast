"""Modelo de datos inicial en PostGIS, vía SQLAlchemy Core (sin ORM).

Tres tablas: fire_event (incendios reales usados para calibración y
backtesting), model_run (una ejecución de un modelo sobre un evento) y
evaluation_result (métricas de esa ejecución, por split train/val/test).
"""
from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
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
