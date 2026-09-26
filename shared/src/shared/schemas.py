"""Esquemas normalizados compartidos entre módulos de ingesta y features.

FireDetection es el resultado de normalizar la respuesta cruda de
cualquier fuente de detecciones activas de fuego (hoy: NASA FIRMS) a un
esquema común, independiente del formato CSV/columnas propias de cada
sensor.
"""
from datetime import datetime

from pydantic import BaseModel, Field


class FireDetection(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    detected_at: datetime
    frp: float | None = None
    # Se mantiene como texto tal cual lo entrega la fuente: VIIRS usa
    # categorías ("l"/"n"/"h" o "low"/"nominal"/"high" según versión del
    # producto), MODIS usa un porcentaje numérico como string. Forzar un
    # tipo único perdería información.
    confidence: str
    satellite: str
    instrument: str
