"""Esquemas de request/response de la API.

Las respuestas de predicción y de detecciones son GeoJSON
`FeatureCollection` válidos con "foreign members" (permitidos por RFC
7946 §6.1): el aviso de herramienta de investigación, metadatos del
modelo, advertencias, etc. -- así Leaflet (`L.geoJSON`) puede consumir
la respuesta directamente.
"""
import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_HORIZON_DAYS = 7
DEFAULT_HORIZON_DAYS = 3
LIMITATIONS_DOC = "docs/limitations.md"
RESEARCH_DISCLAIMER = (
    "Herramienta de investigación. No usar para decisiones operativas de "
    "combate de incendios sin validación de CONAF/SENAPRED."
)


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitud del punto de ignición (WGS84)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitud (WGS84)")
    date: dt.date = Field(..., description="Fecha de la ignición (día 0, estado conocido)")
    horizon_days: int = Field(
        DEFAULT_HORIZON_DAYS, ge=1, le=MAX_HORIZON_DAYS, description="Días a predecir"
    )


class ResearchNotice(BaseModel):
    research_tool: Literal[True] = True
    limitations: str = LIMITATIONS_DOC
    disclaimer: str = RESEARCH_DISCLAIMER


class ModelInfo(BaseModel):
    name: str
    calibrated: bool


class DayInfo(BaseModel):
    day: int
    date: dt.date


class GridInfo(BaseModel):
    crs: str
    resolution_m: float
    rows: int
    cols: int
    bbox_wgs84: tuple[float, float, float, float]


class RequestEcho(BaseModel):
    lat: float
    lon: float
    snapped_lat: float
    snapped_lon: float
    date: dt.date
    horizon_days: int


class PredictResponse(ResearchNotice):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]
    model: ModelInfo
    request: RequestEcho
    grid: GridInfo
    days: list[DayInfo]
    warnings: list[str]
    cached: bool = False


class ActiveFiresResponse(ResearchNotice):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]
    source: str
    bbox: tuple[float, float, float, float]
    days: int
    cached: bool = False
