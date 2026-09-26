"""Configuración central de PyroCast.

Toda credencial se lee de variables de entorno, sin valores por defecto
reales: si falta una variable requerida, pydantic-settings falla con un
ValidationError legible (no un KeyError críptico) en cuanto se instancia
Settings(). El área de estudio por defecto cubre Biobío, Ñuble y La
Araucanía (las regiones más afectadas en la temporada 2025-2026); las
coordenadas son un bounding box aproximado en WGS84 (lon/lat), fuente:
límites administrativos de las regiones VIII, XVI y IX de Chile (BCN /
INE, simplificados a un rectángulo envolvente para uso interno del
proyecto — no es un límite administrativo exacto).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# (min_lon, min_lat, max_lon, max_lat) en WGS84 — envolvente de
# Biobío + Ñuble + La Araucanía.
DEFAULT_STUDY_AREA_BBOX: tuple[float, float, float, float] = (
    -73.7,
    -39.3,
    -71.0,
    -36.5,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Área de estudio y grilla — configurable, nunca hardcodeado por módulo.
    study_area_bbox: tuple[float, float, float, float] = DEFAULT_STUDY_AREA_BBOX
    crs: str = "EPSG:32719"
    spatial_resolution_m: int = 250
    temporal_resolution: str = "daily"

    # Rutas de datos (fuera de la imagen Docker, ver docker-compose.yml).
    data_raw_dir: Path = Path("data/raw")
    data_interim_dir: Path = Path("data/interim")
    data_processed_dir: Path = Path("data/processed")

    # NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/api/map_key/
    firms_map_key: str = Field(...)

    # Copernicus CDS (ERA5-Land) — https://cds.climate.copernicus.eu/how-to-api
    cds_api_url: str = Field(...)
    cds_api_key: str = Field(...)

    # Copernicus Data Space Ecosystem (Sentinel-2) — https://dataspace.copernicus.eu/
    copernicus_dataspace_client_id: str = Field(...)
    copernicus_dataspace_client_secret: str = Field(...)

    # PostgreSQL/PostGIS
    postgres_host: str = Field(...)
    postgres_port: int = Field(...)
    postgres_db: str = Field(...)
    postgres_user: str = Field(...)
    postgres_password: str = Field(...)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
