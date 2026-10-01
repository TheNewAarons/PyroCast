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
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Nombre de la variable de entorno para cada campo de Settings que no
# tiene un valor por defecto real (todas las credenciales). pydantic
# reporta el nombre de campo en minúsculas en sus errores; esto traduce
# de vuelta al nombre de variable de entorno que el usuario realmente
# tiene que setear.
_ENV_VAR_BY_FIELD: dict[str, str] = {
    "firms_map_key": "FIRMS_MAP_KEY",
    "cds_api_url": "CDS_API_URL",
    "cds_api_key": "CDS_API_KEY",
    "copernicus_dataspace_client_id": "COPERNICUS_DATASPACE_CLIENT_ID",
    "copernicus_dataspace_client_secret": "COPERNICUS_DATASPACE_CLIENT_SECRET",
    "postgres_host": "POSTGRES_HOST",
    "postgres_port": "POSTGRES_PORT",
    "postgres_db": "POSTGRES_DB",
    "postgres_user": "POSTGRES_USER",
    "postgres_password": "POSTGRES_PASSWORD",
}


class ConfigurationError(RuntimeError):
    """Configuración inválida o incompleta, con un mensaje accionable
    (nombra las variables de entorno exactas que faltan o están vacías)."""

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

    # Seguridad de la API (serving/). Por defecto, "production": sin /docs ni
    # /openapi.json. `make serve` fija ENVIRONMENT=development. CORS: lista
    # EXPLÍCITA de orígenes (JSON en la variable, p. ej.
    # CORS_ALLOW_ORIGINS='["https://mi-app.example"]'); vacía = ninguno (la
    # interfaz web se sirve desde el mismo origen y no lo necesita).
    environment: Literal["development", "production"] = "production"
    cors_allow_origins: list[str] = Field(default_factory=list)

    @field_validator("cors_allow_origins")
    @classmethod
    def _no_wildcard_origin(cls, origins: list[str]) -> list[str]:
        if "*" in origins:
            raise ValueError("CORS_ALLOW_ORIGINS no admite '*': listar los orígenes explícitos")
        return origins

    # Rutas de datos (fuera de la imagen Docker, ver docker-compose.yml).
    data_raw_dir: Path = Path("data/raw")
    data_interim_dir: Path = Path("data/interim")
    data_processed_dir: Path = Path("data/processed")

    # NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/api/map_key/
    firms_map_key: str = Field(..., min_length=1, repr=False)

    # Copernicus CDS (ERA5-Land) — https://cds.climate.copernicus.eu/how-to-api
    cds_api_url: str = Field(..., min_length=1)
    cds_api_key: str = Field(..., min_length=1, repr=False)

    # Copernicus Data Space Ecosystem (Sentinel-2) — https://dataspace.copernicus.eu/
    copernicus_dataspace_client_id: str = Field(..., min_length=1, repr=False)
    copernicus_dataspace_client_secret: str = Field(..., min_length=1, repr=False)

    # PostgreSQL/PostGIS
    postgres_host: str = Field(..., min_length=1)
    postgres_port: int = Field(...)
    postgres_db: str = Field(..., min_length=1)
    postgres_user: str = Field(..., min_length=1)
    postgres_password: str = Field(..., min_length=1, repr=False)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        missing = sorted(
            {_ENV_VAR_BY_FIELD.get(str(err["loc"][0]), str(err["loc"][0])) for err in exc.errors()}
        )
        raise ConfigurationError(
            "Faltan variables de entorno requeridas (o están vacías): "
            + ", ".join(missing)
            + ". Copia .env.example a .env y complétalas — ese archivo indica "
            "dónde obtener cada credencial."
        ) from exc
