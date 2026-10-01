"""Errores explícitos de la API: cada falla de datos o de una fuente
externa se reporta con un código estable y un mensaje legible -- nunca
se devuelve una predicción construida con datos faltantes ni un 500
genérico (CLAUDE.md: "nunca devolver un número inventado")."""
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ServingError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class OutsideStudyAreaError(ServingError):
    status_code = 422
    code = "outside_study_area"


class WeatherUnavailableError(ServingError):
    status_code = 422
    code = "weather_unavailable"


class TerrainCoverageError(ServingError):
    status_code = 422
    code = "terrain_coverage_unavailable"


class StaticLayersUnavailableError(ServingError):
    status_code = 503
    code = "static_layers_unavailable"


class FirmsUnavailableError(ServingError):
    status_code = 502
    code = "firms_unavailable"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServingError)
    async def _handle(_request: Request, exc: ServingError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )
