"""API de PyroCast: /predict, /active-fires, /healthz.

Aviso obligatorio (ver CLAUDE.md): esta es una herramienta de
investigación, no un sistema operativo de combate de incendios.

`create_app` recibe sus dependencias (settings, cargador de modelo,
cliente FIRMS) para poder testearse sin red ni credenciales reales; la
instancia `app` de módulo usa los valores por defecto y resuelve la
configuración recién al iniciar (lifespan), no al importar.
"""
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from ingestion.firms.client import FirmsClient
from shared.config import Settings, get_settings

from serving.api.active_fires import (
    CACHE_TTL_SECONDS,
    ActiveFiresService,
    FirmsClientLike,
)
from serving.api.cache import LRUCache
from serving.api.errors import register_error_handlers
from serving.api.model_registry import LoadedModel, load_default_model
from serving.api.prediction import PredictionService
from serving.api.routers import fires, health, predict
from serving.api.schemas import ActiveFiresResponse, PredictResponse

PREDICTION_CACHE_SIZE = 256
ACTIVE_FIRES_CACHE_SIZE = 8


def create_app(
    settings: Settings | None = None,
    model_loader: Callable[[], LoadedModel] = load_default_model,
    firms_client: FirmsClientLike | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Falla rápido y con mensaje claro (ConfigurationError) si faltan
        # credenciales, en vez de arrancar "en verde" con una config inválida.
        resolved = settings if settings is not None else get_settings()
        loaded = model_loader()  # una sola vez, no por request
        app.state.prediction_service = PredictionService(
            resolved, loaded, LRUCache[PredictResponse](PREDICTION_CACHE_SIZE)
        )
        app.state.active_fires_service = ActiveFiresService(
            resolved,
            firms_client if firms_client is not None else FirmsClient(resolved.firms_map_key),
            LRUCache[ActiveFiresResponse](ACTIVE_FIRES_CACHE_SIZE, ttl_seconds=CACHE_TTL_SECONDS),
        )
        yield

    app = FastAPI(
        title="PyroCast API",
        description=(
            "Herramienta de investigación. No usar para decisiones operativas "
            "de combate de incendios sin validación de CONAF/SENAPRED."
        ),
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(fires.router)
    return app


app = create_app()
