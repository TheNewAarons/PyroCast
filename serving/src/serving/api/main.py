"""API y mapa web de PyroCast: /predict, /active-fires, /healthz, / (mapa).

Aviso obligatorio (ver CLAUDE.md): esta es una herramienta de
investigación, no un sistema operativo de combate de incendios.

`create_app` recibe sus dependencias (settings, cargador de modelo,
cliente FIRMS) para poder testearse sin red ni credenciales reales; la
instancia `app` de módulo usa los valores por defecto y resuelve la
configuración recién al iniciar (lifespan), no al importar.
"""
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ingestion.firms.client import FirmsClient
from shared.config import ConfigurationError, Settings, get_settings

from serving.api.active_fires import (
    CACHE_TTL_SECONDS,
    ActiveFiresService,
    FirmsClientLike,
)
from serving.api.cache import LRUCache
from serving.api.errors import register_error_handlers
from serving.api.model_registry import LoadedModel, load_default_model
from serving.api.prediction import PredictionService
from serving.api.routers import fires, health, predict, web
from serving.api.schemas import ActiveFiresResponse, PredictResponse

# serving/web/ (plantillas Jinja2 + estáticos) vive junto al paquete, no
# dentro de él: serving/src/serving/api/main.py -> parents[3] == serving/
WEB_DIR = Path(__file__).resolve().parents[3] / "web"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # origen (no la ruta) hacia otros sitios: los tiles de OpenStreetMap
    # exigen un Referer según su política de uso.
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}

PREDICTION_CACHE_SIZE = 256
ACTIVE_FIRES_CACHE_SIZE = 8


def _early_settings(settings: Settings | None) -> Settings | None:
    # Los middlewares y las URLs de /docs se fijan al CREAR la app, antes del
    # lifespan. Si la configuración falta, se usan los valores seguros por
    # defecto (sin docs, sin CORS) y el lifespan falla con ConfigurationError.
    if settings is not None:
        return settings
    try:
        return get_settings()
    except ConfigurationError:
        return None


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
        app.state.settings = resolved
        app.state.prediction_service = PredictionService(
            resolved, loaded, LRUCache[PredictResponse](PREDICTION_CACHE_SIZE)
        )
        app.state.active_fires_service = ActiveFiresService(
            resolved,
            firms_client if firms_client is not None else FirmsClient(resolved.firms_map_key),
            LRUCache[ActiveFiresResponse](ACTIVE_FIRES_CACHE_SIZE, ttl_seconds=CACHE_TTL_SECONDS),
        )
        yield

    early = _early_settings(settings)
    docs_enabled = early is not None and early.environment == "development"
    origins = list(early.cors_allow_origins) if early is not None else []
    if "*" in origins:
        raise ValueError("CORS_ALLOW_ORIGINS no admite '*': listar los orígenes explícitos")

    app = FastAPI(
        debug=False,  # nunca: DEBUG expone trazas y el depurador
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        title="PyroCast API",
        description=(
            "Herramienta de investigación. No usar para decisiones operativas "
            "de combate de incendios sin validación de CONAF/SENAPRED."
        ),
        lifespan=lifespan,
    )
    register_error_handlers(app)
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"], allow_credentials=False,
        )

    @app.middleware("http")
    async def _security_headers(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(fires.router)
    app.include_router(web.router)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return app


app = create_app()
