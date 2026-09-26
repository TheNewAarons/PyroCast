"""API de PyroCast: por ahora solo expone /healthz.

Aviso obligatorio (ver CLAUDE.md): esta es una herramienta de
investigación, no un sistema operativo de combate de incendios.
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from shared.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Falla rápido y con mensaje claro (ConfigurationError) si faltan
    # credenciales, en vez de arrancar "en verde" con una config inválida.
    get_settings()
    yield


app = FastAPI(
    title="PyroCast API",
    description=(
        "Herramienta de investigación. No usar para decisiones operativas "
        "de combate de incendios sin validación de CONAF/SENAPRED."
    ),
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
