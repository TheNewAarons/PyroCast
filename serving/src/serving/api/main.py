"""API de PyroCast: por ahora solo expone /healthz.

Aviso obligatorio (ver CLAUDE.md): esta es una herramienta de
investigación, no un sistema operativo de combate de incendios.
"""
from fastapi import FastAPI

app = FastAPI(
    title="PyroCast API",
    description=(
        "Herramienta de investigación. No usar para decisiones operativas "
        "de combate de incendios sin validación de CONAF/SENAPRED."
    ),
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
