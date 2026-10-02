from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from serving.api.schemas import LIMITATIONS_DOC, RESEARCH_DISCLAIMER

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: el proceso responde. No toca datos ni modelo."""
    return {"status": "ok"}


@router.get("/api/health/")
def api_health(request: Request) -> JSONResponse:
    """Readiness: hay capas estáticas, clima procesado y modelo cargado (lo que
    /predict necesita). 503 con el motivo si falta algo. La app no usa base de
    datos, así que no hay conexión que verificar (ver DEPLOY.md)."""
    service = request.app.state.prediction_service
    static_problem = service.static_layers_problem()
    weather = service.weather_range()
    checks: dict[str, Any] = {
        "static_layers": static_problem or "ok",
        "weather": (
            {"first_day": weather[0].isoformat(), "last_day": weather[1].isoformat()}
            if weather else None
        ),
        "model": service.model_name,
    }
    healthy = static_problem is None and weather is not None
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded", "checks": checks,
            "research_tool": True, "limitations": LIMITATIONS_DOC,
            "disclaimer": RESEARCH_DISCLAIMER,
        },
    )
