from fastapi import APIRouter, Query, Request

from serving.api.schemas import ActiveFiresResponse

router = APIRouter()


@router.get("/active-fires", response_model=ActiveFiresResponse)
def active_fires(
    request: Request,
    days: int = Query(2, ge=1, le=5, description="Ventana en días (límite del Area API de FIRMS)"),
) -> ActiveFiresResponse:
    """Detecciones activas recientes de FIRMS (VIIRS NRT) dentro del bbox
    de trabajo. Ver docs/api.md."""
    result: ActiveFiresResponse = request.app.state.active_fires_service.get(days)
    return result
