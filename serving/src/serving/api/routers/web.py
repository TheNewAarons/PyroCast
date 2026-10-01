"""Página principal del mapa (Jinja2). El JS (`serving/web/static/app.js`)
lee la configuración de atributos `data-*` de `#app`, no de URLs
hardcodeadas."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from serving.api.schemas import (
    DEFAULT_HORIZON_DAYS,
    LIMITATIONS_DOC,
    MAX_HORIZON_DAYS,
    RESEARCH_DISCLAIMER,
)

router = APIRouter()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    from serving.api.main import WEB_DIR  # evita el import circular main <-> routers

    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    weather = request.app.state.prediction_service.weather_range()
    weather_min = weather[0].isoformat() if weather else ""
    weather_max = weather[1].isoformat() if weather else ""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "disclaimer": RESEARCH_DISCLAIMER,
            "limitations_doc": LIMITATIONS_DOC,
            "study_bbox": ",".join(str(v) for v in request.app.state.settings.study_area_bbox),
            "weather_min": weather_min,
            "weather_max": weather_max,
            "default_date": weather_min,
            "default_horizon": DEFAULT_HORIZON_DAYS,
            "max_horizon": MAX_HORIZON_DAYS,
        },
    )
