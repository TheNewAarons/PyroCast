from fastapi import APIRouter, Request

from serving.api.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Probabilidad (acumulada) de propagación por día y por celda desde
    un punto de ignición. Ver docs/api.md."""
    service = request.app.state.prediction_service
    result: PredictResponse = service.predict(body)
    return result
