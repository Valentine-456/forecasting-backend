from fastapi import APIRouter, Depends, HTTPException, Response
from app.dtos.ForecastRequest import ForecastRequest
from app.services.forecast_service import ForecastService

router = APIRouter(prefix="/forecast", tags=["forecast"])

def get_service():
    return ForecastService()

@router.get("/models")
def get_models(service: ForecastService = Depends(get_service)):
    return [
        {
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "description": m.description,
        }
        for m in service.registry.list()
    ]

@router.post("/run")
def run(req: ForecastRequest, service: ForecastService = Depends(get_service)):
    try:
        return service.run(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))