import uuid
from datetime import datetime

from app.services.model_registry import ModelRegistry
from app.services.engines.mlr_engine import MlrEngine
from app.dtos.ForecastRequest import ForecastRequest
from app.dtos.ForecastResponse import ForecastResponse

class ForecastService:
    def __init__(self):
        self.registry = ModelRegistry()
        self.engines = {
            "mlr": MlrEngine
        }

    def run(self, req: ForecastRequest) -> ForecastResponse:
        info = self.registry.get(req.model_id)
        EngineClass = self.engines[info.engine]

        engine = EngineClass(info.path)
        points = engine.predict(req.uav_state, req.horizon_seconds)

        response = ForecastResponse(
            forecast_id=str(uuid.uuid4()),
            model_id=req.model_id,
            generated_at=datetime.utcnow(),
            points=points,
        )

        self.cache[response.forecast_id] = response
        return response
