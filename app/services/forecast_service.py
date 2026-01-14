from pathlib import Path
import uuid
from datetime import datetime

from app.services.engines.xgb_ar_lag1_engine import XgbArEngine
from app.services.engines.xgb_engine import XgbEngine
from app.services.model_registry import ModelRegistry
from app.services.engines.mlr_engine import MlrEngine
from app.dtos.ForecastRequest import ForecastRequest
from app.dtos.ForecastResponse import ForecastResponse
from app.services.telemetry_repository import TelemetryRepository

class ForecastService:
    def __init__(self):
        self.registry = ModelRegistry()
        self.engines = {
            "mlr": MlrEngine,
            "xgb": XgbEngine,
        }
        test_csv = Path("data/test_dataset.csv")
        self.telemetry_repo = TelemetryRepository(test_csv)


    def run(self, req: ForecastRequest) -> ForecastResponse:
        model_info = self.registry.get(req.model_id)

        if model_info.engine == "mlr":
            engine = MlrEngine(
                model_info.path,
                telemetry_repo=self.telemetry_repo,
            )
        elif model_info.engine == "xgb":
            engine = XgbEngine(
                model_info.path,
                telemetry_repo=self.telemetry_repo,
            )
        elif model_info.engine == "xgb_ar_lag1":
            engine = XgbArEngine(
                model_info.path,
                telemetry_repo=self.telemetry_repo,
            )
        else:
            raise NotImplementedError(model_info.engine)

        points = engine.predict(req.uav_state, req.horizon_seconds)

        return ForecastResponse(
            model_id=req.model_id,
            generated_at=datetime.utcnow(),
            points=points,
        )

