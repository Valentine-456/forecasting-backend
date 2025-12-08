import joblib
from datetime import datetime, timedelta
from typing import List

from app.dtos.ForecastRequest import UAVState
from app.dtos.ForecastResponse import ForecastPoint
from app.services.telemetry_repository import TelemetryRepository


class MlrEngine:
    def __init__(
        self,
        model_path,
        telemetry_repo: TelemetryRepository,
        horizon_step_sec: int = 5,
    ):
        self.model = joblib.load(model_path)
        self.repo = telemetry_repo
        self.horizon_step_sec = horizon_step_sec

    def predict(self, uav: UAVState, horizon_seconds: int) -> List[ForecastPoint]:

        # 1) Load a flight from the repository
        base_df = self.repo.get_random_test_flight()

        # 2) Prepare the feature matrix X
        X = self.repo.prepare_features(base_df, uav)

        # 3) Run the model
        preds = self.model.predict(X)

        last_value = float(preds[-1])

        # 4) Produce time-based forecast response
        now = datetime.utcnow()
        result = []

        for i in range(
            self.horizon_step_sec,
            horizon_seconds + 1,
            self.horizon_step_sec,
        ):
            ts = now + timedelta(seconds=i)
            result.append(ForecastPoint(timestamp=ts, value=last_value))

        return result
