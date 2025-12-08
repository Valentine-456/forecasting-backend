import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import List

from app.dtos.ForecastRequest import UAVState
from app.dtos.ForecastResponse import ForecastPoint

class MlrEngine:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, uav: UAVState, horizon_seconds: int) -> List[ForecastPoint]:
        # Build features vector; must match training script!
        X = np.array([[uav.wind_speed, uav.payload_kg, uav.uav_mass]])

        y_pred = float(self.model.predict(X)[0])

        now = datetime.utcnow()
        result = []

        for i in range(1, horizon_seconds + 1, 5):  # 1 per 5 seconds
            ts = now + timedelta(seconds=i)
            result.append(ForecastPoint(timestamp=ts, value=y_pred))

        return result
