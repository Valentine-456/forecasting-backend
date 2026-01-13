import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from app.dtos.ForecastRequest import UAVState
from app.dtos.ForecastResponse import ForecastPoint
from app.services.engines.base_engine import BaseEngine
from app.services.telemetry_repository import TelemetryRepository
from app.services.feature_builder import FeatureBuilder


class MlrEngine(BaseEngine):
    """
    Predicts battery current for each forecast step,
    then converts it to SoC using Coulomb counting.
    """

    def __init__(
        self,
        model_path,
        telemetry_repo: TelemetryRepository,
        horizon_step_sec: int = 5,
    ):
        super().__init__(model_path, telemetry_repo, horizon_step_sec)

    def predict(self, uav: UAVState, horizon_seconds: int):
        capacity_mAh = float(uav.battery_capacity_mAh)
        soc_mAh = capacity_mAh * (uav.soc_percentage / 100.0)

        results: List[ForecastPoint] = []
        now = datetime.utcnow()

        # forecast loop: one ML prediction per horizon step
        dt = self.horizon_step_sec

        for t in range(dt, horizon_seconds + 1, dt):
            # Preprocessing data before inference 
            Xraw = self.repo.step(dt)
            X = FeatureBuilder.add_is_flying(Xraw)
            X = FeatureBuilder.insert_uav_state(X, uav, self.training_features)

            # Inference and integration
            pred_current_A = self._predict_current(X)
            mAh_used = (dt / 3600.0) * (pred_current_A * 1000.0)
            soc_mAh -= mAh_used
            if soc_mAh < 0:
                soc_mAh = 0.0

            soc_percent = (soc_mAh / capacity_mAh) * 100.0

            # append forecast point
            ts = now + timedelta(seconds=t)
            results.append(ForecastPoint(timestamp=ts, value=soc_percent))

        return results
