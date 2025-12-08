from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class UAVState(BaseModel):
    soc: float
    battery_current: float
    wind_speed: float
    payload_kg: float
    uav_mass: float

class ForecastRequest(BaseModel):
    uav_state: UAVState
    model_id: str
    horizon_seconds: int = Field(..., ge=10, le=3600)
