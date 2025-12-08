from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional

from app.dtos.ForecastRequest import UAVState


class TelemetryRepository:
    """
    Loads a prepared test dataset ONCE and provides slices of telemetry
    ready for ML inference.

    The ML engines do NOT know about CSV files.
    """

    def __init__(self, csv_path: Path, features: List[str]):
        self.features = features
        if not csv_path.exists():
            raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df.sort_values(["flight", "time"]).reset_index(drop=True)
        self.df = df
        self.test_flights = df["flight"].unique()

    def get_random_test_flight(self) -> pd.DataFrame:
        """Selects one random flight from test split."""
        flight = np.random.choice(self.test_flights)
        return self.df[self.df["flight"] == flight].reset_index(drop=True)

    def get_flight_by_id(self, flight_id: int) -> pd.DataFrame:
        """Retrieves a specific flight (if frontend wants to choose)."""
        df = self.df[self.df["flight"] == flight_id]
        if df.empty:
            raise ValueError(f"Flight {flight_id} not found in telemetry dataset.")
        return df.reset_index(drop=True)

    def prepare_features(
        self,
        base_df: pd.DataFrame,
        uav: UAVState,
        replace_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Takes a dataframe slice and replaces selected feature
        values with values from UAVState.
        """

        df = base_df.copy()

        # Replace relevant fields with request data
        # (This matches your training features!)
        df["wind_speed"] = uav.wind_speed
        df["payload_kg"] = uav.payload_kg
        df["uav_mass"] = uav.uav_mass

        # Additional replacements (future extension)
        if replace_map:
            for col, value in replace_map.items():
                df[col] = value

        # Ensure only the trained features are returned
        return df[self.features].fillna(0.0)
