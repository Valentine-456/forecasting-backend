import pandas as pd

class FeatureBuilder:
    @staticmethod
    def add_is_flying(df):
        df = df.copy()
        if "speed_h_ms" in df.columns and "alt" in df.columns:
            df["is_flying"] = (
                (df["speed_h_ms"] > 0.1) | (df["alt"] > 1)
            ).astype(int)
        else:
            df["is_flying"] = 1

        return df
