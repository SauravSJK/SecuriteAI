import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Tuple


def extract_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms timestamps into sine and cosine pairs.

    This preserves cyclical relationships (e.g., 23:59 is close to 00:01)
    across Hour, Minute, Second, Day, and Month dimensions.
    """
    time_specs = {
        "Hour": (df["Timestamp"].dt.hour, 24),
        "Minute": (df["Timestamp"].dt.minute, 60),
        "Second": (df["Timestamp"].dt.second, 60),
        "Day": (df["Timestamp"].dt.day, 31),
    }

    for name, (values, period) in time_specs.items():
        df[f"{name}_Sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_Cos"] = np.cos(2 * np.pi * values / period)
    return df


def normalize_event_id(
    df: pd.DataFrame,
    scaler_path: Optional[str] = None,
    params: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Applies Min-Max scaling to the Event_ID.

    If params are provided, it uses them (Inference). If not, it fits on the
    data and saves the parameters to the specified path (Training).
    """
    if params:
        min_v, max_v = params
    else:
        min_v, max_v = df["Event_ID"].min(), df["Event_ID"].max()
        if scaler_path:
            np.save(scaler_path, np.array([min_v, max_v]))

    df["Event_ID_Normalized"] = (df["Event_ID"] - min_v) / (max_v - min_v)
    return df


def feature_engineering_pipeline(
    df: pd.DataFrame,
    window_size: int = 20,
    scaler_path: Optional[str] = None,
    scaler_params: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Orchestrates the conversion of a DataFrame into a 3D windowed tensor.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = extract_cyclical_temporal_features(df)
    df = normalize_event_id(df, scaler_path=scaler_path, params=scaler_params)

    feature_cols = [
        "Hour_Sin",
        "Hour_Cos",
        "Minute_Sin",
        "Minute_Cos",
        "Second_Sin",
        "Second_Cos",
        "Day_Sin",
        "Day_Cos",
        "Event_ID_Normalized",
    ]

    features = df[feature_cols].values
    # Memory-efficient sliding window creation
    return sliding_window_view(features, (window_size, features.shape[1])).squeeze(
        axis=1
    )
