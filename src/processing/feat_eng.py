"""
SecuriteAI Feature Engineering Pipeline
---------------------------------------
Description: Converts system logs into a windowed 3D tensor. Implements
cyclical time encoding to solve the 'Midnight Cliff' problem.
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer


def extract_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes time as Sin/Cos pairs to maintain adjacent temporal distance.
    """
    time_specs = {
        "Hour": (df["Timestamp"].dt.hour, 24),
        "Minute": (df["Timestamp"].dt.minute, 60),
        "Second": (df["Timestamp"].dt.second, 60),
        "Day": (df["Timestamp"].dt.day, 31),
    }

    for name, (values, period) in time_specs.items():
        # Sine/Cosine decomposition
        df[f"{name}_Sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_Cos"] = np.cos(2 * np.pi * values / period)
    return df


def normalize_event_id(
    df: pd.DataFrame,
    scaler_path: Optional[str] = None,
    params: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Min-Max scaling for Event_ID. Implements Isolation Normalization.
    """
    if params:
        min_v, max_v = params
    else:
        min_v, max_v = df["Event_ID"].min(), df["Event_ID"].max()
        if scaler_path:
            np.save(scaler_path, np.array([min_v, max_v]))

    # Scaled values ensure features are within a [0, 1] range for LSTM
    df["Event_ID_Normalized"] = (df["Event_ID"] - min_v) / (max_v - min_v)
    return df


def event_embedding(
    df: pd.DataFrame,
    model: Optional[SentenceTransformer] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Converts log content into dense semantic vectors using a transformer.
    """
    if model is None:
        model = SentenceTransformer(model_name)
    # 384-dimensional dense representation of log semantics
    embeddings = model.encode(
        df["Content"].tolist(), show_progress_bar=False, convert_to_numpy=True
    )

    embed_cols = [f"Embed_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols, index=df.index)
    return pd.concat([df, embed_df], axis=1)


def feature_engineering_pipeline(
    df: pd.DataFrame,
    window_size: int = 20,
    scaler_path: Optional[str] = None,
    scaler_params: Optional[Tuple[float, float]] = None,
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """
    Orchestrates the conversion of a DataFrame into a windowed 3D tensor.
    """
    df = extract_cyclical_temporal_features(df)
    df = normalize_event_id(df, scaler_path=scaler_path, params=scaler_params)
    df = event_embedding(df, model=model)

    # 9 Core features + Semantic Embeddings
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

    for col in df.columns:
        if col.startswith("Embed_"):
            feature_cols.append(col)

    features = df[feature_cols].values
    # Sliding window view converts the series into a context-aware tensor
    return sliding_window_view(features, (window_size, features.shape[1])).squeeze(
        axis=1
    )
