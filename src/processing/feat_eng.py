import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer


def extract_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes time as Sin/Cos pairs. We use Hour, Minute, Second, and Day
    to provide context without overfitting to a specific month.
    Args:
        df: DataFrame with a 'Timestamp' column of datetime type.
    Returns:
        DataFrame with added cyclical features.
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
    Min-Max scaling for Event_ID. During training, we fit ONLY on Normal logs.
    During inference, we load the same scaler parameters to ensure consistency.
    Args:
        df: DataFrame with an 'Event_ID' column to normalize.
        scaler_path: Optional path to save/load scaler parameters (min, max).
        params: Optional tuple of (min, max) for scaling. If None, it will be computed from the data.
    Returns:
        DataFrame with an added 'Event_ID_Normalized' column.
    """
    if params:
        min_v, max_v = params
    else:
        min_v, max_v = df["Event_ID"].min(), df["Event_ID"].max()
        if scaler_path:
            np.save(scaler_path, np.array([min_v, max_v]))

    # Scale and handle clipping for inference
    df["Event_ID_Normalized"] = (df["Event_ID"] - min_v) / (max_v - min_v)
    return df


def event_embedding(
    df: pd.DataFrame,
    model: Optional[SentenceTransformer] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Converts the 'Content' column into dense vector embeddings using a pre-trained SentenceTransformer.
    Args:
        df: DataFrame with a 'Content' column containing log messages.
        model: Optional pre-loaded SentenceTransformer model. If None, it will be loaded based on model_name.
        model_name: Name of the pre-trained model to use for embedding.
    Returns:
        DataFrame with added embedding columns (e.g., 'Embed_0', 'Embed_1', ..., 'Embed_N').
    """
    if model is None:
        model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["Content"].tolist(), show_progress_bar=False, convert_to_numpy=True
    )

    # Add embedding dimensions as separate columns
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
    df = event_embedding(df)

    # 9 Primary features: 8 cyclical + 1 normalized Event ID
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
    return sliding_window_view(features, (window_size, features.shape[1])).squeeze(
        axis=1
    )
