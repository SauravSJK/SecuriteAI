import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def extract_cyclical_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclical temporal features from the Timestamp column for enhanced sequence modeling.
    """
    # Hour (0-23)
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)

    # Minute (0-59)
    df["Minute_Sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.minute / 60)
    df["Minute_Cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.minute / 60)

    # Second (0-59)
    df["Second_Sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.second / 60)
    df["Second_Cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.second / 60)

    # Day of Week (0-6)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.dayofweek / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.dayofweek / 7)

    # Month (1-12)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)

    return df


def normalize_event_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the Event_ID feature to a 0-1 range using Min-Max scaling.
    """
    min_val = df["Event_ID"].min()
    max_val = df["Event_ID"].max()
    df["Event_ID_Normalized"] = (df["Event_ID"] - min_val) / (max_val - min_val)
    return df


def create_sliding_windows(df: pd.DataFrame, window_size: int = 20) -> np.ndarray:
    """
    Creates overlapping sliding windows using high-performance memory views.

    Args:
        df (pd.DataFrame): DataFrame containing the log data.
        window_size (int): The number of consecutive log entries per window.

    Returns:
        np.ndarray: A 3D array of shape (num_windows, window_size, num_features).
    """
    # Updated feature list to include the high-frequency temporal components
    feature_cols = [
        "Hour_Sin",
        "Hour_Cos",
        "Minute_Sin",
        "Minute_Cos",
        "Second_Sin",
        "Second_Cos",
        "DayOfWeek_Sin",
        "DayOfWeek_Cos",
        "Month_Sin",
        "Month_Cos",
        "Event_ID_Normalized",
    ]
    features = df[feature_cols].values

    # Efficient windowing using stride tricks
    # Input shape: (N, 11) -> Output shape: (N-window_size+1, window_size, 11)
    windows = sliding_window_view(features, (window_size, features.shape[1])).squeeze(
        axis=1
    )

    return windows


def feature_engineering_pipeline(file_path: str, output_path: str, window_size: int = 20) -> None:
    """
    Executes the full feature engineering pipeline from raw logs to windowed sequences.

    Args:
        file_path (str): Path to the cleaned log CSV file.
        output_path (str): Path where the processed sequences will be saved.
        window_size (int): The number of log entries per sequence window.

    Returns:
        None
    """
    try:
        # 1. Loading
        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # 2. Engineering
        df = extract_cyclical_temporal_features(df)
        df = normalize_event_id(df)

        # 3. Windowing
        windows = create_sliding_windows(df, window_size)

        # 4. Serialization
        np.save(output_path, windows)

        print("[SUCCESS] Feature Engineering Complete with sub-minute granularity.")
        print(f"[*] Processed shape: {windows.shape}")
        print(f"[*] Sequences saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process logs: {e}")


if __name__ == "__main__":
    input_path = "data/linux_logs_cleaned.csv"
    output_path = "data/data_sequences.npy"
    window_size = 20

    feature_engineering_pipeline(input_path, output_path, window_size)
