import pandas as pd
from datetime import datetime


def clean_linux_logs(input_file: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and structures Linux log data for sequence-based anomaly detection.

    This function processes the LogHub Linux dataset by synthesizing timestamps
    from discrete columns, extracting numerical Event IDs via regex, and
    ensuring strict chronological order for the LSTM.

    Args:
        input_file: DataFrame with a structured log CSV file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        print(
            "[*] Loading raw log data from DataFrame and performing cleaning operations..."
        )
        df = input_file

        def parse_timestamp(row: pd.Series) -> datetime:
            """Combines log date/time columns into a unified datetime object."""
            date_str = f"{row['Year']} {row['Month']} {row['Date']} {row['Time']}"
            return pd.to_datetime(date_str, format="%Y %b %d %H:%M:%S")

        # Feature Extraction and Synthesis
        df["Timestamp"] = df.apply(parse_timestamp, axis=1)
        df["Event_ID"] = df["EventId"].str.extract(r"(\d+)").astype(int)

        # Retain necessary features and sort for temporal sequence integrity
        cleaned_df = df[["Timestamp", "Component", "Event_ID"]]
        cleaned_df = cleaned_df.sort_values(by="Timestamp").reset_index(drop=True)

        return cleaned_df

    except Exception as e:
        print(f"[ERROR] Log cleaning failed: {e}")
        raise
