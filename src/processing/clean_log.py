"""
SecuriteAI Log Cleaning Service
-------------------------------
Description: Standardizes raw log inputs and enforces chronological sorting
to maintain temporal integrity in the sliding windows.
"""

import pandas as pd
from datetime import datetime


def clean_linux_logs(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw log DataFrames and synthesizes unified timestamps.

    Args:
        input_df: Raw log data from the /ingest endpoint.
    Returns:
        Cleaned DataFrame sorted by system time.
    """
    try:
        df = input_df.copy()

        def parse_timestamp(row: pd.Series) -> datetime:
            # Combine discrete columns into a standard datetime object
            date_str = f"{row['Year']} {row['Month']} {row['Date']} {row['Time']}"
            return pd.to_datetime(date_str, format="%Y %b %d %H:%M:%S")

        df["Timestamp"] = df.apply(parse_timestamp, axis=1)
        # Extract numerical digits from the EventID for normalization
        df["Event_ID"] = df["EventId"].str.extract(r"(\d+)").astype(int)

        # Enforce chronological ordering to prevent window pollution
        cleaned_df = df[["Timestamp", "Component", "Event_ID", "Content"]]
        cleaned_df = cleaned_df.sort_values(by="Timestamp").reset_index(drop=True)

        return cleaned_df

    except Exception as e:
        print(f"[ERROR] Cleaning failed: {e}")
        raise
