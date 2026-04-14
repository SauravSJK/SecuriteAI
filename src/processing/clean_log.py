import pandas as pd
from datetime import datetime


def clean_linux_logs(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw log DataFrames and enforces chronological sorting.
    Args:
        input_df (pd.DataFrame): The raw log DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        df = input_df.copy()

        def parse_timestamp(row: pd.Series) -> datetime:
            """Synthesizes a datetime object from year/month/date/time columns."""
            date_str = f"{row['Year']} {row['Month']} {row['Date']} {row['Time']}"
            return pd.to_datetime(date_str, format="%Y %b %d %H:%M:%S")

        df["Timestamp"] = df.apply(parse_timestamp, axis=1)
        df["Event_ID"] = df["EventId"].str.extract(r"(\d+)").astype(int)

        # Select core features and sort to maintain temporal integrity
        cleaned_df = df[["Timestamp", "Component", "Event_ID"]]
        cleaned_df = cleaned_df.sort_values(by="Timestamp").reset_index(drop=True)

        return cleaned_df

    except Exception as e:
        print(f"[ERROR] Cleaning failed: {e}")
        raise
