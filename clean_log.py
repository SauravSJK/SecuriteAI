import pandas as pd
from datetime import datetime


def clean_linux_logs(file_path: str, output_path: str) -> None:
    """
    Cleans and structures Linux log data for sequence-based anomaly detection.

    This function processes the LogHub Linux dataset by:
    1. Parsing Month, Date, and Time into a unified Timestamp.
    2. Extracting numerical Event IDs from categorical strings.
    3. Sorting records chronologically to preserve temporal patterns.
    4. Filtering for features essential to LSTM-Autoencoder modeling.

    Args:
        file_path (str): Path to the structured log CSV file.
        output_path (str): Path where the cleaned CSV will be saved.

    Returns:
        None
    """
    try:
        print(f"Loading log data from: {file_path}")
        df = pd.read_csv(file_path)

        def parse_timestamp(row: pd.Series) -> datetime:
            """
            Synthesizes a full datetime object from log columns.
            Assumes 2024 as a dummy year for sequential consistency.
            """
            # Format: '2024 Jun 14 15:16:01'
            date_str = f"2024 {row['Month']} {row['Date']} {row['Time']}"
            return pd.to_datetime(date_str, format="%Y %b %d %H:%M:%S")

        print("Transforming temporal and categorical features...")

        # Create unified timestamp for windowing
        df["Timestamp"] = df.apply(parse_timestamp, axis=1)

        # Extract numeric ID from EventId string (e.g., 'E16' -> 16)
        # This allows the model to handle event types as numerical categories
        df["Event_ID"] = df["EventId"].str.extract(r"(\d+)").astype(int)

        # Select features required for the reconstruction-based anomaly detection pipeline
        # Component provides system context, Event_ID provides the action pattern
        cleaned_df = df[["Timestamp", "Component", "Event_ID"]]

        # Ensure the sequence is strictly chronological
        cleaned_df = cleaned_df.sort_values(by="Timestamp").reset_index(drop=True)

        print(f"Saving processed data to: {output_path}")
        cleaned_df.to_csv(output_path, index=False)

        print("\nData preparation successful.")
        print("Sample of cleaned data:")
        print("-" * 30)
        print(cleaned_df.head())
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in input file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Standard file definitions for the SecuriteAI preprocessing stage
    RAW_LOGS = "data/Linux_2k.log_structured.csv"
    CLEANED_LOGS = "data/linux_logs_cleaned.csv"

    clean_linux_logs(RAW_LOGS, CLEANED_LOGS)
