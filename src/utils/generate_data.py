import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Generates 10,000 Normal logs and 1,000 Anomaly logs across a full year.
    By spreading logs across different months, days, and hours, the model
    can better learn cyclical patterns and generalize to new data.
    """
    logs = []

    # Configuration for Normal behavior
    components = ["kernel", "systemd", "cron", "network-mgr"]
    templates = {
        "E01": "Heartbeat",
        "E02": "Mem check",
        "E03": "Task OK",
        "E04": "Net up",
    }

    # 1. Generate Normal Logs in varied clusters throughout the year 2024
    total_normal = 10000
    logs_per_cluster = 500  # Smaller chunks to ensure temporal variety
    num_clusters = total_normal // logs_per_cluster

    for _ in range(num_clusters):
        # Pick a random starting point in the year
        random_days = random.randint(0, 364)
        random_seconds = random.randint(0, 86399)
        cluster_start = datetime(2024, 1, 1) + timedelta(
            days=random_days, seconds=random_seconds
        )

        current_time = cluster_start
        for _ in range(logs_per_cluster):
            # Normal logs occur every 5-15 seconds
            current_time += timedelta(seconds=random.randint(5, 15))
            eid = random.choice(list(templates.keys()))
            logs.append(
                {
                    "Year": current_time.year,
                    "Month": current_time.strftime("%b"),
                    "Date": current_time.day,
                    "Time": current_time.strftime("%H:%M:%S"),
                    "Component": random.choice(components),
                    "EventId": eid,
                }
            )

    # 2. Generate Anomaly Logs (High-entropy bursts) at random dates
    total_anomalies = 1000
    anomalies_per_burst = 100
    num_bursts = total_anomalies // anomalies_per_burst

    for _ in range(num_bursts):
        # Randomize when the attack happens
        random_days = random.randint(0, 364)
        burst_start = datetime(2024, 1, 1) + timedelta(days=random_days)

        current_time = burst_start
        for _ in range(anomalies_per_burst):
            # Anomalies are high-frequency (milliseconds)
            current_time += timedelta(milliseconds=random.randint(10, 500))
            logs.append(
                {
                    "Year": current_time.year,
                    "Month": current_time.strftime("%b"),
                    "Date": current_time.day,
                    "Time": current_time.strftime("%H:%M:%S"),
                    "Component": "auth-service",
                    "EventId": "E999",
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(logs)

    # Chronological sorting is handled by clean_log.py during the pipeline
    print(
        f"[SUCCESS] Generated {len(logs)} logs distributed across various dates/times."
    )
    return df
