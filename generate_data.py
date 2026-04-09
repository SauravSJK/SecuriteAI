import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Generates a large-scale synthetic dataset of Linux logs.

    Normal Pattern: Consistent heartbeat events from kernel/system components.
    Anomaly Pattern: High-entropy bursts from unauthorized components with rare IDs.
    Args:
        None
    Returns:
        pd.DataFrame: The generated dataset.
    """
    print("[*] Generating 11,000 synthetic log entries...")

    logs = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # --- 1. Generate 10,000 NORMAL logs ---
    # These represent stable system state
    components = ["kernel", "systemd", "cron", "network-mgr"]
    normal_templates = {
        "E01": "System heartbeat check successful",
        "E02": "Kernel memory allocation: OK",
        "E03": "Scheduled task execution: Completed",
        "E04": "Network interface eth0: Link up",
    }

    current_time = start_time
    for i in range(10000):
        # Normal logs happen every 5-15 seconds
        current_time += timedelta(seconds=random.randint(5, 15))

        comp = random.choice(components)
        event_id = random.choice(list(normal_templates.keys()))

        logs.append(
            {
                "Year": current_time.year,
                "Month": current_time.strftime("%b"),
                "Date": current_time.day,
                "Time": current_time.strftime("%H:%M:%S"),
                "Level": "Info",
                "Component": comp,
                "PID": random.randint(1, 2000),
                "Content": normal_templates[event_id],
                "EventId": event_id,
                "EventTemplate": normal_templates[event_id],
            }
        )

    # --- 2. Generate 1,000 ANOMALY logs ---
    # These represent a brute-force attack or system failure
    anomaly_templates = {
        "E999": "CRITICAL: Unauthorized access attempt",
        "E888": "FAILED: Authentication failure for user root",
        "E777": "ALARM: Kernel panic - unexpected state",
    }

    current_time = start_time
    for i in range(1000):
        # Anomalies happen in a burst (very rapid succession)
        current_time += timedelta(milliseconds=random.randint(10, 500))

        event_id = random.choice(list(anomaly_templates.keys()))

        logs.append(
            {
                "Year": current_time.year,
                "Month": current_time.strftime("%b"),
                "Date": current_time.day,
                "Time": current_time.strftime("%H:%M:%S"),
                "Level": "Alert",
                "Component": "auth-service",
                "PID": 9999,
                "Content": anomaly_templates[event_id],
                "EventId": event_id,
                "EventTemplate": anomaly_templates[event_id],
            }
        )

    df = pd.DataFrame(logs)
    print(f"[SUCCESS] Synthetic data generated with {len(df)} entries.")
    print("Normal Entries: 10,000 | Anomaly Entries: 1,000")
    return df
