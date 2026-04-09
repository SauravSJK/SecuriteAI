import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Generates 10,000 Normal logs and 1,000 Anomaly logs.
    Normal logs simulate a stable system state with predictable events.
    Anomaly logs simulate a high-entropy burst of unusual activity.
    """
    logs = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Normal Logs (stable system state)
    components = ["kernel", "systemd", "cron", "network-mgr"]
    templates = {
        "E01": "Heartbeat",
        "E02": "Mem check",
        "E03": "Task OK",
        "E04": "Net up",
    }

    current_time = start_time
    for _ in range(10000):
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

    # Anomaly Logs (High-entropy burst)
    current_time = start_time + timedelta(days=5)  # Start anomalies in the future
    for _ in range(1000):
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

    return pd.DataFrame(logs)
