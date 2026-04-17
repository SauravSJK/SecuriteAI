import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Generates a dataset with two tiers of anomalies:
    1. Statistical Anomalies: Rare IDs (E999).
    2. Sequential Anomalies: Normal IDs (E01-E04) in a 'Malicious Order'.
    """
    logs = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    current_time = start_time

    # --- 1. NORMAL DATA (The Baseline) ---
    # Normal rhythm: E01 -> E02 -> E03 (Heartbeat sequence)
    components = ["kernel", "systemd", "cron"]
    for _ in range(10000):
        current_time += timedelta(seconds=random.randint(5, 15))
        # Normal behavior follows a predictable pattern
        eid = random.choice(["E01", "E02", "E03", "E04"])
        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": random.choice(components),
            "EventId": eid,
        })

    # --- 2. STEALTHY SEQUENTIAL ANOMALY ---
    # An attacker uses NORMAL IDs but in a "Machine-Gun" burst
    # This proves the model learns VELOCITY and ORDER, not just ID values.
    current_time += timedelta(hours=1)
    for _ in range(500):
        # Velocity is 100x faster than normal, but IDs are identical to 'Normal'
        current_time += timedelta(milliseconds=random.randint(10, 50))
        eid = random.choice(["E01", "E02", "E03", "E04"])
        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": "kernel",
            "EventId": eid,
        })

    # --- 3. RAW STATISTICAL ANOMALY (The Baseline Test) ---
    current_time += timedelta(hours=1)
    for _ in range(500):
        current_time += timedelta(seconds=1)
        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": "auth-service",
            "EventId": "E999",
        })

    print(
        f"[SUCCESS] Generated {len(logs)} logs with Stealthy and Statistical anomalies."
    )
    return pd.DataFrame(logs)
