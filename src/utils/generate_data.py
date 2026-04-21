import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Generates a synthetic Linux log dataset with three tiers of anomalies:
    1. Statistical Anomalies: Rare IDs (E999) and malicious semantic content.
    2. Sequential Anomalies: Normal IDs and content occurring at high velocity.

    Returns:
        pd.DataFrame: A structured DataFrame containing system logs and content.
    """
    logs = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)  # Retaining 2024 as the baseline year
    current_time = start_time

    # --- 1. CORRELATED TEMPLATE MAPPING ---
    # We map Event IDs to specific Components and Content strings to establish a
    # 'System Identity'. This allows the NLP encoder to learn the semantic
    # grammar of 'Normal' vs 'Anomalous' states.
    template_map = {
        "E01": {"Comp": "kernel", "Msg": "Hardware initialization successful"},
        "E02": {"Comp": "systemd", "Msg": "Started User Manager for UID 1000"},
        "E03": {"Comp": "cron", "Msg": "pam_unix(cron:session): session opened"},
        "E04": {"Comp": "kernel", "Msg": "Memory block allocated at 0x4f3e"},
        "E999": {
            "Comp": "auth-service",
            "Msg": "Failed password for invalid user admin from 192.168.1.42",
        },
    }

    # --- 2. NORMAL DATA (The Baseline) ---
    # Represents steady-state system behavior with random intervals between 5-15 seconds.
    for _ in range(10000):
        current_time += timedelta(seconds=random.randint(5, 15))

        # Pick a normal Event ID and retrieve its correlated metadata
        eid = random.choice(["E01", "E02", "E03", "E04"])
        data = template_map[eid]

        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": data["Comp"],
            "EventId": eid,
            "Content": data["Msg"],  # Correlated unstructured text
        })

    # --- 3. STEALTHY SEQUENTIAL ANOMALY ---
    # An attacker uses NORMAL IDs and NORMAL Content, but in a 'Machine-Gun' burst.
    # This forces the model to learn VELOCITY and TEMPORAL ORDER.
    current_time += timedelta(hours=1)
    for _ in range(500):
        # Velocity is ~100x faster than normal (milliseconds vs seconds)
        current_time += timedelta(milliseconds=random.randint(10, 50))

        # Using normal IDs E01 and E04 (Kernel activities)
        eid = random.choice(["E01", "E04"])
        data = template_map[eid]

        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": data["Comp"],
            "EventId": eid,
            "Content": data["Msg"],
        })

    # --- 4. RAW STATISTICAL ANOMALY ---
    # High-signal anomaly featuring an unknown Event ID and malicious content.
    # This serves as the primary test for the LSTM's reconstruction sensitivity.
    current_time += timedelta(hours=1)
    for _ in range(500):
        current_time += timedelta(seconds=1)
        data = template_map["E999"]

        logs.append({
            "Year": current_time.year,
            "Month": current_time.strftime("%b"),
            "Date": current_time.day,
            "Time": current_time.strftime("%H:%M:%S"),
            "Component": data["Comp"],
            "EventId": "E999",
            "Content": data["Msg"],  # Malicious semantic signal
        })

    print(
        f"[SUCCESS] Generated {len(logs)} logs with correlated Semantic Content for NLP Engineering."
    )
    return pd.DataFrame(logs)
