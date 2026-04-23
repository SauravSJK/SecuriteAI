"""
SecuriteAI Synthetic Data Engine
--------------------------------
Description: Generates a Linux log dataset with multi-tier anomalies.
Includes steady-state 'Normal' logs, statistical 'unknown' IDs, and
velocity-based 'Machine-Gun' sequential bursts.
"""

import pandas as pd
import random
from datetime import datetime, timedelta


def generate_securiteai_dataset() -> pd.DataFrame:
    """
    Synthesizes a correlated Linux log dataset.

    Tiers:
    1. Normal: Stable system heartbeat.
    2. Sequential: Normal events at lethal velocity.
    3. Statistical: Unknown Event IDs (E999) and malicious content.
    """
    logs = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    current_time = start_time

    # Establishment of 'System Identity' templates
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

    # --- THE BASELINE (10,000 logs) ---
    for _ in range(10000):
        current_time += timedelta(seconds=random.randint(5, 15))
        eid = random.choice(["E01", "E02", "E03", "E04"])
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

    # --- SEQUENTIAL ANOMALY (Machine-Gun Burst) ---
    # Attacker activity using NORMAL IDs at 100x velocity.
    current_time += timedelta(hours=1)
    for _ in range(500):
        current_time += timedelta(milliseconds=random.randint(10, 50))
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

    # --- STATISTICAL ANOMALY (Extreme Surprise) ---
    # Malicious semantic content with rare Event IDs.
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
            "Content": data["Msg"],
        })

    return pd.DataFrame(logs)
