"""
SecuriteAI Red-Team Attack Simulator
------------------------------------
Description: Generates an adversarial 'Slow Walk' attack. This bypasses
short-term burst detection by interleaving malicious activity with noise
and utilizing temporal delays.

Target: Validates the efficacy of the Long-Window Density Check.
"""

import requests
import time
import random
from datetime import datetime

# API Ingestion point
INGEST_URL = "http://localhost:8000/ingest"

# Noise templates to bury the malicious content
NOISE = [
    {"Comp": "kernel", "ID": "E01", "Msg": "Hardware initialization successful"},
    {"Comp": "systemd", "ID": "E02", "Msg": "Started User Manager for UID 1000"},
    {"Comp": "cron", "ID": "E03", "Msg": "pam_unix(cron:session): session opened"},
]


def run_slow_walk():
    """
    Executes a stealthy attack by diluting the temporal context window.
    """
    print("🛡️ Starting SecuriteAI Red-Team Attack...")

    while True:
        # 1. Inject Malicious Log
        # Uses normal EventID but malicious content to test behavioral detection.
        payload = {
            "Year": 2024,
            "Month": "Jan",
            "Date": 15,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Component": "auth-service",
            "EventId": "E02",
            "Content": "CRITICAL: Unauthorized attempt to overwrite /etc/shadow",
        }
        print(f"[*] [{payload['Time']}] Injecting Malicious Content...")
        requests.post(INGEST_URL, json=payload)

        # 2. Inject 19 Normal Logs to 'Dilute' the Window
        # This ensures the 20-log window MSE remains below the sudden-burst threshold.
        print("[*] Diluting window with 19 normal logs...")
        for _ in range(19):
            t = random.choice(NOISE)
            noise_payload = {
                "Year": 2024,
                "Month": "Jan",
                "Date": 15,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Component": t["Comp"],
                "EventId": t["ID"],
                "Content": t["Msg"],
            }
            requests.post(INGEST_URL, json=noise_payload)

        # 3. Wait to evade velocity alarms
        # Bypasses rate-limiting and short-term anomaly density checks.
        print("[*] Stealth phase complete. Sleeping for 10 minutes...")
        time.sleep(600)


if __name__ == "__main__":
    run_slow_walk()
