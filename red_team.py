import requests
import time
import random
from datetime import datetime

INGEST_URL = "http://localhost:8000/ingest"

# Noise templates to bury the malicious content
NOISE = [
    {"Comp": "kernel", "ID": "E01", "Msg": "Hardware initialization successful"},
    {"Comp": "systemd", "ID": "E02", "Msg": "Started User Manager for UID 1000"},
    {"Comp": "cron", "ID": "E03", "Msg": "pam_unix(cron:session): session opened"},
]


def run_slow_walk():
    """Bypasses burst detection by sending 1 malicious log every 10 minutes."""
    print("🛡️ Starting SecuriteAI Red-Team Attack...")

    while True:
        # 1. Inject Malicious Log
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
        print("[*] Stealth phase complete. Sleeping for 10 minutes...")
        time.sleep(600)


if __name__ == "__main__":
    run_slow_walk()
