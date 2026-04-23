"""
SecuriteAI Retrain Trigger
--------------------------
Description: A background monitoring service that implements the GRC
'Champion-Challenger' flow. It triggers automated fine-tuning once a
sufficient volume of auditor feedback is collected.
"""

import os
import glob
import subprocess
import time

# CONFIGURATION
FEEDBACK_DIR = "artifacts/feedback"
THRESHOLD = 50  # Windows required to trigger fine-tuning
CHECK_INTERVAL = 300  # 5-minute polling interval


def count_feedback_files():
    """Counts individual auditor-verified feedback events."""
    return len(glob.glob(os.path.join(FEEDBACK_DIR, "*.json")))


def trigger_retraining():
    """
    Executes the fine-tuning pipeline as a subprocess.
    """
    print(
        f"[*] Threshold reached ({THRESHOLD}). Triggering Champion-Challenger Fine-Tuning..."
    )

    # Run the pipeline with the --finetune flag
    result = subprocess.run(["python", "-m", "modeling.pipeline", "--finetune"])

    if result.returncode == 0:
        print("[SUCCESS] Fine-tuning complete. Clearing feedback buffer.")
        # Reset the trigger by removing processed feedback
        for f in glob.glob(os.path.join(FEEDBACK_DIR, "*.json")):
            os.remove(f)
    else:
        print("[!] Fine-tuning failed or discarded. Retaining feedback.")


def monitor_loop():
    """Continuous GRC feedback monitoring loop."""
    print(f"[*] SecuriteAI Retrain Trigger active. Monitoring {FEEDBACK_DIR}...")
    while True:
        count = count_feedback_files()
        if count >= THRESHOLD:
            trigger_retraining()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # Directory initialization
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    monitor_loop()
