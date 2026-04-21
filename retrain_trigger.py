import os
import glob
import subprocess
import time

# CONFIGURATION
FEEDBACK_DIR = "artifacts/feedback"
THRESHOLD = 50  # Number of auditor-verified windows required to trigger retraining
CHECK_INTERVAL = 300  # Check for new feedback every 5 minutes


def count_feedback_files():
    """Counts individual JSON feedback events."""
    return len(glob.glob(os.path.join(FEEDBACK_DIR, "*.json")))


def trigger_retraining():
    """Executes the fine-tuning pipeline as a subprocess."""
    print(
        f"[*] Threshold reached ({THRESHOLD}). Triggering Champion-Challenger Fine-Tuning..."
    )

    # Run the updated pipeline with the --finetune flag
    result = subprocess.run(["python", "-m", "modeling.pipeline", "--finetune"])

    if result.returncode == 0:
        print("[SUCCESS] Fine-tuning complete. Clearing feedback buffer.")
        # Archive or delete processed feedback to reset the trigger
        for f in glob.glob(os.path.join(FEEDBACK_DIR, "*.json")):
            os.remove(f)
    else:
        print(
            "[!] Fine-tuning failed or was discarded. Retaining feedback for next attempt."
        )


def monitor_loop():
    """Continuous monitoring service for GRC feedback."""
    print(f"[*] SecuriteAI Retrain Trigger active. Monitoring {FEEDBACK_DIR}...")
    while True:
        count = count_feedback_files()
        if count >= THRESHOLD:
            trigger_retraining()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # Ensure directory exists before starting
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    monitor_loop()
