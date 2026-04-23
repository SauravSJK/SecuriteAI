"""
SecuriteAI Core Services
------------------------
Description: Encapsulates ML model management, background notifications,
and operational telemetry.

This module handles the heavy lifting of loading weights, calculating
statistical risk, and persisting auditor feedback to disk.
"""

import os
import time
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram

# Internal SecuriteAI imports for model and configuration
from src.models.autoencoder import Autoencoder
from api.config import (
    MODEL_WEIGHTS,
    THRESHOLD_PATH,
    SCALER_PATH,
    LOSS_METRICS_PATH,
    INPUT_DIM,
    HIDDEN_DIM,
    WINDOW_SIZE,
    FEEDBACK_DIR,
)

# =================================================================
# 1. OPERATIONAL TELEMETRY (PROMETHEUS)
# =================================================================
# Tracks the distribution of reconstruction error for threshold tuning
MSE_HISTOGRAM = Histogram(
    "securiteai_mse_reconstruction_error",
    "Distribution of reconstruction MSE scores",
    buckets=[0.01, 0.05, 0.1, 0.138, 0.2, 0.5, 1.0, 5.0],
)

# Counters for high-level dashboard metrics
ANOMALY_COUNTER = Counter(
    "securiteai_anomalies_total", "Count of detected security anomalies"
)
PREDICTION_COUNTER = Counter(
    "securiteai_predictions_total", "Total processed log windows"
)
LOG_INGEST_COUNTER = Counter(
    "securiteai_logs_ingested_total", "Total ingested raw logs"
)

# Global thread-safe container for active model artifacts
model_artifacts = {}


async def load_model_artifacts():
    """
    Centralized logic to load or refresh model weights and baseline metrics.

    This function performs an atomic update of the `model_artifacts` container,
    allowing for 'Live Reloads' without API downtime.

    Returns:
        bool: True if synchronization with RAM was successful, False otherwise.
    """
    print("[*] Synchronizing model artifacts with RAM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the LSTM-Autoencoder weights
        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()

        # Load statistical parameters for Z-score risk mapping
        # $z = \frac{mse - \mu}{\sigma}$
        threshold = np.load(THRESHOLD_PATH)
        scaler = np.load(SCALER_PATH)
        loss_metrics = np.load(LOSS_METRICS_PATH)

        # Initialize the NLP semantic encoder
        nlp_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        model_artifacts.update({
            "model": model,
            "threshold": float(threshold),
            "scaler": scaler,
            "device": device,
            "stats": {"mean": np.mean(loss_metrics), "std": np.std(loss_metrics)},
            "nlp_model": nlp_model,
        })
        print(
            f"[*] Success: Model synchronized on {device}. Threshold: {threshold:.4f}"
        )
        return True
    except Exception as e:
        print(f"[!] Artifact Sync Failure: {e}")
        return False


async def send_security_notification(mse: float, risk: str):
    """
    Dispatches a simulated security alert webhook for an incident.

    Args:
        mse (float): The reconstruction error that triggered the alert.
        risk (str): The calculated severity (Low, Medium, High, Critical).
    """
    print(f"[SECURITY NOTIFICATION] Incident Logged: {risk} Risk (MSE: {mse:.6f})")


def save_feedback_to_disk(feedback_data: dict):
    """
    Persists auditor-corrected logs for future model fine-tuning.

    This function offloads blocking disk I/O to a background task to ensure
    ingestion latency remains unaffected.

    Args:
        feedback_data (dict): The log window payload submitted by the auditor.
    """
    timestamp = int(time.time())
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    file_path = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(feedback_data, f, indent=4)
