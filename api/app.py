"""
SecuriteAI: Enterprise Anomaly Detection API
Description: A distributed, high-throughput inference engine for Linux log analysis.
Utilizes an LSTM-Autoencoder to detect security threats via reconstruction error.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import time
import json
from redis.asyncio import Redis  # Explicit async import to resolve Pylance issues
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, make_asgi_app

# Custom module imports for the SecuriteAI pipeline
from src.models.autoencoder import Autoencoder
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline

# Hugging Face & SentenceTransformer warnings can be verbose; we will suppress them for cleaner output
import warnings
import logging

# 1. Silence Hugging Face & Tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. Suppress library-specific loggers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# 3. Suppress the specific Pandas PerformanceWarning
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# =================================================================
# 1. CONFIGURATION & REDIS INITIALIZATION
# =================================================================

# Path configuration for model weights and baseline metrics
WEIGHTS_DIR = "artifacts/weights"
PARAMETERS_DIR = "artifacts/parameters"
FEEDBACK_DIR = "artifacts/feedback"
MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")

# Model dimensions matching the trained LSTM-Autoencoder
INPUT_DIM = (
    9 + 384
)  # 8 cyclical features + 1 normalized Event ID + 384 embedding dimensions
HIDDEN_DIM = 128
WINDOW_SIZE = 20

# Distributed State: Redis ensures continuity across multiple container replicas
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# Explicitly type hint the client to satisfy strict Pylance checking
redis_client: Redis = Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
BUFFER_KEY = "securiteai_sliding_window"

# Operational Telemetry for Prometheus/Grafana monitoring
MSE_HISTOGRAM = Histogram(
    "securiteai_mse_reconstruction_error",
    "Distribution of reconstruction MSE scores across incoming requests",
    buckets=[0.01, 0.05, 0.1, 0.138, 0.2, 0.5, 1.0, 5.0],
)
ANOMALY_COUNTER = Counter(
    "securiteai_anomalies_total", "Running count of detected security anomalies"
)
PREDICTION_COUNTER = Counter(
    "securiteai_predictions_total", "Total volume of processed log windows"
)
LOG_INGEST_COUNTER = Counter(
    "securiteai_logs_ingested_total", "Total volume of raw logs hitting the system"
)

# Global container for model artifacts and statistical baselines
model_artifacts = {}


# =================================================================
# 2. ARTIFACT LOADING HELPER
# =================================================================
async def load_model_artifacts():
    """
    Centralized logic to load model weights and parameters into RAM.
    This allows for both initial startup and 'Live Reloads'.
    """
    print("[*] Synchronizing model artifacts with RAM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load Model Architecture and Weights
        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()

        # Load Thresholds and Scaling Parameters
        threshold = np.load(THRESHOLD_PATH)
        scaler = np.load(SCALER_PATH)

        # Load training loss distribution for real-time Z-score calculation
        loss_metrics = np.load(LOSS_METRICS_PATH)

        # Load NLP Encoder ONCE and cache it
        print("[*] Loading Semantic Encoder (all-MiniLM-L6-v2)...")
        nlp_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        # Update global state atomically
        model_artifacts.update({
            "model": model,
            "threshold": float(threshold),
            "scaler": scaler,
            "device": device,
            "stats": {
                "mean": np.mean(loss_metrics),
                "std": np.std(loss_metrics),
            },
            "nlp_model": nlp_model,
        })
        print(
            f"[*] Success: Model synchronized on {device}. Threshold: {threshold:.4f}"
        )
        return True
    except Exception as e:
        print(f"[!] Artifact Sync Failure: {e}")
        return False


# =================================================================
# 3. LIFESPAN MANAGEMENT (Resource Caching)
# =================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application lifecycle, ensuring heavy model weights and
    scaler parameters are cached in RAM to minimize per-request latency.
    """
    print("[*] Initializing Production Inference Engine...")

    # Ensure feedback directory exists
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    # Clear Redis buffer on startup to ensure a clean state
    await redis_client.delete(BUFFER_KEY)

    # Initial load of all artifacts
    success = await load_model_artifacts()
    if not success:
        raise RuntimeError("Failed to initialize model artifacts on startup.")

    yield
    # Cleanup on shutdown
    model_artifacts.clear()


app = FastAPI(title="SecuriteAI Enterprise API", lifespan=lifespan)

# Expose metrics endpoint for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# =================================================================
# 4. ASYNC UTILITIES
# =================================================================
async def send_security_notification(mse: float, risk: str):
    """Simulated security alert dispatch."""
    alert_payload = {
        "event": "ANOMALY_ALERT",
        "risk_level": risk,
        "reconstruction_mse": round(mse, 6),
        "timestamp": time.time(),
    }
    print(f"[SECURITY NOTIFICATION] Incident Logged: {alert_payload}")


def save_feedback_to_disk(feedback_data: dict):
    """Offloads disk I/O for auditor feedback."""
    timestamp = int(time.time())
    file_path = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(feedback_data, f, indent=4)


# =================================================================
# 5. DATA VALIDATION (Pydantic Schema)
# =================================================================
class LogEntry(BaseModel):
    Year: int = Field(
        ..., ge=2000, le=2100, description="Year of the log entry", examples=[2024]
    )
    Month: str = Field(
        ...,
        pattern="^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
        description="Month of the log entry",
        examples=["Jan"],
    )
    Date: int = Field(
        ..., ge=1, le=31, description="Day of the log entry", examples=[15]
    )
    Time: str = Field(..., description="Time of the log entry", examples=["14:23:45"])
    Component: str = Field(..., description="Component from the log entry")
    EventId: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Event ID from the log entry",
        examples=["E02"],
    )
    Content: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Unstructured content of the log entry",
        examples=["Started User Manager for UID 1000"],
    )


class FeedbackRequest(BaseModel):
    logs: list[LogEntry] = Field(
        ...,
        description="List of log entries for feedback",
        min_length=20,
        max_length=20,
    )


# =================================================================
# 6. ENDPOINTS
# =================================================================


@app.post("/ingest")
async def ingest_log(entry: LogEntry, background_tasks: BackgroundTasks):
    """
    Analyzes a log stream using an atomic sliding window in Redis.
    """
    try:
        LOG_INGEST_COUNTER.inc()
        await redis_client.rpush(BUFFER_KEY, json.dumps(entry.model_dump()))  # type: ignore
        await redis_client.ltrim(BUFFER_KEY, -WINDOW_SIZE, -1)  # type: ignore
        current_depth = await redis_client.llen(BUFFER_KEY)  # type: ignore

        if current_depth < WINDOW_SIZE:
            return {
                "status": "BUFFERING",
                "logs_required": WINDOW_SIZE - current_depth,
                "current_depth": f"{current_depth}/{WINDOW_SIZE}",
            }

        PREDICTION_COUNTER.inc()
        raw_window = await redis_client.lrange(BUFFER_KEY, 0, -1)  # type: ignore
        log_window = [json.loads(log) for log in raw_window]

        df = pd.DataFrame(log_window)
        cleaned_df = clean_linux_logs(df)

        s_params = model_artifacts["scaler"]
        features = feature_engineering_pipeline(
            cleaned_df,
            window_size=WINDOW_SIZE,
            scaler_params=(s_params[0], s_params[1]),
            model=model_artifacts["nlp_model"],
        )

        device, model = model_artifacts["device"], model_artifacts["model"]
        input_tensor = torch.tensor(features).float().to(device)

        with torch.no_grad():
            reconstruction = model(input_tensor)
            mse_score = (
                torch
                .mean((reconstruction - input_tensor) ** 2, dim=(1, 2))
                .cpu()
                .item()
            )

        mu, sigma = model_artifacts["stats"]["mean"], model_artifacts["stats"]["std"]
        z_score = (mse_score - mu) / sigma
        severity = (
            "Critical"
            if z_score >= 10
            else "High"
            if z_score >= 5
            else "Medium"
            if z_score >= 2
            else "Low"
        )

        is_anomaly = mse_score > model_artifacts["threshold"]
        MSE_HISTOGRAM.observe(mse_score)

        if is_anomaly:
            ANOMALY_COUNTER.inc()
            background_tasks.add_task(send_security_notification, mse_score, severity)

        return {
            "status": "ACTIVE_STREAM",
            "anomaly_detected": bool(is_anomaly),
            "risk": {"z_score": round(float(z_score), 4), "severity": severity},
            "mse": float(mse_score),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming Error: {str(e)}")


@app.post("/feedback")
async def provide_feedback(
    feedback: FeedbackRequest, background_tasks: BackgroundTasks
):
    """Accepts auditor feedback via background tasks."""
    try:
        background_tasks.add_task(save_feedback_to_disk, feedback.model_dump())
        return {"status": "FEEDBACK_QUEUED_FOR_RETRAINING"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback Queue Error: {str(e)}")


@app.post("/reload")
async def reload_model():
    """
    Triggers a live reload of model weights and parameters.
    Required for promoting a 'Challenger' model after automated retraining.
    """
    success = await load_model_artifacts()
    if success:
        return {
            "status": "SUCCESS",
            "message": "Production model artifacts reloaded successfully.",
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model artifacts. Check server logs.",
        )
