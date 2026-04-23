"""
SecuriteAI: Enterprise Anomaly Detection API
Description: A distributed, high-throughput inference engine for Linux log analysis.
Utilizes an LSTM-Autoencoder to detect security threats via reconstruction error,
supplemented by long-term density analysis for stealth attack detection.
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
from redis.asyncio import Redis
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
INPUT_DIM = 9 + 384  # 8 cyclical + 1 normalized Event ID + 384 embedding dims
HIDDEN_DIM = 128
WINDOW_SIZE = 20
LONG_WINDOW_SIZE = 1000  # Threshold for 'Slow Walk' defense

# Distributed State Management via Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client: Redis = Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

# Redis Keys for various data streams
BUFFER_KEY = "securiteai_sliding_window"
ANOMALY_HISTORY_KEY = "securiteai_anomaly_hits"  # Tracks long-term hit frequency
RECENT_ANOMALIES_KEY = "securiteai_recent_anomalies"  # Data hook for Dashboard Explorer
MSE_STREAM_KEY = "securiteai_mse_stream"  # Stream of MSE scores for dashboard heartbeat

# Operational Telemetry for Prometheus monitoring
MSE_HISTOGRAM = Histogram(
    "securiteai_mse_reconstruction_error",
    "Distribution of reconstruction MSE scores",
    buckets=[0.01, 0.05, 0.1, 0.138, 0.2, 0.5, 1.0, 5.0],
)
ANOMALY_COUNTER = Counter(
    "securiteai_anomalies_total", "Count of detected security anomalies"
)
PREDICTION_COUNTER = Counter(
    "securiteai_predictions_total", "Total processed log windows"
)
LOG_INGEST_COUNTER = Counter(
    "securiteai_logs_ingested_total", "Total ingested raw logs"
)

# Global container for model artifacts
model_artifacts = {}


# =================================================================
# 2. ARTIFACT LOADING HELPER
# =================================================================
async def load_model_artifacts():
    """
    Centralized logic to load or refresh model weights and baseline metrics.
    Refactoring this allows for seamless 'Live Reloads'.
    """
    print("[*] Synchronizing model artifacts with RAM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Initialize and load weights for the LSTM-Autoencoder
        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()

        # Load statistical thresholds and scaling parameters
        threshold = np.load(THRESHOLD_PATH)
        scaler = np.load(SCALER_PATH)
        loss_metrics = np.load(LOSS_METRICS_PATH)

        # Initialize the NLP Semantic Encoder
        print("[*] Loading Semantic Encoder (all-MiniLM-L6-v2)...")
        nlp_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        # Atomic update of the global state
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
# 3. LIFESPAN MANAGEMENT
# =================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles resource caching and directory preparation on startup."""
    print("[*] Initializing Production Inference Engine...")
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    # Clear the Redis state to prevent temporal pollution on restart
    await redis_client.delete(BUFFER_KEY)
    await redis_client.delete(ANOMALY_HISTORY_KEY)

    # Initial load of artifacts
    if not await load_model_artifacts():
        raise RuntimeError("Failed to initialize model artifacts on startup.")

    yield
    model_artifacts.clear()


app = FastAPI(title="SecuriteAI Enterprise API", lifespan=lifespan)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# =================================================================
# 4. ASYNC UTILITIES
# =================================================================
async def send_security_notification(mse: float, risk: str):
    """Dispatches a simulated security alert webhook."""
    print(f"[SECURITY NOTIFICATION] Incident Logged: {risk} Risk (MSE: {mse:.6f})")


def save_feedback_to_disk(feedback_data: dict):
    """Offloads blocking disk I/O for auditor feedback."""
    timestamp = int(time.time())
    file_path = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(feedback_data, f, indent=4)


# =================================================================
# 5. DATA VALIDATION (Pydantic Schemas)
# =================================================================
class LogEntry(BaseModel):
    Year: int = Field(..., ge=2000, le=2100, examples=[2024])
    Month: str = Field(
        ...,
        pattern="^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
        examples=["Jan"],
    )
    Date: int = Field(..., ge=1, le=31, examples=[15])
    Time: str = Field(..., examples=["14:23:45"])
    Component: str = Field(...)
    EventId: str = Field(..., min_length=1, max_length=255, examples=["E02"])
    Content: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        examples=["Started User Manager for UID 1000"],
    )


class FeedbackRequest(BaseModel):
    logs: list[LogEntry] = Field(..., min_length=20, max_length=20)


# =================================================================
# 6. ENDPOINTS
# =================================================================


@app.post("/ingest")
async def ingest_log(entry: LogEntry, background_tasks: BackgroundTasks):
    """
    Performs dual-scale analysis:
    1. Short-term Burst Detection (20 logs).
    2. Long-term Anomaly Density Analysis (1000 logs) to catch 'Slow-Walk' attacks.
    """
    try:
        LOG_INGEST_COUNTER.inc()

        # Maintain a long-term buffer of 1000 logs for multi-scale analysis
        await redis_client.rpush(BUFFER_KEY, json.dumps(entry.model_dump()))  # type: ignore
        await redis_client.ltrim(BUFFER_KEY, -LONG_WINDOW_SIZE, -1)  # type: ignore
        current_depth = await redis_client.llen(BUFFER_KEY)  # type: ignore

        if current_depth < WINDOW_SIZE:
            return {
                "status": "BUFFERING",
                "current_depth": f"{current_depth}/{WINDOW_SIZE}",
            }

        # --- PHASE 1: SHORT-TERM INFERENCE (20 Logs) ---
        raw_window = await redis_client.lrange(BUFFER_KEY, -WINDOW_SIZE, -1)  # type: ignore
        log_window = [json.loads(log) for log in raw_window]

        df = pd.DataFrame(log_window)
        s_params = model_artifacts["scaler"]
        features = feature_engineering_pipeline(
            clean_linux_logs(df),
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

        # Push every MSE to a dedicated stream for the dashboard heartbeat
        await redis_client.lpush(MSE_STREAM_KEY, str(mse_score))  # type: ignore
        await redis_client.ltrim(MSE_STREAM_KEY, 0, 49)  # type: ignore # Keep the last 50 points

        # Statistical Risk Mapping (Z-Score)
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

        # --- PHASE 2: LONG-TERM DENSITY DEFENSE (Wednesday) ---
        # Track hit history over the last 1000 entries
        await redis_client.rpush(ANOMALY_HISTORY_KEY, 1 if is_anomaly else 0)  # type: ignore
        await redis_client.ltrim(ANOMALY_HISTORY_KEY, -LONG_WINDOW_SIZE, -1)  # type: ignore

        history = await redis_client.lrange(ANOMALY_HISTORY_KEY, 0, -1)  # type: ignore
        anomaly_density = sum([int(h) for h in history]) / len(history)

        # Trigger 'Slow Walk' alert if anomaly density exceeds 2.5%
        is_slow_walk = anomaly_density > 0.025

        # --- PHASE 3: DASHBOARD HOOKS (Thursday) ---
        if is_anomaly or is_slow_walk:
            anomaly_payload = {
                "timestamp": time.time(),
                "mse": float(mse_score),
                "severity": severity,
                "reason": "Sustained Stealth Attack"
                if is_slow_walk
                else "Sudden Burst Anomaly",
                "logs": log_window,
            }
            # Store the high-MSE window for the Dashboard Log Explorer
            await redis_client.lpush(RECENT_ANOMALIES_KEY, json.dumps(anomaly_payload))  # type: ignore
            await redis_client.ltrim(RECENT_ANOMALIES_KEY, 0, 19)  # type: ignore

            ANOMALY_COUNTER.inc()
            background_tasks.add_task(send_security_notification, mse_score, severity)

        PREDICTION_COUNTER.inc()
        MSE_HISTOGRAM.observe(mse_score)

        return {
            "status": "ACTIVE_STREAM",
            "anomaly_detected": bool(is_anomaly),
            "slow_walk_detected": bool(is_slow_walk),
            "anomaly_density": f"{anomaly_density:.2%}",
            "risk": {"z_score": round(float(z_score), 4), "severity": severity},
            "mse": float(mse_score),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming Error: {str(e)}")


@app.post("/feedback")
async def provide_feedback(
    feedback: FeedbackRequest, background_tasks: BackgroundTasks
):
    """Accepts auditor feedback via background tasks for GRC governance."""
    try:
        background_tasks.add_task(save_feedback_to_disk, feedback.model_dump())
        return {"status": "FEEDBACK_QUEUED_FOR_RETRAINING"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model():
    """Triggers a live reload of model weights and parameters."""
    if await load_model_artifacts():
        return {
            "status": "SUCCESS",
            "message": "Model artifacts reloaded successfully.",
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload artifacts.")
