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

# =================================================================
# 1. CONFIGURATION & REDIS INITIALIZATION
# =================================================================

# Path configuration for model weights and baseline metrics
WEIGHTS_DIR = "artifacts/weights"
PARAMETERS_DIR = "artifacts/parameters"
MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")

# Model dimensions matching the trained LSTM-Autoencoder
INPUT_DIM = 9  # 8 cyclical features + 1 normalized Event ID
HIDDEN_DIM = 64
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
# 2. LIFESPAN MANAGEMENT (Resource Caching)
# =================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application lifecycle, ensuring heavy model weights and
    scaler parameters are cached in RAM to minimize per-request latency.
    """
    print("[*] Initializing Production Inference Engine...")
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

        # Cache artifacts in global state for zero-I/O access
        model_artifacts["model"] = model
        model_artifacts["threshold"] = float(threshold)
        model_artifacts["scaler"] = scaler
        model_artifacts["device"] = device
        model_artifacts["stats"] = {
            "mean": np.mean(loss_metrics),
            "std": np.std(loss_metrics),
        }

        print(f"[*] Success: Model loaded on {device}. Threshold: {threshold:.4f}")

    except Exception as e:
        print(f"[!] Initialization Failure: {e}")
        raise e

    yield
    # Cleanup on shutdown
    model_artifacts.clear()


app = FastAPI(title="SecuriteAI Enterprise API", lifespan=lifespan)

# Expose metrics endpoint for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# =================================================================
# 3. ASYNCHRONOUS SECURITY ALERTING
# =================================================================
async def send_security_notification(mse: float, risk: str):
    """
    Simulated security webhook. In production, this would dispatch to
    PagerDuty, Slack, or an enterprise SIEM.
    """
    alert_payload = {
        "event": "ANOMALY_ALERT",
        "risk_level": risk,
        "reconstruction_mse": round(mse, 6),
        "timestamp": time.time(),
    }
    print(f"[SECURITY NOTIFICATION] Incident Logged: {alert_payload}")


# =================================================================
# 4. DATA VALIDATION (Pydantic Schema)
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


# =================================================================
# 5. INGESTION ENDPOINT (The Async Stream)
# =================================================================
@app.post("/ingest")
async def ingest_log(entry: LogEntry, background_tasks: BackgroundTasks):
    """
    Analyzes a log stream using an atomic sliding window in Redis.
    Transitions to active inference once 20 logs are in the buffer.
    """
    try:
        LOG_INGEST_COUNTER.inc()

        # 1. Update Distributed State (Asynchronous)
        # Push new log to Redis and maintain a fixed window size of 20
        await redis_client.rpush(BUFFER_KEY, json.dumps(entry.model_dump()))  # type: ignore
        await redis_client.ltrim(BUFFER_KEY, -WINDOW_SIZE, -1)  # type: ignore

        # 'await' unwraps the coroutine into a concrete 'int' for comparison
        current_depth = await redis_client.llen(BUFFER_KEY)  # type: ignore

        # 2. Dynamic Logic: Phase 1 (Warm-up / Buffering)
        if current_depth < WINDOW_SIZE:
            return {
                "status": "BUFFERING",
                "logs_required": WINDOW_SIZE - current_depth,
                "current_depth": f"{current_depth}/{WINDOW_SIZE}",
            }

        # 3. Dynamic Logic: Phase 2 (Active Inference)
        PREDICTION_COUNTER.inc()

        # Retrieve the current sliding window from Redis asynchronously
        raw_window = await redis_client.lrange(BUFFER_KEY, 0, -1)  # type: ignore
        log_window = [json.loads(log) for log in raw_window]

        # Transform window into model-ready features
        df = pd.DataFrame(log_window)
        cleaned_df = clean_linux_logs(df)

        s_params = model_artifacts["scaler"]
        features = feature_engineering_pipeline(
            cleaned_df,
            window_size=WINDOW_SIZE,
            scaler_params=(s_params[0], s_params[1]),
        )

        # 4. Neural Network Inference
        device = model_artifacts["device"]
        model = model_artifacts["model"]
        input_tensor = torch.tensor(features).float().to(device)

        with torch.no_grad():
            reconstruction = model(input_tensor)
            mse_score = (
                torch
                .mean((reconstruction - input_tensor) ** 2, dim=(1, 2))
                .cpu()
                .item()
            )

        # 5. Statistical Risk Assessment (Z-Score)
        mu, sigma = model_artifacts["stats"]["mean"], model_artifacts["stats"]["std"]
        z_score = (mse_score - mu) / sigma

        # Risk Mapping Logic based on deviation from training mean
        if z_score < 2:
            severity = "Low"
        elif 2 <= z_score < 5:
            severity = "Medium"
        elif 5 <= z_score < 10:
            severity = "High"
        else:
            severity = "Critical"

        # 6. Telemetry and Asynchronous Alerting
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
        # Standardize error response for distributed observability
        raise HTTPException(status_code=500, detail=f"Streaming Error: {str(e)}")
