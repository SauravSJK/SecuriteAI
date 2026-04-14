from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
import os
import pandas as pd
import torch
import time
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, make_asgi_app

# Custom module imports
from src.models.autoencoder import Autoencoder
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline

# =================================================================
# 1. CONFIGURATION & METRICS DEFINITION
# =================================================================
WEIGHTS_DIR = "artifacts/weights"
PARAMETERS_DIR = "artifacts/parameters"
MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")

INPUT_DIM = 9
HIDDEN_DIM = 64
WINDOW_SIZE = 20

# Operational Telemetry: Tracking model behavior in real-time
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

# Global container for model artifacts and statistical baselines
model_artifacts = {}


# =================================================================
# 2. LIFESPAN MANAGEMENT
# =================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application lifecycle, ensuring heavy artifacts and
    statistical baselines are cached in memory to minimize disk I/O.
    """
    print("[*] Initializing Production Inference Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load Architecture and Weights
        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()

        # Load thresholds and scaling parameters
        threshold = np.load(THRESHOLD_PATH)
        scaler = np.load(SCALER_PATH)

        # Load training loss distribution for real-time Z-score calculation
        loss_metrics = np.load(LOSS_METRICS_PATH)

        # Cache artifacts in global state
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
    model_artifacts.clear()


app = FastAPI(title="SecuriteAI Enterprise API", lifespan=lifespan)

# Expose metrics endpoint for Prometheus scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# =================================================================
# 3. BACKGROUND TASKS
# =================================================================
async def send_security_notification(mse: float, risk: str):
    """
    Dispatches automated alerts when anomalies are detected.
    """
    alert_payload = {
        "event": "ANOMALY_ALERT",
        "risk_level": risk,
        "reconstruction_mse": round(mse, 6),
        "timestamp": time.time(),
    }
    # Simulated webhook delivery
    print(f"[SECURITY NOTIFICATION] Incident Logged: {alert_payload}")


# =================================================================
# 4. DATA VALIDATION
# =================================================================
class LogEntry(BaseModel):
    Year: int = Field(
        ..., ge=2000, le=2100, description="Year of the log entry", examples=[2023]
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


class LogWindow(BaseModel):
    """
    Mandates a sequence of at least {WINDOW_SIZE} log entries to form a valid temporal window.
    """

    logs: list[LogEntry] = Field(
        ...,
        description=f"A sequence of {WINDOW_SIZE} log entries representing a time window for analysis.",
        min_length=WINDOW_SIZE,
    )


class LogBatch(BaseModel):
    """
    Supports vectorized batch processing by accepting multiple 20-log windows.
    """

    batch: list[LogWindow] = Field(
        ..., description="List of validated log windows for high-throughput inference."
    )


# =================================================================
# 5. INFERENCE ENDPOINT
# =================================================================
@app.post("/predict")
async def get_prediction(window_batch: LogBatch, background_tasks: BackgroundTasks):
    """
    Analyzes log sequences, calculates statistical risk, and triggers
    asynchronous telemetry reporting.
    """
    try:
        results = []
        s_params = model_artifacts["scaler"]
        device = model_artifacts["device"]
        model = model_artifacts["model"]

        # Process each window in the batch
        for window in window_batch.batch:
            PREDICTION_COUNTER.inc()

            # Feature Engineering
            df = pd.DataFrame([log.model_dump() for log in window.logs])
            cleaned_df = clean_linux_logs(df)
            features = feature_engineering_pipeline(
                cleaned_df,
                window_size=WINDOW_SIZE,
                scaler_params=(s_params[0], s_params[1]),
            )

            # Vectorized Inference
            input_tensor = torch.tensor(features).float().to(device)
            with torch.no_grad():
                reconstruction = model(input_tensor)
                mse_score = (
                    torch.mean((reconstruction - input_tensor) ** 2, dim=(1, 2))
                    .cpu()
                    .item()
                )

            # Statistical Risk Assessment (Z-Score)
            mu, sigma = (
                model_artifacts["stats"]["mean"],
                model_artifacts["stats"]["std"],
            )
            z_score = (mse_score - mu) / sigma

            if z_score < 2:
                risk_level = "Low"
            elif 2 <= z_score < 5:
                risk_level = "Medium"
            elif 5 <= z_score < 10:
                risk_level = "High"
            else:
                risk_level = "Critical"

            # Telemetry and Alerting
            is_anomaly = mse_score > model_artifacts["threshold"]
            MSE_HISTOGRAM.observe(mse_score)

            if is_anomaly:
                ANOMALY_COUNTER.inc()
                background_tasks.add_task(
                    send_security_notification, mse_score, risk_level
                )

            results.append(
                {
                    "anomaly_detected": bool(is_anomaly),
                    "risk_assessment": {
                        "z_score": round(float(z_score), 4),
                        "severity": risk_level,
                    },
                    "mse": float(mse_score),
                }
            )

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")
