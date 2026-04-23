"""
SecuriteAI: Enterprise Anomaly Detection API
-------------------------------------------
Main Author: Saurav Jayakumar
Description: A high-throughput, distributed inference engine designed to protect
Linux environments by analyzing system log sequences. It utilizes an LSTM-Autoencoder
to detect zero-day threats via reconstruction failure.

Features:
- Stateful Sliding Windows via Redis
- Multi-scale Defense (Burst & Slow-Walk Detection)
- Live Artifact Hot-Swapping
- Prometheus Operational Telemetry
"""

import os
import time
import json
import warnings
import logging
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from redis.asyncio import Redis
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app

# Internal SecuriteAI module imports
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline
from api.config import (
    REDIS_HOST,
    REDIS_PORT,
    BUFFER_KEY,
    ANOMALY_HISTORY_KEY,
    RECENT_ANOMALIES_KEY,
    MSE_STREAM_KEY,
    WINDOW_SIZE,
    LONG_WINDOW_SIZE,
    FEEDBACK_DIR,
)
from api.schemas import LogEntry, FeedbackRequest
from api.services import (
    model_artifacts,
    load_model_artifacts,
    send_security_notification,
    save_feedback_to_disk,
    LOG_INGEST_COUNTER,
    PREDICTION_COUNTER,
    ANOMALY_COUNTER,
    MSE_HISTOGRAM,
)

# =================================================================
# 1. ENVIRONMENT & LOGGING SETUP
# =================================================================

# Disable parallelism for tokenizers to prevent deadlocks in Docker
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress verbose warnings from transformers and pandas performance warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Initialize the asynchronous Redis client for distributed state management
redis_client: Redis = Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)


# =================================================================
# 2. APPLICATION LIFECYCLE
# =================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application startup and shutdown sequence.

    1. Ensures the feedback directory exists for auditor corrections.
    2. Clears the Redis buffer to prevent stale data pollution.
    3. Loads model weights and statistical parameters into RAM.
    """
    print("[*] Initializing SecuriteAI Production Inference Engine...")
    os.makedirs(FEEDBACK_DIR, exist_ok=True)

    # Flush existing sliding windows to ensure a clean temporal start
    await redis_client.delete(BUFFER_KEY)
    await redis_client.delete(ANOMALY_HISTORY_KEY)

    # Attempt to cache weights; crash immediately if artifacts are missing
    if not await load_model_artifacts():
        raise RuntimeError("Startup Failure: Critical model artifacts not found.")

    yield
    # Cleanup on shutdown
    model_artifacts.clear()


# Initialize FastAPI with the lifespan manager
app = FastAPI(title="SecuriteAI Enterprise API", lifespan=lifespan)

# Mount the Prometheus metrics endpoint for external scraping
app.mount("/metrics", make_asgi_app())


# =================================================================
# 3. ENDPOINTS
# =================================================================


@app.post("/ingest")
async def ingest_log(entry: LogEntry, background_tasks: BackgroundTasks):
    """
    Main ingestion pipeline for real-time log analysis.

    Processes incoming logs through a dual-scale detection framework:
    - Burst Detection: Analyzes a 20-log window for immediate anomalies.
    - Slow-Walk Detection: Monitors anomaly density over 1,000 logs.

    Args:
        entry (LogEntry): The raw system log data.
        background_tasks: FastAPI handler for non-blocking disk/alert I/O.
    """
    try:
        LOG_INGEST_COUNTER.inc()

        # 1. Update the Redis Sliding Window
        await redis_client.rpush(BUFFER_KEY, json.dumps(entry.model_dump()))  # type: ignore
        await redis_client.ltrim(BUFFER_KEY, -LONG_WINDOW_SIZE, -1)  # type: ignore

        # Check if we have enough context to perform inference
        current_depth = await redis_client.llen(BUFFER_KEY)  # type: ignore
        if current_depth < WINDOW_SIZE:
            return {
                "status": "BUFFERING",
                "current_depth": f"{current_depth}/{WINDOW_SIZE}",
            }

        # 2. Extract the current 20-log window for analysis
        raw_window = await redis_client.lrange(BUFFER_KEY, -WINDOW_SIZE, -1)  # type: ignore
        log_window = [json.loads(log) for log in raw_window]

        # 3. Preprocessing & Feature Engineering
        s_params = model_artifacts["scaler"]
        features = feature_engineering_pipeline(
            clean_linux_logs(pd.DataFrame(log_window)),
            window_size=WINDOW_SIZE,
            scaler_params=(s_params[0], s_params[1]),
            model=model_artifacts["nlp_model"],
        )

        # 4. Neural Inference (LSTM-Autoencoder)
        device, model = model_artifacts["device"], model_artifacts["model"]
        input_tensor = torch.tensor(features).float().to(device)

        with torch.no_grad():
            reconstruction = model(input_tensor)
            # Calculate Mean Squared Error (MSE) across the window
            mse_score = (
                torch
                .mean((reconstruction - input_tensor) ** 2, dim=(1, 2))
                .cpu()
                .item()
            )

        # 5. Dashboard Heartbeat Integration
        await redis_client.lpush(MSE_STREAM_KEY, str(mse_score))  # type: ignore
        await redis_client.ltrim(MSE_STREAM_KEY, 0, 49)  # type: ignore

        # 6. Statistical Risk Assessment
        # Calculate $z = \frac{mse - \mu}{\sigma}$ to determine severity
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

        # 7. Multi-Scale Density Defense (Slow-Walk)
        await redis_client.rpush(ANOMALY_HISTORY_KEY, 1 if is_anomaly else 0)  # type: ignore
        await redis_client.ltrim(ANOMALY_HISTORY_KEY, -LONG_WINDOW_SIZE, -1)  # type: ignore

        history = await redis_client.lrange(ANOMALY_HISTORY_KEY, 0, -1)  # type: ignore
        anomaly_density = sum([int(h) for h in history]) / len(history)

        # Alert if more than 2.5% of recent logs are anomalous
        is_slow_walk = anomaly_density > 0.025

        # 8. Alerting & Dashboard Persistence
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
            # Capture the full window context for auditor drill-down
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
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")


@app.post("/feedback")
async def provide_feedback(
    feedback: FeedbackRequest, background_tasks: BackgroundTasks
):
    """
    Accepts auditor feedback for model governance and retraining.

    This endpoint allows human experts to flag windows for inclusion in the
    next training cycle, closing the GRC loop.
    """
    background_tasks.add_task(save_feedback_to_disk, feedback.model_dump())
    return {"status": "FEEDBACK_QUEUED_FOR_RETRAINING"}


@app.post("/reload")
async def reload_model():
    """
    Triggers a live reload of the model artifacts from disk.

    Used by the Champion-Challenger trigger to hot-swap weights without
    restarting the API or losing the Redis buffer.
    """
    if await load_model_artifacts():
        return {
            "status": "SUCCESS",
            "message": "Production model artifacts reloaded successfully.",
        }
    raise HTTPException(status_code=500, detail="Failed to reload artifacts from disk.")
