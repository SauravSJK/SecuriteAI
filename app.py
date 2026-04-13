from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
import pandas as pd
import torch
from contextlib import asynccontextmanager

# Custom module imports for the SecuriteAI pipeline
from autoencoder import Autoencoder
from clean_log import clean_linux_logs
from feat_eng import feature_engineering_pipeline

# =================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =================================================================
MODEL_DIR = "models"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.npy")

# Architecture specs: 8 cyclical time features + 1 normalized Event ID
INPUT_DIM = 9
HIDDEN_DIM = 64
WINDOW_SIZE = 20

# Global dictionary to persist model artifacts and avoid redundant disk I/O
model_artifacts = {}


# =================================================================
# 2. LIFESPAN MANAGEMENT (STARTUP/SHUTDOWN)
# =================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown logic for the FastAPI application.

    This manager ensures that heavy resources (PyTorch model, weights,
    and scalers) are loaded into memory exactly once when the server starts,
    making individual prediction requests significantly faster.
    """
    print("[*] Loading SecuriteAI artifacts into memory...")

    # Select the best available hardware accelerator
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    try:
        # Initialize the Autoencoder architecture
        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(device)

        # Load pre-trained weights and set to evaluation mode (disables dropout/batchnorm)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()

        # Cache artifacts in the global container
        model_artifacts["model"] = model
        model_artifacts["threshold"] = np.load(THRESHOLD_PATH)
        model_artifacts["scaler"] = np.load(SCALER_PATH)
        model_artifacts["device"] = device

        print(f"[*] Success: Model loaded on {device}")

    except FileNotFoundError as e:
        print(f"[!] Critical Error: Model artifact not found. {e}")
        raise e

    yield

    # Cleanup: Release memory on shutdown
    model_artifacts.clear()
    print("[*] Artifacts cleared. Server shutting down.")


# Initialize FastAPI with the lifespan context manager
app = FastAPI(
    title="SecuriteAI Log Anomaly Detector",
    description="Real-time Linux log analysis using Deep Learning (Autoencoders).",
    lifespan=lifespan,
)


# =================================================================
# 3. DATA VALIDATION SCHEMAS
# =================================================================
class LogEntry(BaseModel):
    """
    Validates the structure of a single Linux log line.
    """

    Year: int = Field(examples=[2024])
    Month: str = Field(examples=["Oct"])
    Date: int = Field(examples=[12])
    Time: str = Field(examples=["14:20:05"])
    Component: str = Field(examples=["systemd"])
    EventId: str = Field(examples=["E02"])


class LogWindow(BaseModel):
    """
    Validates a batch of logs. The model requires a sliding window
    of exactly 20 logs to capture temporal patterns.
    """

    logs: list[LogEntry] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description="A list of exactly 20 log entries.",
    )


# =================================================================
# 4. PREDICTION ENDPOINT
# =================================================================
@app.post("/predict")
async def get_prediction(window: LogWindow):
    """
    Analyzes a window of 20 logs and returns an anomaly status.

    Args:
        window (LogWindow): A JSON object containing 20 log entries.

    Returns:
        dict: A dictionary containing:
            - anomaly (bool): True if the MSE exceeds the threshold.
            - mse_avg (float): The calculated reconstruction error.
            - status (str): A human-readable status message.

    Raises:
        HTTPException: 500 error if processing or inference fails.
    """
    try:
        # 1. Data Preparation: Convert Pydantic objects to Pandas DataFrame
        raw_data = [log.model_dump() for log in window.logs]
        df = pd.DataFrame(raw_data)

        # 2. Preprocessing: Clean strings and extract raw components
        cleaned_df = clean_linux_logs(df)
        s_params = model_artifacts["scaler"]

        # 3. Feature Engineering: Convert timestamps to cyclical features and scale data
        features = feature_engineering_pipeline(
            cleaned_df,
            window_size=WINDOW_SIZE,
            scaler_params=(s_params[0], s_params[1]),
        )

        # 4. Neural Network Inference
        device = model_artifacts["device"]
        model = model_artifacts["model"]

        # Prepare input tensor (batch_size=1, window_size=20, input_dim=9)
        input_tensor = torch.tensor(features).float().to(device)

        with torch.no_grad():
            reconstruction = model(input_tensor)

            # Calculate Mean Squared Error (MSE) between input and reconstruction
            # This represents how "surprised" the model is by the data
            mse_score = (
                torch.mean((reconstruction - input_tensor) ** 2, dim=(1, 2))
                .cpu()
                .numpy()
            )

        # 5. Threshold Comparison
        avg_mse = mse_score.mean()
        is_anomaly = avg_mse > model_artifacts["threshold"]

        return {
            "threshold": float(model_artifacts["threshold"]),
            "anomaly": bool(is_anomaly),
            "mse_avg": float(avg_mse),
            "status": "⚠️ ANOMALY DETECTED" if is_anomaly else "✅ SYSTEM HEALTHY",
        }

    except Exception as e:
        # Catch-all for unexpected errors during the pipeline
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")
