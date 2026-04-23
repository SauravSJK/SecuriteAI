"""
SecuriteAI Configuration Module
-------------------------------
Main Author: Saurav Jayakumar
Description: Centralizes all global constants, directory paths, and
environment-specific settings for the SecuriteAI ecosystem.

This file ensures that model dimensions, pathing, and distributed state keys
remain consistent across the API, the retraining trigger, and the dashboard.
"""

import os

# =================================================================
# 1. PATH CONFIGURATION
# =================================================================
# Root directory for all persistent ML artifacts
ARTIFACTS_DIR = "artifacts"

# Granular subdirectories for weights, params, and auditor logs
WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "weights")
PARAMETERS_DIR = os.path.join(ARTIFACTS_DIR, "parameters")
FEEDBACK_DIR = os.path.join(ARTIFACTS_DIR, "feedback")

# Specific file pointers for hot-swapping model state
MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")

# =================================================================
# 2. MODEL DIMENSIONS & HYPERPARAMETERS
# =================================================================
# The input layer must accommodate cyclical time features, normalized
# Event IDs, and the 384-dimensional semantic embedding.
INPUT_DIM = 9 + 384
HIDDEN_DIM = 128

# Temporal context windows:
# WINDOW_SIZE: Short-term burst detection (20 logs).
# LONG_WINDOW_SIZE: Long-term density analysis (1,000 logs) for 'Slow Walk' defense.
WINDOW_SIZE = 20
LONG_WINDOW_SIZE = 1000

# =================================================================
# 3. DISTRIBUTED STATE (REDIS)
# =================================================================
# Connection settings optimized for Docker-based orchestration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379

# Redis key registry for cross-container communication
BUFFER_KEY = "securiteai_sliding_window"
ANOMALY_HISTORY_KEY = "securiteai_anomaly_hits"
RECENT_ANOMALIES_KEY = "securiteai_recent_anomalies"
MSE_STREAM_KEY = "securiteai_mse_stream"
