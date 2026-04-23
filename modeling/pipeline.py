"""
SecuriteAI Modeling Pipeline
----------------------------
Main Author: Saurav Jayakumar
Description: Orchestrates the end-to-end lifecycle of the LSTM-Autoencoder.
Handles synthetic data generation, 'Poisoned Normalization', threshold
definition, and GRC-driven fine-tuning.

Phases:
1. Full Training: Establishes the initial statistical baseline.
2. Fine-Tuning: Refines the model using human-validated feedback.
"""

import os
import argparse
import json
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Core SecuriteAI component imports
from src.models.autoencoder import Autoencoder
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline
from src.utils.generate_data import generate_securiteai_dataset

# Environment configuration for cleaner execution
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# =================================================================
# 1. GLOBAL CONFIGURATION & PATHS
# =================================================================
# Directory structure aligned with production API and Docker volumes
WEIGHTS_DIR = "artifacts/weights"
PARAMETERS_DIR = "artifacts/parameters"
FEEDBACK_DIR = "artifacts/feedback"
VISUALIZATION_DIR = "visualizations"

MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")
VISUAL_PATH = os.path.join(VISUALIZATION_DIR, "securiteai_visual_report.png")

# Model hyper-parameters optimized for log temporal context
INPUT_DIM = 9 + 384  # 9 cyclical/categorical + 384 semantic dimensions
HIDDEN_DIM = 128
WINDOW_SIZE = 20
BATCH_SIZE = 64


# =================================================================
# 2. CORE UTILITY FUNCTIONS
# =================================================================


def get_per_sequence_losses(
    model: Autoencoder, loader: DataLoader, device: torch.device
):
    """
    Computes the Reconstruction MSE for every window in a dataset.

    Avoids averaging to allow for statistical distribution analysis, which
    is critical for defining the 99.5th percentile threshold.

    Returns:
        np.ndarray: A 1D array of MSE scores for each input window.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            # Calculate MSE over the temporal window
            reconstruction = model(x)
            loss = criterion(reconstruction, x).mean(dim=(1, 2))
            losses.extend(loss.cpu().numpy())

    return np.array(losses)


def load_feedback_data():
    """
    Processes auditor-validated feedback as isolated temporal units.

    Utilizes the existing production scaler to ensure feedback data
    is mapped into the correct feature space.

    Returns:
        np.ndarray or None: Stacked 3D tensor of engineered windows.
    """
    feedback_files = glob.glob(os.path.join(FEEDBACK_DIR, "*.json"))
    if not feedback_files:
        return None

    # Sync with production scaling parameters
    s_params = np.load(SCALER_PATH)
    all_engineered_windows = []

    for f_path in feedback_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data["logs"])
            cleaned_df = clean_linux_logs(df)

            # Engineer as a discrete sequence
            window = feature_engineering_pipeline(
                cleaned_df,
                window_size=WINDOW_SIZE,
                scaler_params=(s_params[0], s_params[1]),
            )
            all_engineered_windows.append(window)

    return np.concatenate(all_engineered_windows, axis=0)


# =================================================================
# 3. MAIN PIPELINE ORCHESTRATION
# =================================================================


def main():
    """
    Orchestrates initial training or Champion-Challenger refinement.
    """
    parser = argparse.ArgumentParser(description="SecuriteAI Modeling Pipeline")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable Champion-Challenger fine-tuning mode",
    )
    args = parser.parse_args()

    # Hardware acceleration detection
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if args.finetune:
        # --- PHASE: FINE-TUNING (CHAMPION-CHALLENGER) ---
        print("[*] Entering Fine-Tuning mode...")
        feedback_windows = load_feedback_data()

        if feedback_windows is None:
            print("[!] No feedback data found. Aborting.")
            return

        # Initialize current 'Champion' for competition
        champion = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(DEVICE)
        champion.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

        fb_loader = DataLoader(
            TensorDataset(torch.tensor(feedback_windows).float()), batch_size=BATCH_SIZE
        )

        champ_losses = get_per_sequence_losses(champion, fb_loader, DEVICE)
        champ_mean_mse = np.mean(champ_losses)

        # Use conservative Learning Rate to prevent catastrophic forgetting
        optimizer = torch.optim.Adam(champion.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        print(f"[*] Champion MSE: {champ_mean_mse:.6f}. Fine-tuning...")
        champion.train()
        for _ in range(10):  # Brief pass for refinement
            for batch in fb_loader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(champion(x), x)
                loss.backward()
                optimizer.step()

        # CHAMPION-CHALLENGER VALIDATION
        chall_mean_mse = np.mean(get_per_sequence_losses(champion, fb_loader, DEVICE))
        if chall_mean_mse < champ_mean_mse:
            print(f"[SUCCESS] Challenger MSE: {chall_mean_mse:.6f}. Promoting weights.")
            torch.save(champion.state_dict(), MODEL_WEIGHTS)
        else:
            print("[!] Challenger failed to improve. Pass discarded.")

    else:
        # --- PHASE: FULL TRAINING ---
        print("[*] Generating system logs...")
        raw_logs = generate_securiteai_dataset()

        # ISOLATION NORMALIZATION: FIT ON NORMAL ONLY
        norm_df = raw_logs[raw_logs["Component"] != "auth-service"].copy()
        anom_df = raw_logs[raw_logs["Component"] == "auth-service"].copy()

        clean_norm, clean_anom = clean_linux_logs(norm_df), clean_linux_logs(anom_df)

        # Fit scaler ONLY on normal logs to maximize anomaly 'surprise'
        norm_windows = feature_engineering_pipeline(clean_norm, scaler_path=SCALER_PATH)
        s_params = np.load(SCALER_PATH)

        anom_windows = feature_engineering_pipeline(
            clean_anom, scaler_params=(s_params[0], s_params[1])
        )

        # IID SHUFFLING: Ensures model learns system heartbeat, not just timeline
        indices = list(range(len(norm_windows)))
        random.seed(42)
        random.shuffle(indices)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(norm_windows[indices[:8000]]).float()),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_norm_loader = DataLoader(
            TensorDataset(torch.tensor(norm_windows[indices[8000:]]).float()),
            batch_size=BATCH_SIZE,
        )
        test_anom_loader = DataLoader(
            TensorDataset(torch.tensor(anom_windows).float()), batch_size=BATCH_SIZE
        )

        model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print(f"[*] Training for 100 epochs on {DEVICE}...")
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), x)
                loss.backward()
                optimizer.step()

        # Define 'Normal' as 99.5th percentile of training error
        train_losses = get_per_sequence_losses(model, train_loader, DEVICE)
        threshold = np.percentile(train_losses, 99.5)

        norm_test_losses = get_per_sequence_losses(model, test_norm_loader, DEVICE)
        anom_test_losses = get_per_sequence_losses(model, test_anom_loader, DEVICE)

        # PERFORMANCE REPORT
        print("\n" + "=" * 45 + "\n SECURITEAI: PERFORMANCE REPORT\n" + "=" * 45)
        print(
            f"False Positive Rate:      {(np.sum(norm_test_losses > threshold) / len(norm_test_losses)) * 100:.2f}%"
        )
        print(
            f"Detection Rate (Recall):  {(np.sum(anom_test_losses > threshold) / len(anom_test_losses)) * 100:.2f}%"
        )
        print(
            f"Signal-to-Noise Ratio:    {np.mean(anom_test_losses) / np.mean(norm_test_losses):.2f}x"
        )
        print("=" * 45)

        torch.save(model.state_dict(), MODEL_WEIGHTS)
        np.save(THRESHOLD_PATH, threshold)
        np.save(LOSS_METRICS_PATH, np.array([norm_test_losses]))

        # SKYSCRAPER VISUALIZATION
        print(f"[*] Generating Visual Report: {VISUAL_PATH}")
        all_windows = np.concatenate([norm_windows, anom_windows])
        model.eval()
        with torch.no_grad():
            t_data = torch.tensor(all_windows).float().to(DEVICE)
            mse_scores = (
                torch.mean((model(t_data) - t_data) ** 2, dim=(1, 2)).cpu().numpy()
            )

        plt.figure(figsize=(14, 7))
        plt.plot(mse_scores, color="#1f77b4", alpha=0.7, label="Reconstruction MSE")
        plt.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ({threshold:.4f})",
        )
        plt.yscale("log")  # Log scale highlights massive SNR gap
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.text(
            len(norm_windows) // 2,
            threshold * 2,
            "NORMAL",
            color="green",
            fontweight="bold",
            ha="center",
        )
        plt.text(
            len(norm_windows) + (len(anom_windows) // 2),
            threshold * 50,
            "ATTACK",
            color="red",
            fontweight="bold",
            ha="center",
        )

        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        plt.savefig(VISUAL_PATH, dpi=300)


if __name__ == "__main__":
    main()
