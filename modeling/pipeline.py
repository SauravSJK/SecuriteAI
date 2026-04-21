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

# SecuriteAI core component imports
from src.models.autoencoder import Autoencoder
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline
from src.utils.generate_data import generate_securiteai_dataset

# Configuration of library-specific environment variables for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# =================================================================
# 1. GLOBAL CONFIGURATION & PATHS
# =================================================================
# Artifact directories align with the production API expectations
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
INPUT_DIM = 9 + 384  # 9 base features + 384 NLP embedding dimensions
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
    Computes the Mean Squared Error (MSE) for every window in a dataset.

    This function avoids reduction (averaging) to allow for statistical
    analysis of the reconstruction error distribution. This is critical
    for defining the anomaly threshold.

    Args:
        model: The trained LSTM-Autoencoder model.
        loader: A PyTorch DataLoader containing log sequences.
        device: The compute device (CPU/GPU) to perform inference on.

    Returns:
        np.ndarray: A 1D array of MSE scores, one for each input window.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            # Calculate MSE over the flattened temporal window (WINDOW_SIZE * INPUT_DIM)
            reconstruction = model(x)
            loss = criterion(reconstruction, x).mean(dim=(1, 2))
            losses.extend(loss.cpu().numpy())

    return np.array(losses)


def load_feedback_data():
    """
    Individually processes auditor-validated feedback files.

    To maintain temporal integrity, each 20-log feedback window is
    engineered as an isolated unit. This prevents interleaving of
    unrelated log streams during the cleaning phase.

    Returns:
        np.ndarray or None: A stacked 3D tensor of engineered log windows,
                           or None if the feedback directory is empty.
    """
    feedback_files = glob.glob(os.path.join(FEEDBACK_DIR, "*.json"))
    if not feedback_files:
        return None

    # Load the baseline scaler to ensure feedback data matches training space
    s_params = np.load(SCALER_PATH)
    all_engineered_windows = []

    for f_path in feedback_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            # Convert raw JSON logs into a temporary DataFrame for feature extraction
            df = pd.DataFrame(data["logs"])
            cleaned_df = clean_linux_logs(df)

            # Engineer the window as a discrete temporal sequence
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
    Main entry point for initial training and GRC-driven fine-tuning.
    """
    parser = argparse.ArgumentParser(description="SecuriteAI Modeling Pipeline")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable Champion-Challenger fine-tuning mode",
    )
    args = parser.parse_args()

    # Hardware acceleration detection (NVIDIA CUDA, Apple MPS, or CPU)
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if args.finetune:
        # --- PHASE: FINE-TUNING (CHAMPION-CHALLENGER) ---
        # Refines the model using human-validated False Positives
        print("[*] Entering Fine-Tuning mode...")
        feedback_windows = load_feedback_data()

        if feedback_windows is None:
            print("[!] No feedback data found in artifacts/feedback. Aborting.")
            return

        # Initialize the current "Champion" model from disk
        champion = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(DEVICE)
        champion.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

        fb_loader = DataLoader(
            TensorDataset(torch.tensor(feedback_windows).float()), batch_size=BATCH_SIZE
        )

        # Establish a baseline performance for the Champion on the feedback set
        champ_losses = get_per_sequence_losses(champion, fb_loader, DEVICE)
        champ_mean_mse = np.mean(champ_losses)

        # Optimization: Use a lower learning rate (1e-4) to prevent catastrophic forgetting
        optimizer = torch.optim.Adam(champion.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        print(f"[*] Champion MSE on feedback: {champ_mean_mse:.6f}. Fine-tuning...")
        champion.train()
        for _ in range(10):  # Controlled 10-epoch pass for refinement
            for batch in fb_loader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(champion(x), x)
                loss.backward()
                optimizer.step()

        # Validation: Only promote the Challenger if it reduces MSE on the feedback set
        chall_mean_mse = np.mean(get_per_sequence_losses(champion, fb_loader, DEVICE))
        if chall_mean_mse < champ_mean_mse:
            print(f"[SUCCESS] Challenger MSE: {chall_mean_mse:.6f}. Promoting weights.")
            torch.save(champion.state_dict(), MODEL_WEIGHTS)
        else:
            print("[!] Challenger failed to improve over Champion. Pass discarded.")

    else:
        # --- PHASE: FULL TRAINING ---
        # Establishes the initial behavioral baseline of the system
        print("[*] Generating and isolating system logs...")
        raw_logs = generate_securiteai_dataset()

        # Strategy: Poisoned Normalization. We isolate Attack data to prevent
        # the scaler from learning attack-state ranges.
        norm_df = raw_logs[raw_logs["Component"] != "auth-service"].copy()
        anom_df = raw_logs[raw_logs["Component"] == "auth-service"].copy()

        clean_norm, clean_anom = clean_linux_logs(norm_df), clean_linux_logs(anom_df)

        # Fit scaler ONLY on normal logs
        norm_windows = feature_engineering_pipeline(clean_norm, scaler_path=SCALER_PATH)
        s_params = np.load(SCALER_PATH)

        # Apply the 'Normal' scaler to anomaly data to maximize reconstruction failure
        anom_windows = feature_engineering_pipeline(
            clean_anom, scaler_params=(s_params[0], s_params[1])
        )

        # IID Shuffling: Ensures the model learns the system heartbeat, not just a timeline
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

        print(f"[*] Training model for 100 epochs on {DEVICE}...")
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), x)
                loss.backward()
                optimizer.step()

        # Thresholding: Define 'Normal' as the 99.5th percentile of training MSE
        train_losses = get_per_sequence_losses(model, train_loader, DEVICE)
        threshold = np.percentile(train_losses, 99.5)

        norm_test_losses = get_per_sequence_losses(model, test_norm_loader, DEVICE)
        anom_test_losses = get_per_sequence_losses(model, test_anom_loader, DEVICE)

        # Performance Reporting
        print("\n" + "=" * 45 + "\n SECURITEAI: FINAL PERFORMANCE REPORT\n" + "=" * 45)
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

        # Artifact Persistence
        torch.save(model.state_dict(), MODEL_WEIGHTS)
        np.save(THRESHOLD_PATH, threshold)
        np.save(LOSS_METRICS_PATH, np.array([norm_test_losses]))

        # --- PHASE: SKYSCRAPER VISUALIZATION ---
        # Generates a log-scale plot showing the mathematical gap between states
        print(f"[*] Generating Skyscraper Plot: {VISUAL_PATH}")
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
            label=f"Anomaly Threshold ({threshold:.4f})",
        )
        plt.yscale("log")  # Log scale is required to visualize the massive SNR gap
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.text(
            len(norm_windows) // 2,
            threshold * 2,
            "NORMAL STATE",
            color="green",
            fontweight="bold",
            ha="center",
        )
        plt.text(
            len(norm_windows) + (len(anom_windows) // 2),
            threshold * 50,
            "ATTACK BURST",
            color="red",
            fontweight="bold",
            ha="center",
        )

        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        plt.savefig(VISUAL_PATH, dpi=300)


if __name__ == "__main__":
    main()
