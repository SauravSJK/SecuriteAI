import os
from turtle import pd
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# SecuriteAI core component imports
from src.models.autoencoder import Autoencoder
from src.processing.clean_log import clean_linux_logs
from src.processing.feat_eng import feature_engineering_pipeline
from src.utils.generate_data import generate_securiteai_dataset

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
# 1. CONFIGURATION & HYPERPARAMETERS
# =================================================================
WEIGHTS_DIR = "artifacts/weights"
PARAMETERS_DIR = "artifacts/parameters"
VISUALIZATION_DIR = "visualizations"

MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(PARAMETERS_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(PARAMETERS_DIR, "scaler_params.npy")
LOSS_METRICS_PATH = os.path.join(PARAMETERS_DIR, "loss_metrics.npy")
VISUAL_PATH = os.path.join(VISUALIZATION_DIR, "securiteai_visual_report.png")

# Architecture and Training Config
N_EPOCHS = 100
BATCH_SIZE = 64
INPUT_DIM = (
    9 + 384
)  # 8 cyclical features + 1 normalized Event ID + 384 embedding dimensions
HIDDEN_DIM = 128
WINDOW_SIZE = 20
LEARNING_RATE = 1e-3


# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def get_per_sequence_losses(
    model: Autoencoder, loader: DataLoader, device: torch.device
):
    """
    Computes per-sequence MSE without reduction for statistical distribution analysis.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            # Loss calculated over the 180-dimensional flattened window (20 x 9)
            loss = criterion(model(x), x).mean(dim=(1, 2))
            losses.extend(loss.cpu().numpy())
    return np.array(losses)


# =================================================================
# 3. THE INTEGRATED PIPELINE
# =================================================================
def main():
    # Device Selection
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- PHASE 1: DATA PREPARATION & ISOLATION ---
    print("[*] Generating and isolating system logs...")
    raw_logs = generate_securiteai_dataset()

    # Split Normal from Anomaly BEFORE feature engineering to prevent leakage
    norm_df = raw_logs[raw_logs["Component"] != "auth-service"].copy()
    anom_df = raw_logs[raw_logs["Component"] == "auth-service"].copy()

    clean_norm = clean_linux_logs(norm_df)
    clean_anom = clean_linux_logs(anom_df)

    # Fit scaler ONLY on normal data (Poisoned Normalization)
    norm_windows = feature_engineering_pipeline(clean_norm, scaler_path=SCALER_PATH)
    s_params = np.load(SCALER_PATH)

    # Apply baseline scaler to attack data to induce high reconstruction error
    anom_windows = feature_engineering_pipeline(
        clean_anom, scaler_params=(s_params[0], s_params[1])
    )

    # IID Shuffling for robust temporal learning
    indices = list(range(len(norm_windows)))
    random.seed(42)
    random.shuffle(indices)

    train_idx, test_idx = indices[:8000], indices[8000:]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(norm_windows[train_idx]).float()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_norm_loader = DataLoader(
        TensorDataset(torch.tensor(norm_windows[test_idx]).float()),
        batch_size=BATCH_SIZE,
    )
    test_anom_loader = DataLoader(
        TensorDataset(torch.tensor(anom_windows).float()), batch_size=BATCH_SIZE
    )

    # --- PHASE 2: MODEL TRAINING ---
    model = Autoencoder(INPUT_DIM, HIDDEN_DIM, WINDOW_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"[*] Training model for {N_EPOCHS} epochs on {DEVICE}...")
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1}/{N_EPOCHS} | Avg Loss: {epoch_loss / len(train_loader):.6f}"
            )

    # --- PHASE 3: EVALUATION & THRESHOLDING ---
    print("[*] Calculating anomaly thresholds and metrics...")
    train_losses = get_per_sequence_losses(model, train_loader, DEVICE)
    # Threshold at 99.5th percentile of normal reconstruction error
    threshold = np.percentile(train_losses, 99.5)

    norm_test_losses = get_per_sequence_losses(model, test_norm_loader, DEVICE)
    anom_test_losses = get_per_sequence_losses(model, test_anom_loader, DEVICE)

    # Metrics Summary
    fpr = (np.sum(norm_test_losses > threshold) / len(norm_test_losses)) * 100
    recall = (np.sum(anom_test_losses > threshold) / len(anom_test_losses)) * 100
    snr = np.mean(anom_test_losses) / np.mean(norm_test_losses)

    print("\n" + "=" * 45 + "\n SECURITEAI: MODEL ARTIFACT REPORT\n" + "=" * 45)
    print(f"False Positive Rate:      {fpr:.2f}%")
    print(f"Detection Rate (Recall):  {recall:.2f}%")
    print(f"Signal-to-Noise Ratio:    {snr:.2f}x")
    print("=" * 45)

    # PERSISTENCE
    torch.save(model.state_dict(), MODEL_WEIGHTS)
    np.save(THRESHOLD_PATH, threshold)
    np.save(LOSS_METRICS_PATH, np.array([norm_test_losses]))

    # --- PHASE 4: VISUALIZATION (THE SKYSCRAPER) ---
    print(f"[*] Generating visual report to {VISUAL_PATH}...")
    all_windows = np.concatenate([norm_windows, anom_windows])
    input_t = torch.tensor(all_windows).float().to(DEVICE)

    model.eval()
    with torch.no_grad():
        reconstruction = model(input_t)
        # Use log scale for the Y-axis due to the massive reconstruction failure gap
        mse_scores = (
            torch.mean((reconstruction - input_t) ** 2, dim=(1, 2)).cpu().numpy()
        )

    plt.figure(figsize=(14, 7))
    plt.plot(mse_scores, color="#1f77b4", alpha=0.7, label="Reconstruction Error (MSE)")
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Anomaly Threshold ({threshold:.4f})",
    )

    plt.yscale("log")
    plt.title(
        "SecuriteAI: Reconstruction Error Distribution", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Log Window Index", fontsize=12)
    plt.ylabel("MSE Loss (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Section Annotations
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

    plt.legend(loc="upper left")

    # Create visualizations dir if missing
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    plt.savefig(VISUAL_PATH, dpi=300)
    print(f"[SUCCESS] Skyscraper plot saved as: {VISUAL_PATH}")


if __name__ == "__main__":
    main()
