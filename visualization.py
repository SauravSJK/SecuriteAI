import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from clean_log import clean_linux_logs
from feat_eng import feature_engineering_pipeline
from generate_data import generate_securiteai_dataset

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_DIR = "models"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.npy")

INPUT_DIM = 9  # 8 cyclical features + 1 normalized Event ID
HIDDEN_DIM = 64
WINDOW_SIZE = 20


def generate_visual_report():
    """
    Generates a 'Skyscraper' plot visualizing reconstruction error
    across the entire log history.
    """
    print("[*] Preparing visualization data...")

    # Load raw data and clean
    raw_logs = generate_securiteai_dataset()

    # Separate to apply the specific normalization logic used in training
    norm_df = raw_logs[raw_logs["Component"] != "auth-service"].copy()
    anom_df = raw_logs[raw_logs["Component"] == "auth-service"].copy()

    clean_norm = clean_linux_logs(norm_df)
    clean_anom = clean_linux_logs(anom_df)

    # Load artifacts
    s_params = np.load(SCALER_PATH)
    threshold = np.load(THRESHOLD_PATH)

    # Engineer windows
    norm_windows = feature_engineering_pipeline(
        clean_norm, scaler_params=(s_params[0], s_params[1])
    )
    anom_windows = feature_engineering_pipeline(
        clean_anom, scaler_params=(s_params[0], s_params[1])
    )

    # Combine for the full timeline visualization
    all_windows = np.concatenate([norm_windows, anom_windows])

    # 2. MODEL INFERENCE
    device = torch.device("cpu")
    model = Autoencoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, window_size=WINDOW_SIZE
    ).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    input_t = torch.tensor(all_windows).float()
    print(f"[*] Calculating error for {len(all_windows)} windows...")

    with torch.no_grad():
        reconstruction = model(input_t)
        mse_scores = torch.mean((reconstruction - input_t) ** 2, dim=(1, 2)).numpy()

    # 3. PLOTTING THE "SKYSCRAPER"
    plt.figure(figsize=(14, 7))

    # Plot the reconstruction errors
    plt.plot(mse_scores, color="#1f77b4", alpha=0.7, label="Reconstruction Error (MSE)")

    # Plot the threshold line
    plt.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Anomaly Threshold ({threshold:.4f})",
    )

    # Aesthetic adjustments
    plt.yscale("log")  # Log scale is essential due to the 95,000x gap
    plt.title(
        "SecuriteAI: Reconstruction Error Over Time", fontsize=16, fontweight="bold"
    )
    plt.xlabel("Log Window Index", fontsize=12)
    plt.ylabel("Loss (MSE) - Log Scale", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Annotate the sections
    plt.text(
        len(norm_windows) // 2,
        threshold * 2,
        "NORMAL SYSTEM STATE",
        color="green",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    plt.text(
        len(norm_windows) + (len(anom_windows) // 2),
        threshold * 50,
        "ATTACK BURST",
        color="red",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    plt.legend(loc="upper left")

    # Save the result
    plot_filename = "securiteai_visual_report.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"[SUCCESS] Skyscraper plot saved as: {plot_filename}")


if __name__ == "__main__":
    generate_visual_report()
