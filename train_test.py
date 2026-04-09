import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from autoencoder import Autoencoder
from clean_log import clean_linux_logs
from feat_eng import feature_engineering_pipeline
from generate_data import generate_securiteai_dataset

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Model Artifacts
MODEL_DIR = "models"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, "securiteai_model.pth")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.npy")

# Hyperparameters
INPUT_DIM = 9  # 9 engineered features per log entry
HIDDEN_DIM = 64
WINDOW_SIZE = 20
BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 0.001


def initialize_environment(directory: str) -> None:
    """Deletes and recreates the model artifact directory for a clean run."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"[*] Environment initialized: '{directory}/' folder ready.")


def get_per_sequence_losses(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Calculates reconstruction error for every individual sequence in the loader.
    Used for thresholding and accuracy evaluation.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            loss = criterion(model(x), x).mean(dim=(1, 2))
            all_losses.extend(loss.cpu().numpy())
    return np.array(all_losses)


def main():
    # 0. Prep Environment
    initialize_environment(MODEL_DIR)
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 1. DATA GENERATION & PIPELINE
    # Step 1: Generate Synthetic Logs (10k Normal, 1k Anomaly)
    print("[*] Launching synthetic log generation...")
    logs = generate_securiteai_dataset()

    # Step 2: Clean raw data
    cleaned_df = clean_linux_logs(logs)

    # Step 3: Feature Engineering & Windowing
    print("[*] Running feature engineering pipeline...")
    # This also saves the scaler_params to SCALER_PATH for inference consistency
    data = feature_engineering_pipeline(
        cleaned_df, window_size=WINDOW_SIZE, scaler_path=SCALER_PATH
    )
    tensor_data = torch.tensor(data).float()

    # Step 4: Precise Slicing for Accuracy Measurement
    # Based on 10,000 Normal followed by 1,000 Anomaly:
    # Windows 0 to 9,980 are PURE NORMAL
    # Windows 9,981 to 10,980 contain ANOMALIES

    train_indices = range(0, 8000)  # Bulk Normal for training
    norm_test_indices = range(8981, 9981)  # Unseen Normal for FPR check
    anomaly_test_indices = range(9981, len(data))  # Unseen Anomalies for TPR check

    train_loader = DataLoader(
        Subset(TensorDataset(tensor_data), train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    norm_test_loader = DataLoader(
        Subset(TensorDataset(tensor_data), norm_test_indices), batch_size=BATCH_SIZE
    )
    anomaly_test_loader = DataLoader(
        Subset(TensorDataset(tensor_data), anomaly_test_indices), batch_size=BATCH_SIZE
    )

    # 2. MODEL TRAINING
    model = Autoencoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, window_size=WINDOW_SIZE
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"[*] Training SecuriteAI on {DEVICE} for {N_EPOCHS} epochs...")
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

        if (epoch + 1) % (N_EPOCHS // 10) == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"[*] Epoch {epoch + 1}/{N_EPOCHS} | Train Loss: {avg_loss:.6f}")

    # 3. THRESHOLDING (Statistical Baseline)
    print("[*] Calculating statistical anomaly threshold (99.5th percentile)...")
    train_losses = get_per_sequence_losses(model, train_loader, DEVICE)
    threshold = np.percentile(train_losses, 99.5)

    # 4. ACCURACY EVALUATION
    print("[*] Running accuracy validation on unseen Normal and Anomaly sets...")

    # Evaluate on Normal logs (Target: 0% anomalies flagged)
    norm_losses = get_per_sequence_losses(model, norm_test_loader, DEVICE)
    false_positives = np.sum(norm_losses > threshold)
    fpr = (false_positives / len(norm_losses)) * 100

    # Evaluate on Anomaly logs (Target: 100% anomalies flagged)
    anomaly_losses = get_per_sequence_losses(model, anomaly_test_loader, DEVICE)
    true_positives = np.sum(anomaly_losses > threshold)
    tpr = (true_positives / len(anomaly_losses)) * 100

    # 5. FINAL REPORTING
    print("\n" + "=" * 45)
    print(" SECURITEAI: COMPREHENSIVE ACCURACY REPORT")
    print("=" * 45)
    print(f"Anomaly Threshold:        {threshold:.6f}")
    print(f"Normal Test Samples:      {len(norm_losses)}")
    print(f"Anomaly Test Samples:     {len(anomaly_losses)}")
    print("-" * 45)
    print(f"False Positive Rate:      {fpr:.2f}% (Target: < 1%)")
    print(f"Detection Rate (Recall):  {tpr:.2f}% (Target: ~100%)")
    print(
        f"Signal-to-Noise Ratio:    {np.mean(anomaly_losses) / np.mean(norm_losses):.2f}x"
    )
    print("=" * 45)

    # Save artifacts
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    np.save(THRESHOLD_PATH, threshold)
    print(f"[SUCCESS] SecuriteAI artifacts saved to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
