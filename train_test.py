import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import Autoencoder
from clean_log import clean_linux_logs
from feat_eng import feature_engineering_pipeline
from generate_data import generate_securiteai_dataset

# Configuration
MODEL_DIR = "models"
WEIGHTS = os.path.join(MODEL_DIR, "securiteai_model.pth")
THRESHOLD_FILE = os.path.join(MODEL_DIR, "anomaly_threshold.npy")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler_params.npy")
LOSS_METRICS_FILE = os.path.join(MODEL_DIR, "loss_metrics.npy")

N_EPOCHS = 100
BATCH_SIZE = 64
INPUT_DIM = 9  # 8 cyclical features + 1 normalized Event ID
HIDDEN_DIM = 64
WINDOW_SIZE = 20
LEARNING_RATE = 1e-3


def get_per_sequence_losses(
    model: Autoencoder, loader: DataLoader, device: torch.device
):
    """
    Computes the MSE loss for each sequence in the dataset without reduction.
    This allows us to analyze the distribution of reconstruction errors for both normal and anomalous data.
    Args:
        model: The trained Autoencoder model.
        loader: DataLoader for the dataset to evaluate.
        device: The device (CPU/GPU) to perform computations on.
    Returns:
        A numpy array of per-sequence losses.
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            loss = criterion(model(x), x).mean(dim=(1, 2))
            losses.extend(loss.cpu().numpy())
    return np.array(losses)


def main():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # 1. Isolation-based Pipeline
    raw_logs = generate_securiteai_dataset()

    # Split BEFORE cleaning/sorting to prevent leakage
    norm_df = raw_logs[raw_logs["Component"] != "auth-service"].copy()
    anom_df = raw_logs[raw_logs["Component"] == "auth-service"].copy()

    clean_norm = clean_linux_logs(norm_df)
    clean_anom = clean_linux_logs(anom_df)

    # Fit scaler ONLY on normal data
    norm_windows = feature_engineering_pipeline(clean_norm, scaler_path=SCALER_FILE)

    # Apply saved scaler to anomaly data
    s_params = np.load(SCALER_FILE)
    anom_windows = feature_engineering_pipeline(
        clean_anom, scaler_params=(s_params[0], s_params[1])
    )

    # 2. IID Shuffling of Normal Data
    indices = list(range(len(norm_windows)))
    random.seed(42)
    random.shuffle(indices)

    train_idx = indices[:8000]
    test_idx = indices[8000:]

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

    # 3. Training
    model = Autoencoder(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, window_size=WINDOW_SIZE
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("[*] Training...")
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % (N_EPOCHS // 10) == 0:
            print(f"Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss.item():.4f}")

    # 4. Evaluation
    train_losses = get_per_sequence_losses(model, train_loader, DEVICE)
    threshold = np.percentile(train_losses, 99.5)

    norm_test_losses = get_per_sequence_losses(model, test_norm_loader, DEVICE)
    anom_test_losses = get_per_sequence_losses(model, test_anom_loader, DEVICE)

    print("\n" + "=" * 45 + "\n SECURITEAI: FINAL REPORT\n" + "=" * 45)
    print(
        f"False Positive Rate:      {(np.sum(norm_test_losses > threshold) / len(norm_test_losses)) * 100:.2f}%"
    )
    print(
        f"Detection Rate:           {(np.sum(anom_test_losses > threshold) / len(anom_test_losses)) * 100:.2f}%"
    )
    print(
        f"Signal-to-Noise Ratio:    {np.mean(anom_test_losses) / np.mean(norm_test_losses):.2f}x"
    )
    print("=" * 45)

    torch.save(model.state_dict(), WEIGHTS)
    np.save(THRESHOLD_FILE, threshold)
    np.save(LOSS_METRICS_FILE, np.array([norm_test_losses]))


if __name__ == "__main__":
    main()
