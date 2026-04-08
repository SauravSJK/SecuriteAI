import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from clean_log import clean_linux_logs
from feat_eng import feature_engineering_pipeline
from autoencoder import Autoencoder

def calculate_losses(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Calculates the reconstruction error for every individual sequence in a dataset.
    
    Args:
        model: The trained Autoencoder model.
        loader: DataLoader containing the sequences to evaluate.
        device: The computation device (CPU/GPU/MPS).
        
    Returns:
        np.ndarray: A 1D array of Mean Squared Error (MSE) values per sequence.
    """
    model.eval()
    # We use reduction='none' to get the error for each specific sample 
    # instead of the batch average.
    criterion = nn.MSELoss(reduction='none')
    all_losses = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where DataLoader returns (data,) or (data, label)
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            # Calculate MSE: (Batch, Seq, Feat)
            loss = criterion(outputs, inputs)
            
            # Average loss across sequence and feature dimensions to get per-sequence loss
            # Shape transition: (Batch, 20, 11) -> (Batch)
            per_sequence_loss = loss.mean(dim=(1, 2))
            
            all_losses.extend(per_sequence_loss.cpu().numpy())
            
    return np.array(all_losses)

def train_autoencoder(
    model: Autoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Executes the training loop for the LSTM-Autoencoder.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"[*] Starting training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, inputs).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{num_epochs}] | "
                  f"Train Loss: {train_loss/len(train_loader):.6f} | "
                  f"Val Loss: {val_loss/len(val_loader):.6f}")

def main():
    # 1. Configuration & Hyperparameters
    input_dim = 11      # Based on 11 cyclical + EventID features
    hidden_dim = 64     # Bottleneck size
    window_size = 20    # Sliding window length
    num_epochs = 100
    batch_size = 32
    
    input_file = "data/linux_logs.csv"
    cleaned_file = "data/linux_logs_cleaned.csv"
    sequences_file = "data/data_sequences.npy"
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    # 2. Data Loading & Index Slicing
    try:
        data = np.load(sequences_file)
    except FileNotFoundError:
        clean_linux_logs(input_file, cleaned_file)
        feature_engineering_pipeline(cleaned_file, sequences_file, window_size)
        data = np.load(sequences_file)

    # Slicing logic: 
    # Training: End of file (Normal boot logs)
    # Testing: Start of file (SSH Brute Force attacks)
    train_size = 500
    val_size = 100
    test_size = 500

    train_indices = range(len(data) - train_size, len(data))
    val_indices = range(len(data) - (train_size + val_size), len(data) - train_size)
    test_indices = range(0, test_size)

    # Convert to Tensors and create Loaders
    tensor_data = torch.from_numpy(data).float()
    
    train_loader = DataLoader(Subset(TensorDataset(tensor_data), train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(TensorDataset(tensor_data), val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(TensorDataset(tensor_data), test_indices), batch_size=batch_size, shuffle=False)

    # 3. Model Initialization & Training
    model = Autoencoder(input_dim, hidden_dim, window_size)
    train_autoencoder(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

    # 4. Threshold Calculation (The Anomaly Boundary)
    print("\n[*] Calculating anomaly threshold from training distribution...")
    train_losses = calculate_losses(model, train_loader, device)
    threshold = np.percentile(train_losses, 99.5) # 99.5th percentile
    
    # 5. Attack Verification
    print("[*] Evaluating system against 'Attack' data (SSH logs)...")
    test_losses = calculate_losses(model, test_loader, device)
    
    avg_normal_loss = np.mean(train_losses)
    avg_attack_loss = np.mean(test_losses)
    anomalies_detected = np.sum(test_losses > threshold)

    # 6. Final Report
    print("\n" + "="*40)
    print(" SECURITEAI: FINAL SYSTEM REPORT")
    print("="*40)
    print(f"Statistical Threshold:   {threshold:.6f}")
    print(f"Average Normal MSE:      {avg_normal_loss:.6f}")
    print(f"Average Attack MSE:      {avg_attack_loss:.6f}")
    print(f"Signal-to-Noise Ratio:   {avg_attack_loss / avg_normal_loss:.2f}x")
    print("-" * 40)
    print(f"Anomalies Flagged:       {anomalies_detected} / {len(test_losses)}")
    print(f"Detection Rate:          {(anomalies_detected/len(test_losses))*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    main()