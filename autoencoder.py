import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder module for the LSTM-Autoencoder.
    
    This module compresses a temporal sequence of log events into a single 
    latent vector (bottleneck) that captures the underlying system state.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:  
            input_dim (int): Number of input features per log entry (e.g., 11).
            hidden_dim (int): Dimensionality of the compressed latent space.
        """
        super(Encoder, self).__init__()
        # 2-layer LSTM to capture complex temporal dependencies
        self.encoder = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.2
        )
        # Linear layer to refine the final hidden state into the bottleneck vector
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_dim).
        Returns:        
            torch.Tensor: Latent representation (z) of shape (batch_size, hidden_dim).
        """
        # We only care about the final hidden state (h_n) after seeing the whole sequence
        _, (h_n, _) = self.encoder(x)
        
        # h_n shape is (num_layers, batch, hidden_dim). We take the top layer.
        z = self.fc(h_n[-1])
        return z

class Decoder(nn.Module):
    """
    Decoder module for the LSTM-Autoencoder.
    
    This module takes a single latent vector and attempts to 'unfold' it back
    into the original multi-step sequence of log events.
    """
    def __init__(self, hidden_dim: int, output_dim: int, window_size: int):
        """
        Args:
            hidden_dim (int): Dimensionality of the latent space.
            output_dim (int): Number of features to reconstruct (matches input_dim).
            window_size (int): The sequence length to reconstruct (e.g., 20).
        """
        super(Decoder, self).__init__()
        self.window_size = window_size
        
        # LSTM layer to reconstruct the temporal flow from the latent vector
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        
        # Final linear layer to map hidden states back to the original feature space
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder using the 'RepeatVector' mechanism.
        
        Args:            
            z (torch.Tensor): Latent representation of shape (batch_size, hidden_dim).
        Returns:            
            torch.Tensor: Reconstructed sequence of shape (batch_size, window_size, output_dim).
        """
        # 1. Expand the bottleneck vector to match the target sequence length
        # (batch, hidden) -> (batch, 1, hidden) -> (batch, window_size, hidden)
        z_expanded = z.unsqueeze(1).expand(-1, self.window_size, -1)
        
        # 2. Process the expanded context through the LSTM
        output, _ = self.decoder(z_expanded)
        
        # 3. Project hidden states back to original feature dimensions
        # This is applied to every time step in the sequence
        x_recon = self.fc(output)
        return x_recon

class Autoencoder(nn.Module):
    """
    Complete LSTM-Autoencoder architecture for Anomaly Detection.
    
    The model is trained to minimize reconstruction error on normal logs. 
    High reconstruction error on new data indicates a deviation from 'normal' 
    system behavior (an anomaly).
    """
    def __init__(self, input_dim: int, hidden_dim: int, window_size: int):
        """
        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The dimensionality of the latent bottleneck.
            window_size (int): The length of the sliding window (e.g., 20).
        """
        super(Autoencoder, self).__init__()
        # Encoder and Decoder output_dim should match input_dim for reconstruction
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Orchestrates the full compression and reconstruction pipeline.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_dim).
        Returns:
            torch.Tensor: Reconstructed sequence of shape (batch_size, window_size, input_dim).
        """
        # Compression
        z = self.encoder(x)
        
        # Reconstruction
        x_recon = self.decoder(z)
        
        return x_recon