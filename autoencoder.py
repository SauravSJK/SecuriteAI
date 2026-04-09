import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Compresses a temporal sequence of logs into a single latent vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        LSTM-based encoder that processes the input sequence and outputs a fixed-size latent vector.
        Args:
            input_dim: Number of features per log entry (e.g., 5 engineered features).
            hidden_dim: Size of the latent vector (bottleneck).
        """
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input sequence and returns the latent vector.
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim).
        Returns:
            z: Latent vector of shape (batch_size, hidden_dim).
        """
        # We only care about the final hidden state (h_n)
        _, (h_n, _) = self.encoder(x)
        # Take the top layer hidden state
        z = self.fc(h_n[-1])
        return z


class Decoder(nn.Module):
    """
    Reconstructs the original sequence from the latent vector.
    """

    def __init__(self, hidden_dim: int, output_dim: int, window_size: int):
        """
        LSTM-based decoder that takes the latent vector and reconstructs the input sequence.
        Args:
            hidden_dim: Size of the latent vector (bottleneck).
            output_dim: Number of features to reconstruct per log entry (should match input_dim).
            window_size: Length of the sequence to reconstruct.
        """
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input sequence from the latent vector.
        Args:
            z: Latent vector of shape (batch_size, hidden_dim).
        Returns:
            x_recon: Reconstructed sequence of shape (batch_size, window_size, output_dim).
        """
        # RepeatVector mechanism: unfold the bottleneck for every time step
        z_expanded = z.unsqueeze(1).expand(-1, self.window_size, -1)
        output, _ = self.decoder(z_expanded)
        return self.fc(output)


class Autoencoder(nn.Module):
    """
    Full LSTM-Autoencoder for reconstruction-based anomaly detection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, window_size: int):
        """
        Initializes the Autoencoder by creating an Encoder and Decoder instance.
        Args:
            input_dim: Number of features per log entry (e.g., 5 engineered features).
            hidden_dim: Size of the latent vector (bottleneck).
            window_size: Length of the sequence to reconstruct.
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input sequence and then decodes it to reconstruct the original sequence.
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim).
        Returns:
            x_recon: Reconstructed sequence of shape (batch_size, window_size, input_dim).
        """
        z = self.encoder(x)
        return self.decoder(z)
