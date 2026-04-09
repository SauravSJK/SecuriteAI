import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Compresses a temporal sequence of logs into a single latent vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        z = self.fc(h_n[-1])
        return z


class Decoder(nn.Module):
    """
    Unfolds the latent vector back into the original multi-step sequence.
    """

    def __init__(self, hidden_dim: int, output_dim: int, window_size: int):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(1).expand(-1, self.window_size, -1)
        output, _ = self.decoder(z_expanded)
        return self.fc(output)


class Autoencoder(nn.Module):
    """
    Full LSTM-Autoencoder for reconstruction-based anomaly detection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, window_size: int):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
