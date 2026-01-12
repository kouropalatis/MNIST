import torch
from torch import nn


# --- 1. Your Original CNN Model with Input Validation ---
class MyAwesomeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Adding BatchNorm2d after Conv layers for 0-mean/1-std data
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- NEW: Defensive Input Validation ---
        if x.ndim != 4:
            raise ValueError(f"Expected input to be a 4D tensor (batch, channel, h, w), but got {x.ndim}D")

        if x.shape[1:] != (1, 28, 28):
            raise ValueError(f"Expected each sample to have shape [1, 28, 28], but got {list(x.shape[1:])}")
        # ---------------------------------------

        x = torch.relu(self.bn1(self.conv1(x)))  # BN before activation
        x = torch.max_pool2d(x, 2, 2)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


# --- 2. VAE Components ---


class Encoder(nn.Module):
    """Gaussian MLP Encoder for VAE."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        h_ = self.ReLU(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    """Gaussian MLP Decoder for VAE."""

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.ReLU(self.FC_hidden(x))
        x_hat = self.Sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    """Full VAE Model combining Encoder and Decoder."""

    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterization(self, mean, var):
        """The 'Variational' trick: sampling from a distribution."""
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
