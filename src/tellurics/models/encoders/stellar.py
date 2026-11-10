"""Independent encoder for the night-constant stellar spectrum ``S``."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.models.encoders.spectral import _conv_stack

__all__ = ["StellarEncoder"]


class StellarEncoder(nn.Module):
    """Independent 1D CNN encoding the night-constant stellar spectrum ``S``.

    This encoder has its **own** weights and never shares them with the X
    spectral encoder (a fresh ``_conv_stack`` + readout is created here).
    ``S`` is a single spectrum per night: ``(B, N) -> (B, d)``.
    """

    def __init__(
        self,
        n_wavelength: int,
        latent_dim: int,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel: int = 7,
        stride: int = 2,
        dropout: float = 0.0,
        pool_bins: int = 16,
    ) -> None:
        super().__init__()
        assert n_wavelength > 0 and latent_dim > 0 and pool_bins >= 1
        assert channels, "need at least one encoder channel"
        self.n_wavelength = n_wavelength
        self.latent_dim = latent_dim
        self.pool_bins = pool_bins
        self.stack = _conv_stack(channels, kernel, stride, dropout)
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels[-1] * pool_bins, latent_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Encode stellar spectra. ``(B, N) -> (B, d)``."""
        lead = s.shape[:-1]
        x = s.reshape(-1, self.n_wavelength).unsqueeze(1)
        h = self.stack(x)
        h = F.adaptive_avg_pool1d(h, self.pool_bins)
        code = self.readout(h)
        return code.view(*lead, self.latent_dim)
