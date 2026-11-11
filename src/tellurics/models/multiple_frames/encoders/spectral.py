"""Shared per-exposure spectral encoder (strided 1D CNN + readout)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpectralEncoder"]


def _conv_stack(
    channels: tuple[int, ...],
    kernel: int,
    stride: int,
    dropout: float,
) -> nn.Sequential:
    """Strided 1D-conv stack: 1 ch -> channels[0] -> ... -> channels[-1].

    Each stage halves the spectral length (stride ``stride``) while widening
    the receptive field, then normalizes with GroupNorm(1, *) so behaviour is
    independent of batch size.
    """
    blocks: list[nn.Module] = []
    cin = 1
    padding = kernel // 2
    for cout in channels:
        blocks += [
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding),
            nn.GroupNorm(1, cout),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        cin = cout
    return nn.Sequential(*blocks)


class SpectralEncoder(nn.Module):
    """Shared 1D CNN compressing one spectrum ``(N,)`` to a ``d``-dim code.

    The *same instance* is applied to every exposure, i.e. it is never
    re-instantiated per frame.  It progressively downsamples the wavelength
    axis with strided convolutions (local receptive fields, no giant linear
    layer on the raw spectrum), then adaptively pools the remaining axis to a
    fixed number of bins per channel (preserving local spectral structure),
    flattens and projects to exactly ``latent_dim`` numbers per spectrum.

    Forward accepts any leading dimensions: ``(..., N) -> (..., d)``.
    """

    def __init__(
        self,
        n_wavelength: int,
        latent_dim: int,
        channels: tuple[int, ...] = (16, 32, 64, 96, 128),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress spectrum/spectra.

        Args:
            x: (..., N) raw spectrum, wavelength on the last axis.

        Returns:
            (..., d) per-sample embedding.
        """
        lead = x.shape[:-1]                                 # (B, 73, N)
        x = x.reshape(-1, self.n_wavelength).unsqueeze(1)   # (M, 1, N), M = B*T
        h = self.stack(x)                                   # (M, C, N')
        h = F.adaptive_avg_pool1d(h, self.pool_bins)        # (M, C, pool_bins)
        z = self.readout(h)                                 # (M, d)
        return z.view(*lead, self.latent_dim)
