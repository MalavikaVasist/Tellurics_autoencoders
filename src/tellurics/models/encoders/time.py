"""Per-exposure continuous-time encoder (time_hours -> code)."""

import torch
import torch.nn as nn

__all__ = ["TimeEncoder"]


class TimeEncoder(nn.Module):
    """Encode a continuous scalar exposure time into a per-exposure code.

    Maps each scalar exposure time (e.g. ``time_hours``, shape ``(B, T)``)
    through a small MLP to a ``k``-dim code::

        time (B, T)  ->  z_time (B, T, k)

    The code is concatenated with the spectral and metadata codes before the
    temporal Perceiver, giving the attention an explicit notion of *when* each
    exposure was taken -- a continuous-time analogue of a positional encoding
    over the exposure axis.
    """

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar exposure times. ``(B, T) -> (B, T, k)``."""
        return self.net(t.unsqueeze(-1))
