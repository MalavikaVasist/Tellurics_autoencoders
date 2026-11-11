"""Per-exposure metadata encoder (shared MLP)."""

import torch
import torch.nn as nn

__all__ = ["MetadataEncoder"]


class MetadataEncoder(nn.Module):
    """Encode per-exposure metadata scalars into a conditioning code.

    Maps the ``P`` scalar conditions of every exposure (e.g. airmass,
    humidity, seeing, ...) to a ``C``-dim code with a small MLP **shared**
    over the ``T`` exposures (the same instance is applied to each one):

        metadata (B, T, P)  ->  M_code (B, T, C)

    The code is meant to be concatenated with the per-exposure spectral code
    ``Z (B, T, d)``, yielding ``(B, T, d + C)`` tokens for the temporal
    Perceiver, so the attention can condition on per-exposure metadata.

    Forward accepts any leading dimensions: ``(..., P) -> (..., C)``.
    """

    def __init__(
        self,
        n_metadata: int,
        enc_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_metadata > 0 and enc_dim > 0
        self.n_metadata = n_metadata
        self.enc_dim = enc_dim
        hidden = hidden_dim or max(enc_dim, 2 * n_metadata)
        self.net = nn.Sequential(
            nn.Linear(n_metadata, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, enc_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """Encode metadata scalars. ``(..., P) -> (..., C)``."""
        lead = meta.shape[:-1]
        x = meta.reshape(-1, self.n_metadata)   # (M, P)
        code = self.net(x)                      # (M, C)
        return code.view(*lead, self.enc_dim)
