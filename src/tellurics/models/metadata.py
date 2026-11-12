"""Metadata encoder, shared by the CNN and transformer autoencoders.

Maps each observation's metadata ``(B, M) -> (B, C)``, with
``M = len(data.metadata_columns)`` (e.g. airmass, PWV).
"""

import torch
import torch.nn as nn

__all__ = ["MetadataEncoder"]


class MetadataEncoder(nn.Module):
    """Encode per-exposure metadata scalars into a conditioning code.

    Maps the ``M`` scalar conditions of an observation (e.g. airmass, PWV)
    to a ``C``-dim code with a small MLP:

        metadata (B, M)  ->  code (B, C)

    The code is joined to the spectral input of the autoencoders.

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
