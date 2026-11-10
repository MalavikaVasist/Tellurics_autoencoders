"""Temporal decoder: expand latent tokens back to per-exposure codes."""

import torch
import torch.nn as nn

from tellurics.models.temporal.attention import _CrossAttentionBlock

__all__ = ["TemporalDecoder"]


class TemporalDecoder(nn.Module):
    """Expand ``latent (B, Q, d)`` back to one code per exposure ``(B, T, d)``.

    ``T`` learned temporal (exposure-index) queries cross-attend over the ``Q``
    latent tokens, i.e. the reconstruction of each exposure is a learned
    function of the whole latent set -- not a broadcast of a mean vector.
    """

    def __init__(
        self,
        n_frames: int,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.dim = dim
        # T learned temporal query / position embeddings (T, d)
        self.queries = nn.Parameter(
            torch.randn(n_frames, dim) * (dim ** -0.5)
        )
        self.block = _CrossAttentionBlock(dim, num_heads, dropout, ff_mult)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Expand latent tokens to per-exposure codes.

        Args:
            latent: (B, Q, d).

        Returns:
            (B, T, d) per-exposure spectral embeddings ``H_T``.
        """
        batch, q, d = latent.shape
        assert d == self.dim, f"expected dim={self.dim}, got {d}"
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)   # (B, T, d)
        h_t, _ = self.block(query=queries, key=latent, value=latent)  # (B,T,d)
        return h_t
