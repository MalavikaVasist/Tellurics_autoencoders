"""Temporal Perceiver: compress the exposure axis into latent tokens."""

import torch
import torch.nn as nn

from tellurics.models.temporal.attention import _CrossAttentionBlock

__all__ = ["TemporalPerceiver"]


class TemporalPerceiver(nn.Module):
    """Compress ``Z (B, T, in)`` into ``L (B, Q, d)`` with ``Q`` learned queries.

    Uses a single multi-head cross-attention block where:

    * queries are the ``Q`` learned latent vectors (broadcast over the batch),
    * keys and values are the ``T`` per-exposure codes ``Z``.

    The ``Q`` latent tokens are learned parameters (this is *not* mean
    pooling over the time axis).  ``input_dim`` lets the keys/values be wider
    than the latent ``dim`` (e.g. ``64`` spectral + ``16`` metadata = ``80``):
    a learned ``Linear(input_dim -> dim)`` projects them down right at the
    input, so the latent space and everything downstream stay at ``dim``.
    """

    def __init__(
        self,
        n_queries: int,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.dim = dim
        self.input_dim = dim if input_dim is None else input_dim
        # Project wide input tokens (spectral + metadata) down to ``dim``.
        self.input_proj = (
            None if self.input_dim == dim else nn.Linear(self.input_dim, dim)
        )
        self.queries = nn.Parameter(
            torch.randn(n_queries, dim) * (dim ** -0.5)
        )
        self.block = _CrossAttentionBlock(dim, num_heads, dropout, ff_mult)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress the temporal axis.

        Args:
            z: (B, T, input_dim) per-exposure codes (spectral + optional
                metadata), projected to ``(B, T, dim)`` when ``input_dim``
                differs from ``dim``.

        Returns:
            L (B, Q, dim) latent tokens and (B, h, Q, T) attention.
        """
        batch, t, width = z.shape
        assert width == self.input_dim, (
            f"expected input_dim={self.input_dim}, got {width}"
        )
        if self.input_proj is not None:
            z = self.input_proj(z)                         # (B, T, dim)
        # Queries shared across each night (broadcast over the batch).
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)   # (B, Q, dim)
        l, attn = self.block(query=q, key=z, value=z)     # (B,Q,dim),(B,h,Q,T)
        return l, attn
