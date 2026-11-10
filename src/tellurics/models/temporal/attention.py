"""Multi-head cross attention and its pre-norm residual block."""

import torch
import torch.nn as nn

__all__ = ["MultiHeadCrossAttention", "_CrossAttentionBlock"]


class MultiHeadCrossAttention(nn.Module):
    """Scaled dot-product multi-head cross attention.

    ``query (..., Lq, d)`` attends over ``key/value (..., Lk, d)`` and returns
    ``(..., Lq, d)`` plus the raw attention map ``(B, h, Lq, Lk)``.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}"
        )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross attention.

        Args:
            query: (B, Lq, dim)
            key:   (B, Lk, dim)
            value: (B, Lk, dim)

        Returns:
            (B, Lq, dim) output and (B, h, Lq, Lk) attention.
        """
        batch = query.shape[0]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        def _heads(x: torch.Tensor) -> torch.Tensor:
            # (B, L, dim) -> (B, h, L, head_dim)
            L = x.shape[1]
            return x.view(batch, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _heads(q), _heads(k), _heads(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, Lq, Lk)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, h, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.dim)
        out = self.out_proj(out)
        return out, attn


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP residual block.

    ``query`` attends over ``key/value`` (query and key can differ in length),
    which is the building block used both by the Perceiver compressor (Q<<T)
    and the temporal decoder (T>>Q).
    """

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, ff_mult: int = 4
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (B, Lq, dim) block output and (B, h, Lq, Lk) attention."""
        q = self.norm_q(query)
        k = self.norm_kv(key)
        v = self.norm_kv(value)
        out, attn = self.attn(q, k, v)
        x = query + self.drop1(out)
        x = x + self.drop1(self.ff(self.norm_ff(x)))
        return x, attn
