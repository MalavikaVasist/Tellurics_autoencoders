"""Temporal (exposure-axis) attention building blocks.

Shared modules used by the whole-night telluric models to compress the
``T`` exposures of a night into ``Q`` latent tokens and to expand them back:

* :class:`MultiHeadCrossAttention` - raw scaled dot-product multi-head cross
  attention (returns the attention map too).
* :class:`_CrossAttentionBlock`   - pre-norm cross-attention + MLP residual
  block used by both the Perceiver compressor (Q << T) and the temporal
  decoder (T >> Q).
* :class:`TemporalPerceiver`      - compress ``(B, T, d) -> (B, Q, d)`` with
  ``Q`` learned latent queries.
* :class:`TemporalDecoder`        - expand ``(B, Q, d) -> (B, T, d)`` with
  ``T`` learned exposure queries.
"""

from tellurics.models.temporal.attention import (
    MultiHeadCrossAttention,
    _CrossAttentionBlock,
)
from tellurics.models.temporal.decoder import TemporalDecoder
from tellurics.models.temporal.perceiver import TemporalPerceiver

__all__ = [
    "MultiHeadCrossAttention",
    "_CrossAttentionBlock",
    "TemporalPerceiver",
    "TemporalDecoder",
]
