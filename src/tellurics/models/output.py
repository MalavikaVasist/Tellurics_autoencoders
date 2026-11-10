"""Structured model output dataclass."""

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Structured prediction output from telluric models.

    This design allows extending the model's return type without
    breaking the public API. New fields can be added as optional
    attributes without changing downstream code.

    Attributes:
        params: Predicted per-sample/scalar parameters (e.g. telluric / gas
            parameters per exposure). Shape: (B, T, P).
        latent: Latent embedding from the encoder. Shape: (B, D_latent).
        attention_weights: Attention maps if available. Shape varies by architecture.
        intermediate_features: Dictionary of intermediate feature representations.
    """

    params: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    intermediate_features: dict[str, torch.Tensor] | None = None
