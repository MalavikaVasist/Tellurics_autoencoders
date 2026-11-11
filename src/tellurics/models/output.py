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
        telluric: Predicted telluric transmission time series, one full
            spectrum per exposure. Shape: (B, T, N).
        latent: Latent embedding from the encoder. Shape: (B, D_latent).
        attention_weights: Attention maps if available. Shape varies by architecture.
        intermediate_features: Dictionary of intermediate feature representations.
    """

    telluric: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    intermediate_features: dict[str, torch.Tensor] | None = None
