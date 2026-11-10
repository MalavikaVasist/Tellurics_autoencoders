"""Neural network architectures for telluric prediction.

Importing this package executes the ``@ModelRegistry.register`` decorators of
every architecture; the whole-night ``neural_telluric_predictor`` estimator is
the one the training pipeline builds.
"""

from tellurics.models.decoders import ParamDecoder
from tellurics.models.encoders import (
    MetadataEncoder,
    SpectralEncoder,
    StellarEncoder,
    TimeEncoder,
)
from tellurics.models.fusion import FusionMLP
from tellurics.models.predictor import NeuralTelluricPredictor
from tellurics.models.output import ModelOutput
from tellurics.models.temporal import (
    MultiHeadCrossAttention,
    TemporalDecoder,
    TemporalPerceiver,
)

__all__ = [
    # Whole-night telluric parameter estimator (the trained architecture).
    "NeuralTelluricPredictor",
    "ModelOutput",
    # Per-exposure encoders (spectrum / stellar / metadata / time).
    "SpectralEncoder",
    "StellarEncoder",
    "MetadataEncoder",
    "TimeEncoder",
    # Temporal (exposure-axis) attention building blocks.
    "MultiHeadCrossAttention",
    "TemporalPerceiver",
    "TemporalDecoder",
    # Fusion / decoder heads.
    "FusionMLP",
    "ParamDecoder",
]
