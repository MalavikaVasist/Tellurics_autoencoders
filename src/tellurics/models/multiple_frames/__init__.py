"""Multiple-frame (whole-night) telluric model: observed (B, T, N) -> telluric (B, T, N).

Importing this package registers ``neural_telluric_predictor``.

Layout::

    multiple_frames/
    |-- predictor.py   NeuralTelluricPredictor (the whole-night model)
    |-- encoders/      per-exposure spectral / stellar / metadata / time encoders
    |-- temporal/      exposure-axis attention (Perceiver compressor + decoder)
    |-- fusion.py      FusionMLP (temporal latent + stellar code)
    +-- decoders/      TelluricDecoder (per-exposure spectrum head)

Shared with the single-frame models (directly under :mod:`tellurics.models`):
:class:`~tellurics.models.output.ModelOutput` and
:mod:`tellurics.models.activations`.
"""

from tellurics.models.multiple_frames.decoders import TelluricDecoder
from tellurics.models.multiple_frames.encoders import (
    MetadataEncoder,
    SpectralEncoder,
    StellarEncoder,
    TimeEncoder,
)
from tellurics.models.multiple_frames.fusion import FusionMLP
from tellurics.models.multiple_frames.predictor import NeuralTelluricPredictor
from tellurics.models.multiple_frames.temporal import (
    MultiHeadCrossAttention,
    TemporalDecoder,
    TemporalPerceiver,
)

__all__ = [
    "NeuralTelluricPredictor",
    "SpectralEncoder",
    "StellarEncoder",
    "MetadataEncoder",
    "TimeEncoder",
    "MultiHeadCrossAttention",
    "TemporalPerceiver",
    "TemporalDecoder",
    "FusionMLP",
    "TelluricDecoder",
]
