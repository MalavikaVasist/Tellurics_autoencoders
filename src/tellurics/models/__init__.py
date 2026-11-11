"""Neural network architectures for telluric prediction.

Layout::

    models/
    |-- output.py           ModelOutput            (shared)
    |-- activations.py      telluric output activation (shared)
    |-- metadata.py         MetadataEncoder (shared)
    |-- multiple_frames/    whole-night model, (B, T, N) -> (B, T, N)
    +-- single_frame/       single-exposure autoencoders, (B, 1, N) -> (B, 1, N)

Importing this package executes the ``@ModelRegistry.register`` decorators of
every architecture; ``model.architecture`` in a manifest picks one of:

* ``neural_telluric_predictor`` -- multiple frames (B, T, N) -> (B, T, N)
* ``cnn_autoencoder``           -- single frame    (B, 1, N) -> (B, 1, N)
* ``transformer_autoencoder``   -- single frame    (B, 1, N) -> (B, 1, N)
* ``specpt``                    -- single frame    (B, 1, N) -> (B, 1, N)
"""

from tellurics.models.activations import apply_telluric_activation, clear_sky_raw_value
from tellurics.models.metadata import MetadataEncoder
from tellurics.models.output import ModelOutput
from tellurics.models.multiple_frames import (
    FusionMLP,
    MultiHeadCrossAttention,
    NeuralTelluricPredictor,
    SpectralEncoder,
    StellarEncoder,
    TelluricDecoder,
    TemporalDecoder,
    TemporalPerceiver,
    TimeEncoder,
)
from tellurics.models.single_frame import (
    CNNAutoencoder,
    SingleFrameTelluricModel,
    SpecPT,
    TransformerAutoencoder,
)

__all__ = [
    # Shared.
    "ModelOutput",
    "apply_telluric_activation",
    "clear_sky_raw_value",
    "MetadataEncoder",
    # Multiple-frame (whole-night) model and its building blocks.
    "NeuralTelluricPredictor",
    "SpectralEncoder",
    "StellarEncoder",
    "TimeEncoder",
    "MultiHeadCrossAttention",
    "TemporalPerceiver",
    "TemporalDecoder",
    "FusionMLP",
    "TelluricDecoder",
    # Single-frame autoencoders.
    "SingleFrameTelluricModel",
    "CNNAutoencoder",
    "TransformerAutoencoder",
    "SpecPT",
]
