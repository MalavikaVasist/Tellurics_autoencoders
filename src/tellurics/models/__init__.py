"""Neural network architectures for telluric prediction.

Layout::

    models/
    |-- output.py           ModelOutput
    |-- activations.py      telluric output activation
    |-- metadata.py         MetadataEncoder
    +-- single_frame/       per-observation autoencoders, (B, N) -> (B, N)

Importing this package executes the ``@ModelRegistry.register`` decorators of
every architecture; ``model.architecture`` in a manifest picks one of
``cnn_autoencoder``, ``transformer_autoencoder`` or ``specpt``.
"""

from tellurics.models.activations import apply_telluric_activation, clear_sky_raw_value
from tellurics.models.metadata import MetadataEncoder
from tellurics.models.output import ModelOutput
from tellurics.models.single_frame import (
    CNNAutoencoder,
    SingleFrameTelluricModel,
    SpecPT,
    TransformerAutoencoder,
)

__all__ = [
    "ModelOutput",
    "apply_telluric_activation",
    "clear_sky_raw_value",
    "MetadataEncoder",
    "SingleFrameTelluricModel",
    "CNNAutoencoder",
    "TransformerAutoencoder",
    "SpecPT",
]
