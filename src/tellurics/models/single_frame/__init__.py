"""Single-frame telluric autoencoders: observed (B, 1, N) -> telluric (B, 1, N).

Importing this package registers the three single-frame architectures:

* ``cnn_autoencoder``         -> :class:`CNNAutoencoder`
* ``transformer_autoencoder`` -> :class:`TransformerAutoencoder`
* ``specpt``                  -> :class:`SpecPT` (SpecPT adapted to full resolution)
"""

from tellurics.models.single_frame.base import SingleFrameTelluricModel
from tellurics.models.single_frame.cnn_autoencoder import CNNAutoencoder
from tellurics.models.single_frame.specpt import SpecPT
from tellurics.models.single_frame.transformer_autoencoder import TransformerAutoencoder

__all__ = [
    "SingleFrameTelluricModel",
    "CNNAutoencoder",
    "TransformerAutoencoder",
    "SpecPT",
]
