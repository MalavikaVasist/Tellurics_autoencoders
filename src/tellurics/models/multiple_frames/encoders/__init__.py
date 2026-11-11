"""Per-exposure encoders shared by the whole-night telluric models.

Each encoder maps a per-exposure input (a spectrum, per-exposure metadata
scalars, or a scalar exposure time) to a small conditioning code:

* :class:`SpectralEncoder` - shared 1D CNN: spectrum ``(..., N) -> (..., d)``
* :class:`StellarEncoder`  - independent 1D CNN for the night-constant
  stellar spectrum ``(B, N) -> (B, d)``
* :class:`MetadataEncoder` - shared MLP: metadata scalars ``(..., P) -> (..., C)``
* :class:`TimeEncoder`     - shared MLP: scalar exposure time ``(B, T) -> (B, T, k)``
"""

from tellurics.models.metadata import MetadataEncoder  # shared (models/metadata.py)
from tellurics.models.multiple_frames.encoders.spectral import SpectralEncoder
from tellurics.models.multiple_frames.encoders.stellar import StellarEncoder
from tellurics.models.multiple_frames.encoders.time import TimeEncoder

__all__ = [
    "SpectralEncoder",
    "StellarEncoder",
    "MetadataEncoder",
    "TimeEncoder",
]
