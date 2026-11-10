"""Per-exposure encoders shared by the whole-night telluric models.

Each encoder maps a per-exposure input (a spectrum, per-exposure metadata
scalars, or a scalar exposure time) to a small conditioning code:

* :class:`SpectralEncoder` - shared 1D CNN: spectrum ``(..., N) -> (..., d)``
* :class:`StellarEncoder`  - independent 1D CNN for the night-constant
  stellar spectrum ``(B, N) -> (B, d)``
* :class:`MetadataEncoder` - shared MLP: metadata scalars ``(..., P) -> (..., C)``
* :class:`TimeEncoder`     - shared MLP: scalar exposure time ``(B, T) -> (B, T, k)``
"""

from tellurics.models.encoders.metadata import MetadataEncoder
from tellurics.models.encoders.spectral import SpectralEncoder
from tellurics.models.encoders.stellar import StellarEncoder
from tellurics.models.encoders.time import TimeEncoder

__all__ = [
    "SpectralEncoder",
    "StellarEncoder",
    "MetadataEncoder",
    "TimeEncoder",
]
