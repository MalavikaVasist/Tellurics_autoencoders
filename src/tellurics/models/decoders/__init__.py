"""Per-exposure decoders: map exposure codes to predictions.

* :class:`ParamDecoder` - ``(..., d) -> (..., P)`` per-exposure telluric /
  atmospheric *parameter* regression (the whole-night estimator head).
"""

from tellurics.models.decoders.parameter import ParamDecoder

__all__ = ["ParamDecoder"]
