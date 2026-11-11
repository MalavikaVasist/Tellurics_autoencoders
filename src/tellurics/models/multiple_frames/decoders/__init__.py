"""Per-exposure decoders: map exposure codes to predictions.

* :class:`TelluricDecoder` - ``(..., d) -> (..., N)`` per-exposure telluric
  *transmission spectrum* (the whole-night telluric time series head).
"""

from tellurics.models.multiple_frames.decoders.telluric import TelluricDecoder

__all__ = ["TelluricDecoder"]
