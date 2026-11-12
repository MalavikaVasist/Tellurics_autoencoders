"""``data.metadata_bounds`` used to build configs in the tests."""

from __future__ import annotations

# Default metadata columns: airmass, PWV.
BOUNDS: dict[str, tuple[float, float]] = {
    "airmass": (1.0, 2.0),
    "PWV": (0.0, 25.0),
}
