"""Canonical ``data.param_bounds`` table used to build configs in the tests.

The bounds are no longer hard-coded in :mod:`tellurics.config.parameters`: every
experiment declares them explicitly in its manifest under ``data.param_bounds``
(see ``experiments/experiment1.yaml``). The tests assert exact scaled values, so
they build their configs from this fixed table instead of reading a manifest
file.
"""

from __future__ import annotations

from pydantic import Field

from tellurics.config.parameters import Parameters
from tellurics.config.data import DataConfig

# The production envelope (see experiments/experiment1.yaml). ``time_hours``
# keeps the original [0, 16] so the tests can assert ``time_hours / 16``.
BOUNDS: dict[str, tuple[float, float]] = {
    "time_hours": (0.0, 16.0),
    "pressure": (600.0, 850.0),
    "temperature": (233.15, 318.15),
    "humidity": (0.0, 100.0),
    "angle": (0.0, 90.0),
    "airmass": (1.0, 2.0),
    "co2": (420.0, 435.0),
    "o3": (0.018, 0.0429),
    "n2o": (0.335, 0.345),
    "co": (0.088, 0.169),
    "ch4": (1.90, 1.98),
    "o2": (205800.0, 214200.0),
    "no": (1.0e-19, 1.2e-5),
    "so2": (9.8e-5, 1.02e-4),
    "no2": (2.57e-5, 1.0e-4),
    "nh3": (9.8e-5, 1.02e-4),
    "hno3": (1.74e-5, 5.6e-4),
}


class TestDataConfig(DataConfig):
    """``DataConfig`` whose ``param_bounds`` default to :data:`BOUNDS`.

    The production :class:`DataConfig` deliberately has *no* default for
    ``param_bounds`` -- it must be declared per manifest. The scaling tests only
    exercise the transform math, so this subclass pins the canonical table
    while still allowing a full per-test override.
    """

    param_bounds: Parameters = Field(
        default_factory=lambda: Parameters(**BOUNDS)
    )
