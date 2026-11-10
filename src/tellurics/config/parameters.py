"""Define and validate the physical parameter bounds.

The bounds are declared in the experiment YAML file under
``data.param_bounds``. This module defines the parameter names and their
canonical order, and checks that every bound is finite and satisfies
``min < max``.

The field order of :class:`Parameters` defines:

- ``PHYSICAL_COLUMNS``: all physical parameter names.
- ``TARGET_COLUMNS``: parameters predicted by the model.
- ``METADATA_COLUMNS``: parameters provided as metadata.
- ``TIME_COLUMN``: the time input used by the TimeEncoder.

"""

from __future__ import annotations

import math

from pydantic import BaseModel, ValidationInfo, field_validator

__all__ = [
    "Parameters",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
    "METADATA_COLUMNS",
    "TIME_COLUMN",
]

# The column consumed by the TimeEncoder (never a regression target).
TIME_COLUMN = "time_hours"

# The columns consumed by the MetadataEncoder.
METADATA_COLUMNS = ("pressure", "temperature", "humidity")


class Parameters(BaseModel):
    """Physical ``[min, max]`` bounds for all model parameters.

    Bounds are provided by the experiment configuration.
    """

    # -- atmosphere / geometry --------------------------------------------- #
    time_hours: tuple[float, float]      # hours
    pressure: tuple[float, float]        # hPa
    temperature: tuple[float, float]     # K (-40 .. +45 degC)
    humidity: tuple[float, float]        # %
    angle: tuple[float, float]           # zenith angle, deg
    airmass: tuple[float, float]         # sec(z), alt >= 30 deg
    # -- molecular abundances (ppmv) --------------------------------------- #
    co2: tuple[float, float]
    o3: tuple[float, float]
    n2o: tuple[float, float]
    co: tuple[float, float]
    ch4: tuple[float, float]
    o2: tuple[float, float]
    no: tuple[float, float]
    so2: tuple[float, float]
    no2: tuple[float, float]
    nh3: tuple[float, float]
    hno3: tuple[float, float]

    @field_validator("*", mode="after")
    @classmethod
    def _check_ordered_pair(
        cls, value: tuple[float, float], info: ValidationInfo
    ) -> tuple[float, float]:
        """Require finite bounds with ``min < max``."""
        low, high = value
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"{info.field_name} bounds must be finite, got {value}"
            )
        if low >= high:
            raise ValueError(
                f"{info.field_name} bounds must satisfy min < max, got {value}"
            )
        return value

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        """Return parameter names in their canonical order."""
        return tuple(cls.model_fields)

    def to_mapping(self) -> dict[str, tuple[float, float]]:
        """Return bounds as a ``{column: (min, max)}`` mapping."""
        return {name: tuple(getattr(self, name)) for name in self.column_names()}


# The schema derived from the bound declarations (canonical order preserved).
PHYSICAL_COLUMNS: tuple[str, ...] = Parameters.column_names()
TARGET_COLUMNS: tuple[str, ...] = tuple(
    c for c in PHYSICAL_COLUMNS if c != TIME_COLUMN 
)
