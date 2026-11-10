from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, eq=False)
class MinMax:
    """Scale each column to the range [0, 1].

    With ``clip`` enabled :meth:`transform` clamps physical values to the
    declared bounds instead of rejecting them, so the scaled result stays in
    ``[0, 1]``. This is used for the test split, where values stored as
    ``float32`` can sit marginally outside the declared envelope.
    """

    columns: tuple[str, ...]
    minimum: np.ndarray
    maximum: np.ndarray
    clip: bool = False

    def __post_init__(self):
        if len(self.columns) != len(self.minimum):
            raise ValueError("Number of columns must match number of minimums.")

        if self.minimum.shape != self.maximum.shape:
            raise ValueError("Minimum and maximum must have the same shape.")

        if np.any(self.minimum >= self.maximum):
            raise ValueError("Each minimum must be smaller than its maximum.")

    @classmethod
    def from_bounds(
        cls,
        columns: list[str] | tuple[str, ...],
        bounds: dict[str, tuple[float, float]],
        *,
        clip: bool = False,
    ):
        columns = tuple(columns)

        minimum = []
        maximum = []

        for column in columns:
            if column not in bounds:
                raise ValueError(f"Missing bounds for column: {column}")

            low, high = bounds[column]
            minimum.append(low)
            maximum.append(high)

        return cls(
            columns=columns,
            minimum=np.asarray(minimum, dtype=np.float64),
            maximum=np.asarray(maximum, dtype=np.float64),
            clip=clip,
        )

    @property
    def span(self):
        """Difference between maximum and minimum for each column."""
        return self.maximum - self.minimum

    def _check_shape(self, values):
        """Check that the last dimension matches the number of columns."""
        values = np.asarray(values)

        if values.ndim < 2:
            raise ValueError(
                f"Expected at least 2 dimensions, got shape {values.shape}."
            )

        if values.shape[-1] != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} columns, "
                f"got shape {values.shape}."
            )

        return values

    def transform(self, values):
        """Scale physical values to the range [0, 1].

        Raises:
            ValueError: If a value is outside the declared bounds and
                ``clip`` is disabled.
        """
        values = self._check_shape(values)
        # Scale in float64 and clip against the float64 bounds: a value at a
        # bound can be stored as float32 just below it (0.018 -> 0.017999999).
        physical = values.astype(np.float64, copy=False)

        if self.clip:
            physical = np.clip(physical, self.minimum, self.maximum)
        elif np.any(physical < self.minimum) or np.any(physical > self.maximum):
            raise ValueError("Values are outside the declared bounds.")

        scaled = (physical - self.minimum) / self.span

        return scaled.astype(values.dtype, copy=False)

    def inverse(self, values):
        """Convert values from [0, 1] back to physical values."""
        values = self._check_shape(values)

        physical = (values.astype(np.float64, copy=False) * self.span + self.minimum)

        return physical.astype(values.dtype, copy=False)

    def count_out_of_bounds(self, values):
        """Count values outside the allowed range."""
        values = self._check_shape(values)

        outside = (values < self.minimum) | (values > self.maximum)

        return int(outside.sum())

    def summary(self):
        """Return the min and max for each column."""
        return {
            column: [low, high]
            for column, low, high in zip(
                self.columns,
                self.minimum,
                self.maximum,
            )
        }


@dataclass(frozen=True, eq=False)
class ParameterScaler:
    """Manage scaling for metadata, parameters, and time."""

    metadata: MinMax | None = field(default=None)
    params: MinMax | None = field(default=None)
    time: MinMax | None = field(default=None)

    @classmethod
    def from_bounds(
        cls,
        bounds,
        metadata_columns=(),
        target_columns=(),
        time_column=None,
        scale_metadata=True,
        scale_params=True,
        scale_time=True,
        clip=False,
    ):
        def make_scaler(columns, enabled):
            if not enabled or not columns:
                return None

            return MinMax.from_bounds(columns, bounds, clip=clip)

        metadata = make_scaler(
            metadata_columns,
            scale_metadata,
        )

        params = make_scaler(
            target_columns,
            scale_params,
        )

        time = None
        if scale_time and time_column is not None:
            time = MinMax.from_bounds(
                (time_column,),
                bounds,
                clip=clip,
            )

        return cls(
            metadata=metadata,
            params=params,
            time=time,
        )

    @property
    def scale_metadata(self):
        return self.metadata is not None

    @property
    def scale_params(self):
        return self.params is not None

    @property
    def scale_time(self):
        return self.time is not None

    def transform_metadata(self, values):
        if self.metadata is None:
            return values

        return self.metadata.transform(values)

    def transform_params(self, values):
        if self.params is None:
            return values

        return self.params.transform(values)

    def transform_time(self, values):
        if self.time is None:
            return values

        return self.time.transform(values)

    def inverse_params(self, values):
        if self.params is None:
            return values

        return self.params.inverse(values)

    def inverse_params_torch(self, values):
        """Convert normalized PyTorch predictions back to physical values."""
        if self.params is None:
            return values

        import torch

        minimum = torch.as_tensor(
            self.params.minimum,
            dtype=values.dtype,
            device=values.device,
        )

        maximum = torch.as_tensor(
            self.params.maximum,
            dtype=values.dtype,
            device=values.device,
        )

        return values * (maximum - minimum) + minimum

    def summary(self):
        result = {}

        if self.metadata is not None:
            result["metadata"] = self.metadata.summary()

        if self.params is not None:
            result["params"] = self.params.summary()

        if self.time is not None:
            result["time"] = self.time.summary()

        return result