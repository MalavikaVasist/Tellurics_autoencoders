
# simulation/telluric/conditions.py

from pathlib import Path

import h5py

from tellurics.simulation.utils.site_parameters import (
    read_condition_layout,
    load_condition_sample,
)


class ConditionSeries:
    """Read atmospheric conditions indexed by (series, frame)."""

    def __init__(self, path):
        self.path = Path(path)
        (
            self.n_series,
            self.n_frames,
            self.observatory,
        ) = read_condition_layout(self.path)

        self.n_samples = self.n_series * self.n_frames

    def load(self, flat_index):
        """Return (TelFit parameters, labels) for one flattened index."""
        if not 0 <= flat_index < self.n_samples:
            raise IndexError(
                f"flat_index must be in [0, {self.n_samples}), "
                f"got {flat_index}"
            )

        with h5py.File(self.path, "r") as hf:
            params, labels = load_condition_sample(
                hf,
                int(flat_index),
                self.n_frames,
            )

        return params, labels

    def load_range(self, start, stop):
        """Yield (flat_index, params, labels) in the requested range."""
        if not 0 <= start <= stop <= self.n_samples:
            raise ValueError("Invalid condition range.")

        with h5py.File(self.path, "r") as hf:
            for idx in range(start, stop):
                params, labels = load_condition_sample(
                    hf,
                    idx,
                    self.n_frames,
                )
                yield idx, params, labels
