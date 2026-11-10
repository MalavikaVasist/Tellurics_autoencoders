"""The wavelength grid the transmission / stellar spectra live on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Wavegrid:
    """The wavelength grid the transmission / stellar spectra live on.

    Attributes:
        wavelength: (N,) float64 grid in ``unit``.
        unit: physical unit of the grid (default 'nm').
    """

    wavelength: np.ndarray
    unit: str = "nm"

    def __post_init__(self) -> None:
        if self.wavelength.ndim != 1:
            raise ValueError("wavelength must be a 1-D array")
        # Sort-on-init guarantees monotonic interpolation afterwards.
        wl = np.asarray(self.wavelength, np.float64)
        if np.any(np.diff(wl) <= 0):
            order = np.argsort(wl)
            object.__setattr__(self, "wavelength", wl[order])
        else:
            object.__setattr__(self, "wavelength", wl)

    @property
    def n_wavelength(self) -> int:
        return self.wavelength.size

    @property
    def wavelength_min(self) -> float:
        return float(self.wavelength.min())

    @property
    def wavelength_max(self) -> float:
        return float(self.wavelength.max())

    def __len__(self) -> int:
        return self.n_wavelength

    def __getitem__(self, idx):
        return self.wavelength[idx]

    def as_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.wavelength)
