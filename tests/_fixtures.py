"""Tiny synthetic HDF5 files and stand-ins shared by the tests."""

from __future__ import annotations

import h5py
import numpy as np

N_WAVE = 8
N_FRAMES = 4

# Label columns of the synthetic files (any extra columns are simply ignored).
LABEL_COLUMNS = (
    "flat_index", "series_index", "frame_index",
    "pressure", "temperature", "humidity", "airmass", "pwv",
)

AIRMASS = (1.0, 1.5, 1.75, 2.0)           # [1, 2] -> 0, .5, .75, 1


class _FakeStellarPool:
    """Stand-in for ``StellarPool``: only ``pool[i] -> (spectrum, meta)`` is used."""

    def __init__(self, n_wavelength: int = N_WAVE) -> None:
        self.spectrum = np.linspace(1.0, 2.0, n_wavelength).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, None]:
        return self.spectrum, None

    def __len__(self) -> int:
        return 1


def _write_night_file(path, n_nights: int = 2) -> None:
    """Night-major HDF5: flux (nights, N_FRAMES, N_WAVE), labels (nights, N_FRAMES, P)."""
    columns = list(LABEL_COLUMNS)
    labels = np.zeros((n_nights, N_FRAMES, len(columns)), np.float32)
    flux = np.zeros((n_nights, N_FRAMES, N_WAVE), np.float32)
    for night in range(n_nights):
        labels[night, :, columns.index("series_index")] = night
        labels[night, :, columns.index("frame_index")] = np.arange(N_FRAMES)
        labels[night, :, columns.index("pressure")] = 725.0
        labels[night, :, columns.index("temperature")] = 280.0
        labels[night, :, columns.index("humidity")] = 10.0
        labels[night, :, columns.index("airmass")] = AIRMASS if night == 0 else 1.25
        labels[night, :, columns.index("pwv")] = 5.0 + night
        for k in range(N_FRAMES):
            flux[night, k] = 0.5 + 0.1 * night + 0.01 * k      # identifiable per frame
    with h5py.File(path, "w") as f:
        f.create_dataset("flux", data=flux)
        f.create_dataset("wavelength", data=np.linspace(2000.0, 2100.0, N_WAVE))
        f.create_dataset("labels", data=labels).attrs["columns"] = np.array(columns, dtype="S")
