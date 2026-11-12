"""Single-observation dataset: one sample = one exposure of one night.

One observation in, one telluric spectrum out. The exposures of all nights of
a split are pooled, but the index is ``(night, frame)``, so the
train/val/test split stays *night-level*: no exposure of a validation or test
night is ever seen in training.

Two HDF5 layouts are supported:

* **flat** (recommended): ``flux (n_obs, N)`` and ``labels (n_obs, P)``, one
  row per observation, with a ``series_index`` label column saying which
  night (telluric series) a row belongs to. Nights may have *different*
  numbers of frames -- no padding, no NaNs.
* **night-major**: ``flux (n_nights, T, N)``, every night with the same
  ``T`` frames.

``flux`` is the default dataset name of the telluric spectra (the target);
it is set by ``data.target_key``. Every file also holds ``wavelength (N,)``
and ``labels`` with the column names in ``labels.attrs['columns']``.

Input:

* ``observed_key`` None -- the file holds pure tellurics: the input is
  ``flux * S`` with the night's fixed star ``S`` (the same star for every
  frame of a night);
* ``observed_key`` set -- the observed flux is read from that dataset as
  stored; no stellar spectra are used.

Simulated inputs can get Gaussian noise: each observation draws a level
``~ N(noise_level_mean, noise_level_std^2)`` (clipped at 0) and gets
``level * observed * N(0, 1)`` per pixel. With ``repeats = R`` the dataset is
served R times per epoch, each pass with a new noise realization
(augmentation). Real observations (``has_target=False``) have no target.

Only the metadata columns (e.g. airmass, PWV) are read from the labels; they
are scaled to ``[0, 1]`` with the declared bounds. Spectra are never scaled.

A sample containing a non-finite value raises instead of reaching the network.

A sample::

    observed    (N,)   model input
    telluric    (N,)   target (not for real observations)
    wavelength  (N,)
    theta       {"metadata": (M,)}
    night_id    int    night (series) position in the file
    frame_id    int    exposure within the night
    stellar     (N,)   + star_index   (observed_key None only)
    noise_level ()     relative noise sigma used (noisy samples only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tellurics.data.scaling import MinMax
from tellurics.data.stellar import StellarPool

__all__ = ["TelluricFrameDataset", "read_label_columns"]


def _decode_columns(columns) -> list[str]:
    """Decode an h5py 'columns' attribute into a list of strings."""
    return [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in columns]


def read_label_columns(h5_path: str | Path) -> list[str]:
    """Label column names stored in the file's ``labels.attrs['columns']``."""
    import h5py

    with h5py.File(Path(h5_path), "r") as f:
        return _decode_columns(f["labels"].attrs.get("columns", []))


class TelluricFrameDataset(Dataset[dict[str, object]]):
    """Single-observation samples ``(N,)`` drawn from whole nights.

    Args:
        night_ids: Nights of this split (frames of other nights never appear).
        h5_path: Flat or night-major HDF5 (see the module docstring).
        pool: Stellar pool (``observed_key`` None only).
        star_assignment: ``(n_nights,)`` star index per night; every frame of a
            night uses its night's star (``observed_key`` None only).
        n_frames: Night-major layout: exposures per night ``T``.
        night_rows: Flat layout: ``{night: row indices}`` -- the rows of the
            flat arrays that belong to each night (any number per night).
        scaler: Optional ``[0, 1]`` scaler of the metadata columns.
        metadata_columns: Label columns fed to the model, in order.
        observed_key: Dataset with the stored observed flux, or None to build
            ``flux * star``.
        target_key: Dataset with the telluric spectra (the target).
        noise_level_mean, noise_level_std: Relative noise level per
            observation, ``~ N(mean, std^2)``; mean 0 -> no noise.
        noise_seed: None -> fresh noise on every read (training); an int ->
            the same noise for a sample on every read (validation / test).
        has_target: False for real observations (no target is read).
        repeats: Serve the observations this many times (augmentation).
    """

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool | None = None,
        star_assignment: np.ndarray | None = None,
        n_frames: int | None = None,
        night_rows: dict[int, np.ndarray] | None = None,
        scaler: MinMax | None = None,
        metadata_columns: tuple[str, ...] | list[str] = ("airmass", "PWV"),
        observed_key: str | None = None,
        target_key: str = "flux",
        noise_level_mean: float = 0.0,
        noise_level_std: float = 0.0,
        noise_seed: int | None = None,
        has_target: bool = True,
        repeats: int = 1,
    ) -> None:
        super().__init__()
        if not has_target and observed_key is None:
            raise ValueError("target-free (real) data must give observed_key")
        if observed_key is None and (pool is None or star_assignment is None):
            raise ValueError(
                "pure-telluric data needs a stellar pool and a star assignment "
                "(or set observed_key for files that already hold observed flux)"
            )
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")

        self.h5_path = Path(h5_path)
        self.night_ids = list(night_ids)
        self.pool = pool
        self.star_assignment = (
            None if star_assignment is None else np.asarray(star_assignment)
        )
        self.scaler = scaler
        self.metadata_columns = tuple(metadata_columns)
        self.observed_key = observed_key
        self.target_key = target_key
        self.noise_level_mean = float(noise_level_mean)
        self.noise_level_std = float(noise_level_std)
        self.noise_seed = noise_seed
        self.has_target = has_target
        self.repeats = int(repeats)

        # Column layout from the file's stored 'columns' attribute.
        columns = read_label_columns(self.h5_path)
        missing = [c for c in self.metadata_columns if c not in columns]
        if missing:
            raise KeyError(
                f"expected label column(s) {missing} not in file columns {columns}"
            )
        self._metadata_idx = [columns.index(c) for c in self.metadata_columns]
        if scaler is not None and scaler.columns != self.metadata_columns:
            raise ValueError(
                f"scaler is built for columns {scaler.columns} but the dataset "
                f"uses {self.metadata_columns}"
            )
        with self._open() as f:
            self.wavelength = torch.from_numpy(np.asarray(f["wavelength"][:], np.float32))

        # (night, row in the flat arrays | None, frame position within the night)
        self.flat = night_rows is not None
        if self.flat:
            self._index = [
                (night, int(row), pos)
                for night in self.night_ids
                for pos, row in enumerate(night_rows[night])
            ]
        else:
            if not n_frames or n_frames <= 0:
                raise ValueError("night-major layout needs n_frames > 0")
            self._index = [
                (night, None, frame)
                for night in self.night_ids
                for frame in range(int(n_frames))
            ]

    def _open(self):
        import h5py

        return h5py.File(self.h5_path, "r")

    def _n_base(self) -> int:
        """Distinct observations (before augmentation repeats)."""
        return len(self._index)

    def __len__(self) -> int:
        return self._n_base() * self.repeats

    def _noise(self, idx: int, observed: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Add the relative Gaussian noise; return ``(observed, level | None)``."""
        if self.noise_level_mean <= 0:
            return observed, None
        rng = (
            np.random.default_rng() if self.noise_seed is None
            else np.random.default_rng([self.noise_seed, int(idx)])
        )
        level = max(0.0, self.noise_level_mean + self.noise_level_std * rng.standard_normal())
        noise = rng.standard_normal(observed.shape).astype(np.float32)
        return observed + np.float32(level) * observed * noise, level

    def __getitem__(self, idx: int) -> dict[str, object]:
        # idx runs over n_base * repeats: idx % n_base picks the observation,
        # the full idx seeds its noise (a different realization per repeat)
        night, row, frame = self._index[int(idx) % self._n_base()]
        at = row if self.flat else (night, frame)

        with self._open() as f:
            target = (
                np.asarray(f[self.target_key][at], np.float32) if self.has_target else None
            )                                                              # (N,)
            labels = np.asarray(f["labels"][at], np.float32)               # (P,)
            stored = (
                np.asarray(f[self.observed_key][at], np.float32)
                if self.observed_key is not None else None
            )                                                              # (N,)

        star = star_idx = None
        if stored is not None:
            observed = stored                                              # as stored
        else:
            star_idx = int(self.star_assignment[night])                    # the night's star
            star, _ = self.pool[star_idx]                                  # (N,)
            observed = target * star
        observed, level = self._noise(idx, observed)

        where = f"row {row}" if self.flat else f"frame {frame}"
        if not all(np.isfinite(a).all() for a in (target, observed) if a is not None):
            raise ValueError(
                f"non-finite values in night {night}, {where} of {self.h5_path}: "
                "padded frames must not be stored -- use the flat layout (one "
                "row per real observation)"
            )

        metadata = labels[self._metadata_idx][None, :]                     # (1, M)
        if self.scaler is not None:
            metadata = self.scaler.transform(metadata)

        sample = {
            "observed": torch.from_numpy(observed),                        # (N,)
            "wavelength": self.wavelength,                                 # (N,)
            "theta": {"metadata": torch.from_numpy(metadata[0])},          # (M,)
            "night_id": night,
            "frame_id": frame,
        }
        if target is not None:
            sample["telluric"] = torch.from_numpy(target)                  # (N,) target
        if star is not None:
            sample["stellar"] = torch.from_numpy(star)                     # (N,)
            sample["star_index"] = star_idx
        if level is not None:
            sample["noise_level"] = torch.tensor(level, dtype=torch.float32)  # ()
        return sample
