"""Single-observation dataset: one sample = one exposure of one night.

Used by the per-observation models (``cnn_autoencoder``,
``transformer_autoencoder``, ``specpt``): one observation in, one telluric
spectrum out. The exposures of all nights of a split are pooled. It reads the same night-major HDF5 as
:class:`~tellurics.data.datasets.timeseries.TelluricTimeseriesDataset` and
keeps its input/target/metadata handling (column checks, metadata scaling,
star-per-night, noise), but indexes
``(night, frame)`` pairs, so the train/val/test split stays *night-level*: no
exposure of a validation or test night is ever seen in training.

Two HDF5 layouts are supported:

* **flat** (recommended for single frames): ``flux (n_obs, N)`` and
  ``labels (n_obs, P)``, one row per observation, with a ``series_index``
  label column saying which night (telluric series) a row belongs to. Nights
  may have *different* numbers of frames -- no padding, no NaNs.
* **night-major**: ``flux (n_nights, T, N)``, every night with the
  same ``T`` frames.

(``flux`` is the default dataset name of the telluric spectra; set by
``data.target_key``.)

Input: with ``data.observed_key: null`` the observed flux is built as
``flux * S`` with the night's fixed star; with ``observed_key`` set it is
read from that dataset and no stellar spectra are used (samples then carry no
``stellar`` / ``star_index``). Simulated inputs can get Gaussian noise: each
observation draws a level ~ N(noise_level_mean, noise_level_std^2) and gets
``level * observed * N(0, 1)`` per pixel. With ``repeats = R`` the dataset is
served R times per epoch, each pass with a new noise realization
(augmentation). The target is the telluric ``flux``; real observations
(``has_target=False``) have none and carry no ``telluric``.

A sample containing a non-finite value raises instead of reaching the network.

No time and no telluric parameters are read: the single-frame models use
neither (only the whole-night model needs exposure times).

A batch is a set of observations (no frame axis)::

    observed (B, N)   stellar (B, N)   telluric (B, N)   wavelength (B, N)
    theta.metadata (B, M)

and the same training module / loss works unchanged (MSE over (B, N)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tellurics.data.datasets.timeseries import TelluricTimeseriesDataset
from tellurics.data.scaling import ParameterScaler
from tellurics.data.stellar import StellarPool

__all__ = ["TelluricFrameDataset"]


class TelluricFrameDataset(TelluricTimeseriesDataset):
    """Single-observation samples ``(N,)`` drawn from whole nights.

    Args:
        night_ids: Nights of this split (frames of other nights never appear).
        h5_path: Flat or night-major HDF5 (see the module docstring).
        pool: Stellar pool.
        star_assignment: (n_nights,) star index per night; every frame of a
            night uses its night's star.
        n_frames: Night-major layout: exposures per night ``T``.
        night_rows: Flat layout: ``{night: row indices}`` -- the rows of the
            flat arrays that belong to each night (any number per night).
        scaler: Optional ``[0, 1]`` metadata scaler (metadata group only).

    Returns the dict of :class:`TelluricTimeseriesDataset` without the frame
    axis, plus ``frame_id`` (index of the exposure within its night).
    """

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool,
        star_assignment: np.ndarray,
        n_frames: int | None = None,
        scaler: ParameterScaler | None = None,
        metadata_columns: tuple[str, ...] | list[str] | None = None,
        night_rows: dict[int, np.ndarray] | None = None,
        observed_key: str | None = None,
        noise_level_mean: float = 0.0,
        noise_level_std: float = 0.0,
        noise_seed: int | None = None,
        has_target: bool = True,
        target_key: str = "flux",
        repeats: int = 1,
    ) -> None:
        super().__init__(
            night_ids, h5_path, pool, star_assignment, scaler,
            # single frames: metadata only -- no time, no telluric parameters
            metadata_columns=metadata_columns, target_columns=(),
            time_column=None, observed_key=observed_key,
            noise_level_mean=noise_level_mean, noise_level_std=noise_level_std,
            noise_seed=noise_seed, has_target=has_target,
            target_key=target_key, repeats=repeats,
        )
        self.flat = night_rows is not None
        if self.flat:
            # (night, row in the flat arrays, frame position within the night)
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

    def _n_base(self) -> int:
        """Distinct observations (before augmentation repeats)."""
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, object]:
        # idx runs over n_base * repeats: idx % n_base picks the observation,
        # the full idx seeds its noise (a different realization per repeat)
        night, row, frame = self._index[int(idx) % self._n_base()]
        at = np.s_[row:row + 1] if self.flat else np.s_[night, frame:frame + 1]

        # read as (1, N) / (1, P): the scaler works on 2-D groups
        with self._open() as f:
            trans, lab, stored = self._read(f, at)
        observed, s, star_idx, level = self._input(idx, night, trans, stored)  # (1, N)
        where = f"row {row}" if self.flat else f"frame {frame}"
        self._check_finite(f"night {night}, {where}", trans, observed)

        metadata = lab[:, self._metadata_idx]                          # (1, M)
        if self.scaler is not None:
            metadata = self.scaler.transform_metadata(metadata)
        theta = {"metadata": torch.from_numpy(metadata[0])}            # (M,)

        sample = {
            "observed": torch.from_numpy(observed[0]),                 # (N,)
            "wavelength": self.wavelength,                             # (N,)
            "theta": theta,
            "night_id": night,
            "frame_id": frame,
        }
        if trans is not None:
            sample["telluric"] = torch.from_numpy(trans[0])            # (N,) target
        if s is not None:
            sample["stellar"] = torch.from_numpy(s)                    # (N,)
            sample["star_index"] = star_idx
        if level is not None:
            sample["noise_level"] = torch.tensor(level[0])            # () relative sigma used
        return sample
