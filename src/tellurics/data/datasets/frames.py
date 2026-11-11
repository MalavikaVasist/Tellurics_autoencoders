"""Single-observation dataset: one sample = one exposure of one night.

Used by the per-observation models (``cnn_autoencoder``,
``transformer_autoencoder``, ``specpt``): one observation in, one telluric
spectrum out. The exposures of all nights of a split are pooled. It reads the same night-major HDF5 as
:class:`~tellurics.data.datasets.timeseries.TelluricTimeseriesDataset` and
keeps its whole contract (column checks, scaling, star-per-night), but indexes
``(night, frame)`` pairs, so the train/val/test split stays *night-level*: no
exposure of a validation or test night is ever seen in training.

A batch is a set of observations (no frame axis)::

    observed (B, N)   stellar (B, N)   telluric (B, N)   wavelength (B, N)
    theta.time (B,)   theta.metadata (B, M)   theta.output_params (B, P)

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
        h5_path: Night-major HDF5 (``transmission`` (n_nights, T, N), ``labels``).
        pool: Stellar pool.
        star_assignment: (n_nights,) star index per night; every frame of a
            night uses its night's star.
        n_frames: Exposures per night ``T`` (all of them become samples).
        scaler: Optional ``[0, 1]`` label scaler (as for the night dataset).

    Returns the dict of :class:`TelluricTimeseriesDataset` without the frame
    axis, plus ``frame_id`` (index of the exposure within its night).
    """

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool,
        star_assignment: np.ndarray,
        n_frames: int,
        scaler: ParameterScaler | None = None,
        metadata_columns: tuple[str, ...] | list[str] | None = None,
        target_columns: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__(
            night_ids, h5_path, pool, star_assignment, scaler,
            metadata_columns=metadata_columns, target_columns=target_columns,
        )
        assert n_frames > 0
        self.n_frames = int(n_frames)

    def __len__(self) -> int:
        return len(self.night_ids) * self.n_frames

    def __getitem__(self, idx: int) -> dict[str, object]:
        night_pos, frame = divmod(int(idx), self.n_frames)
        night = self.night_ids[night_pos]
        star_idx = int(self.star_assignment[night])
        s, _ = self.pool[star_idx]                                     # (N,)

        with self._open() as f:
            # read as (1, N) / (1, P): the scaler works on 2-D groups
            trans = np.asarray(f["transmission"][night, frame:frame + 1], np.float32)
            lab = np.asarray(f["labels"][night, frame:frame + 1], np.float32)

        observed = trans * s[None, :]                                  # (1, N)
        time = lab[:, [self._time_idx]]                                # (1, 1)
        metadata = lab[:, self._metadata_idx]                          # (1, M)
        output_params = lab[:, self._target_idx]                       # (1, P)

        if self.scaler is not None:
            metadata = self.scaler.transform_metadata(metadata)
            output_params = self.scaler.transform_params(output_params)
            time = self.scaler.transform_time(time)

        return {
            "observed": torch.from_numpy(observed[0]),                 # (N,)
            "stellar": torch.from_numpy(s),                            # (N,)
            "telluric": torch.from_numpy(trans[0]),                    # (N,)
            "wavelength": self.wavelength,                             # (N,)
            "theta": {
                "time": torch.tensor(time[0, 0], dtype=torch.float32), # ()
                "metadata": torch.from_numpy(metadata[0]),             # (M,)
                "output_params": torch.from_numpy(output_params[0]),   # (P,)
            },
            "night_id": night,
            "frame_id": frame,
            "star_index": star_idx,
        }
