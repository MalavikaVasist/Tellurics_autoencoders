"""Single-frame dataset: one sample = one exposure of one night.

Used by the single-frame models (``cnn_autoencoder``, ``transformer_autoencoder``,
``specpt``). It reads the same night-major HDF5 as
:class:`~tellurics.data.datasets.timeseries.TelluricTimeseriesDataset` and
keeps its whole contract (column checks, scaling, star-per-night), but indexes
``(night, frame)`` pairs, so the train/val/test split stays *night-level*: no
exposure of a validation or test night is ever seen in training.

Every tensor keeps a frame axis of length 1, so batches have exactly the
shapes of the whole-night pipeline with ``T = 1``::

    observed (B, 1, N)   stellar (B, N)   telluric (B, 1, N)
    theta.time (B, 1)    theta.metadata (B, 1, 3)   theta.output_params (B, 1, P)

and the same training module / loss works unchanged.
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
    """Single-exposure samples ``(1, N)`` drawn from whole nights.

    Args:
        night_ids: Nights of this split (frames of other nights never appear).
        h5_path: Night-major HDF5 (``transmission`` (n_nights, T, N), ``labels``).
        pool: Stellar pool.
        star_assignment: (n_nights,) star index per night; every frame of a
            night uses its night's star.
        n_frames: Exposures per night ``T`` (all of them become samples).
        scaler: Optional ``[0, 1]`` label scaler (as for the night dataset).

    Returns the dict of :class:`TelluricTimeseriesDataset` with ``T = 1`` plus
    ``frame_id`` (index of the exposure within its night).
    """

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool,
        star_assignment: np.ndarray,
        n_frames: int,
        scaler: ParameterScaler | None = None,
    ) -> None:
        super().__init__(night_ids, h5_path, pool, star_assignment, scaler)
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
            # keep a length-1 frame axis: (1, N) and (1, P)
            trans = np.asarray(f["transmission"][night, frame:frame + 1], np.float32)
            lab = np.asarray(f["labels"][night, frame:frame + 1], np.float32)

        observed = trans * s[None, :]                                  # (1, N)
        time = lab[:, [self._time_idx]]                                # (1, 1)
        metadata = lab[:, self._metadata_idx]                          # (1, 3)
        output_params = lab[:, self._target_idx]                       # (1, P)

        if self.scaler is not None:
            metadata = self.scaler.transform_metadata(metadata)
            output_params = self.scaler.transform_params(output_params)
            time = self.scaler.transform_time(time)

        return {
            "observed": torch.from_numpy(observed),
            "stellar": torch.from_numpy(s),
            "telluric": torch.from_numpy(trans),
            "theta": {
                "time": torch.from_numpy(time[:, 0]),                  # (1,)
                "metadata": torch.from_numpy(metadata),
                "output_params": torch.from_numpy(output_params),
            },
            "night_id": night,
            "frame_id": frame,
            "star_index": star_idx,
        }
