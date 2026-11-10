"""Whole-night dataset: one sample = one night of exposures.

Reads the night-major HDF5 built once by ``tests/testing_dataset_reshape.ipynb``
(``transmission`` / ``labels`` / ``wavelength``) and turns each night into the
inputs the :class:`~tellurics.models.predictor.NeuralTelluricPredictor` consumes.

When a :class:`~tellurics.data.scaling.ParameterScaler` is supplied, the
per-exposure metadata, the target parameters and ``time_hours`` are mapped onto
``[0, 1]`` with the fixed physical bounds declared in
:mod:`tellurics.config.parameters`. That keeps ``MSE(param_pred, params)`` on a
single scale and makes the bounded (``sigmoid``) head able to reach the
targets; predictions come back to physical units via
:meth:`ParameterScaler.inverse_params`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tellurics.config.parameters import (
    METADATA_COLUMNS,
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
)
from tellurics.data.scaling import ParameterScaler
from tellurics.data.stellar import StellarPool

# --------------------------------------------------------------------------- #
# Column book-keeping
# --------------------------------------------------------------------------- #
# The schema and the physical bounds of every column live together in
# tellurics.config.parameters -- the source of truth shared with the config's
# ``data.param_bounds`` -- so the bounds and the index layout can never
# disagree. The names are re-exported here because this module publishes the
# data contract (and did own them before).
__all__ = [
    "TelluricTimeseriesDataset",
    "read_label_columns",
    "TIME_COLUMN",
    "METADATA_COLUMNS",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
]


def _decode_columns(columns) -> list[str]:
    """Decode an h5py 'columns' attribute into a list of strings."""
    out: list[str] = []
    for c in columns:
        if isinstance(c, bytes):
            out.append(c.decode("utf-8"))
        else:
            out.append(str(c))
    return out


def read_label_columns(h5_path: str | Path) -> list[str]:
    """Label column names stored in the file's ``labels.attrs['columns']``."""
    import h5py

    with h5py.File(Path(h5_path), "r") as f:
        return _decode_columns(f["labels"].attrs.get("columns", []))


class TelluricTimeseriesDataset(Dataset[dict[str, object]]):
    """Whole-night samples from the night-major HDF5.

    Each sample is one night ``(T, N)``. The observed spectrum is
    ``observed = transmission * S`` with ``S`` the stellar spectrum assigned
    to that night (fixed), which is also returned so the estimator encodes it.

    Returns a dict with:
        observed:   (T, N)   X = T_tell * S
        stellar:    (N,)     the S used to build observed
        theta: {
            "time":          (T,)
            "metadata":      (T, 3)   pressure, temperature, humidity
            "output_params": (T, P)   the P regression targets
        }
        night_id:   int      index in [0, n_nights)
        star_index: int      index into the stellar pool
    ``time`` and ``metadata`` are model *inputs*; ``output_params`` are the
    regression *targets* and are never fed to the model. With a ``scaler`` all
    three are returned in ``[0, 1]`` (each group independently, according to
    which ``data.scale_*`` flags were enabled).    """

    metadata_columns = METADATA_COLUMNS
    time_column = TIME_COLUMN
    target_columns = TARGET_COLUMNS

    def __init__(
        self,
        night_ids: list[int],
        h5_path: str | Path,
        pool: StellarPool,
        star_assignment: np.ndarray,
        scaler: ParameterScaler | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.night_ids = list(night_ids)
        self.pool = pool
        self.star_assignment = np.asarray(star_assignment)
        self.scaler = scaler

        # Resolve the column layout from the file's stored 'columns' attr.
        # Single source of truth, so the indices can never disagree with the
        # data they index into.
        self._label_columns = self._read_label_columns()
        for expected in (*self.metadata_columns, self.time_column,
                         *self.target_columns):
            if expected not in self._label_columns:
                raise KeyError(
                    f"expected label column '{expected}' not in file columns "
                    f"{self._label_columns}"
                )
        idx = {c: self._label_columns.index(c) for c in self._label_columns}
        self._time_idx = idx[self.time_column]
        self._metadata_idx = [idx[c] for c in self.metadata_columns]
        self._target_idx = [idx[c] for c in self.target_columns]
        self._check_scaler_columns()

    def _check_scaler_columns(self) -> None:
        """Fail loudly if the scaler's column order disagrees with ours.

        A mismatch would silently scale each parameter with another one's
        bounds, so it is checked at construction time.
        """
        if self.scaler is None:
            return
        expected = {
            "metadata": tuple(self.metadata_columns),
            "params": tuple(self.target_columns),
            "time": (self.time_column,),
        }
        for group_name, columns in expected.items():
            group = getattr(self.scaler, group_name)
            if group is not None and group.columns != columns:
                raise ValueError(
                    f"scaler.{group_name} is built for columns "
                    f"{group.columns} but the dataset uses {columns}"
                )

    def _open(self):
        import h5py

        return h5py.File(self.h5_path, "r")

    def _read_label_columns(self) -> list[str]:
        return read_label_columns(self.h5_path)

    def __len__(self) -> int:
        return len(self.night_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        night = self.night_ids[idx]
        star_idx = int(self.star_assignment[night])
        s, _ = self.pool[star_idx]                    # (N,) float32

        with self._open() as f:
            trans = np.asarray(f["transmission"][night], np.float32)   # (T, N)
            lab = np.asarray(f["labels"][night], np.float32)           # (T, P)

        s_t = s[None, :]                              # (1, N)
        observed = trans * s_t                        # (T, N)

        # Slice ``time_hours`` as (T, 1) so every group handed to the scaler is
        # 2-D ``(T, P)``; it is squeezed back to (T,) for the model afterwards.
        time = lab[:, [self._time_idx]]                                 # (T, 1)
        metadata = lab[:, self._metadata_idx]                           # (T, 3)
        output_params = lab[:, self._target_idx]                        # (T, P)

        if self.scaler is not None:
            metadata = self.scaler.transform_metadata(metadata)
            output_params = self.scaler.transform_params(output_params)
            time = self.scaler.transform_time(time)                     # (T, 1)

        time = time[:, 0]                                               # (T,)

        return {
            "observed": torch.from_numpy(observed),
            "stellar": torch.from_numpy(s),
            "theta": {
                "time": torch.from_numpy(time),
                "metadata": torch.from_numpy(metadata),
                "output_params": torch.from_numpy(output_params),
            },
            "night_id": night,
            "star_index": star_idx,
        }
