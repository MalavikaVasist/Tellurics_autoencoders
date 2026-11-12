"""Telluric data: the single-observation dataset, its DataModule, and its pieces.

    datamodule.py          Lightning orchestration (TelluricDataModule)
    datasets/frames.py     one sample = one observation
    scaling.py             [0, 1] min-max scaling of the metadata
    stellar.py             StellarPool + per-night star assignment
    splits.py              night-level train/val/test split
    wavegrid.py            Wavegrid
"""

from tellurics.data.datamodule import TelluricDataModule, build_scaler, read_layout
from tellurics.data.datasets.frames import TelluricFrameDataset, read_label_columns
from tellurics.data.scaling import MinMax
from tellurics.data.splits import split_night_indices
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid

__all__ = [
    "TelluricDataModule",
    "build_scaler",
    "read_layout",
    "TelluricFrameDataset",
    "read_label_columns",
    "MinMax",
    "Wavegrid",
    "StellarPool",
    "assign_stellar_per_night",
    "save_stellar_assignment",
    "split_night_indices",
]
