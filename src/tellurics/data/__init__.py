"""Whole-night telluric data: the dataset, its DataModule, and its pieces.

    datamodule.py          Lightning orchestration (TelluricDataModule)
    datasets/timeseries.py one sample = one whole night
    scaling.py             [0, 1] min-max scaling from the declared bounds
    stellar.py             StellarPool + per-night star assignment
    splits.py              night-level train/val/test split
    wavegrid.py            Wavegrid
"""

from tellurics.data.datamodule import TelluricDataModule, build_scaler
from tellurics.data.datasets.timeseries import (
    METADATA_COLUMNS,
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    TelluricTimeseriesDataset,
    read_label_columns,
)
from tellurics.data.scaling import MinMax, ParameterScaler
from tellurics.data.splits import split_night_indices
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid

__all__ = [
    # DataModule.
    "TelluricDataModule",
    "build_scaler",
    # Dataset + the label-column schema it expects.
    "TelluricTimeseriesDataset",
    "read_label_columns",
    "TIME_COLUMN",
    "METADATA_COLUMNS",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
    # [0, 1] label scaling.
    "ParameterScaler",
    "MinMax",
    # Supporting pieces.
    "Wavegrid",
    "StellarPool",
    "assign_stellar_per_night",
    "save_stellar_assignment",
    "split_night_indices",
]
