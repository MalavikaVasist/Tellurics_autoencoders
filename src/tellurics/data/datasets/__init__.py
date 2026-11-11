"""Datasets: one whole night (TelluricTimeseriesDataset) or one exposure (TelluricFrameDataset)."""

from tellurics.data.datasets.frames import TelluricFrameDataset
from tellurics.data.datasets.timeseries import (
    METADATA_COLUMNS,
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    TelluricTimeseriesDataset,
)

__all__ = [
    "TelluricTimeseriesDataset",
    "TelluricFrameDataset",
    "TIME_COLUMN",
    "METADATA_COLUMNS",
    "PHYSICAL_COLUMNS",
    "TARGET_COLUMNS",
]
