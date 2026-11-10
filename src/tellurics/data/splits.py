"""Night-level train/val/test splitting.

Splits are made on whole *nights*, never on exposures: the exposures of one
night are correlated, so a single night must never straddle two splits.
"""

from __future__ import annotations

import numpy as np


def split_night_indices(
    n_nights: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split night *ids* (not exposures) into train/val/test lists.

    Args:
        n_nights: Total number of independent nights.
        train_fraction, val_fraction: Fractions to allocate; the remainder
            (``1 - train_fraction - val_fraction``) becomes the test split.
        seed: RNG seed, making the split reproducible across runs.

    Returns:
        ``(train_ids, val_ids, test_ids)`` lists of night ids.
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_nights)
    n_train = int(n_nights * train_fraction)
    n_val = int(n_nights * val_fraction)
    return (
        ids[:n_train].tolist(),
        ids[n_train : n_train + n_val].tolist(),
        ids[n_train + n_val :].tolist(),
    )


def split_night_indices_order(
    n_nights: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split night *ids* (not exposures) into train/val/test lists.

    Args:
        n_nights: Total number of independent nights.
        train_fraction, val_fraction: Fractions to allocate; the remainder
            (``1 - train_fraction - val_fraction``) becomes the test split.
        seed: RNG seed, making the split reproducible across runs.

    Returns:
        ``(train_ids, val_ids, test_ids)`` lists of night ids.
    """
    # rng = np.random.default_rng(seed)
    # ids = rng.permutation(n_nights)
    ids = np.arange(n_nights)
    n_train = int(n_nights * train_fraction)
    n_val = int(n_nights * val_fraction)
    return (
        ids[:n_train].tolist(),
        ids[n_train : n_train + n_val].tolist(),
        ids[n_train + n_val :].tolist(),
    )
