"""Tests for the [0, 1] min-max scaling of the metadata from declared bounds."""

from __future__ import annotations

import numpy as np
import pytest

from tellurics.data.scaling import MinMax


def _scaler(clip: bool = False) -> MinMax:
    return MinMax.from_bounds(["airmass", "PWV"], {"airmass": (1.0, 2.0), "PWV": (0.0, 25.0)}, clip=clip)


def test_formula_is_min_max() -> None:
    scaled = _scaler().transform(np.array([[1.5, 5.0], [2.0, 25.0]], np.float32))
    assert np.allclose(scaled, [[0.5, 0.2], [1.0, 1.0]])


def test_inverse_recovers_physical_units() -> None:
    x = np.array([[1.25, 12.5]], np.float32)
    s = _scaler()
    assert np.allclose(s.inverse(s.transform(x)), x)


def test_out_of_bounds_raises_unless_clipped() -> None:
    x = np.array([[2.3, 30.0]], np.float32)
    with pytest.raises(ValueError, match="outside"):
        _scaler().transform(x)
    assert np.allclose(_scaler(clip=True).transform(x), [[1.0, 1.0]])


def test_missing_and_degenerate_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="Missing bounds"):
        MinMax.from_bounds(["airmass", "PWV"], {"airmass": (1.0, 2.0)})
    with pytest.raises(ValueError, match="smaller"):
        MinMax.from_bounds(["airmass"], {"airmass": (1.0, 1.0)})


def test_needs_two_dimensional_input_with_matching_columns() -> None:
    with pytest.raises(ValueError, match="2 dimensions"):
        _scaler().transform(np.array([1.5, 5.0]))
    with pytest.raises(ValueError, match="columns"):
        _scaler().transform(np.zeros((1, 3)))
