"""Telluric output activation, shared by the multiple-frame and single-frame models.

Every model ends with an unconstrained per-wavelength output ``z`` that is
mapped to a transmission by :func:`apply_telluric_activation`, according to
:class:`~tellurics.config.model.TelluricActivation`:

* ``optical_depth`` -- ``T = exp(-softplus(z))``: ``z`` parametrizes an optical
  depth ``tau >= 0``, so ``T`` lies in ``(0, 1]``;
* ``sigmoid``       -- ``T = sigmoid(z)`` in ``(0, 1)``;
* ``none``          -- ``T = z`` (unbounded).

:func:`clear_sky_raw_value` gives the raw value ``z`` at which the bounded
activations output ``T ~ 0.98`` (a nearly transparent sky); heads use it to
initialise their bias / template so training starts from a clear sky.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tellurics.config.model import TelluricActivation

__all__ = ["apply_telluric_activation", "clear_sky_raw_value"]


def apply_telluric_activation(
    z: torch.Tensor, activation: TelluricActivation | str
) -> torch.Tensor:
    """Map a raw output ``z`` to a telluric transmission."""
    activation = TelluricActivation(activation)
    if activation is TelluricActivation.OPTICAL_DEPTH:
        return torch.exp(-F.softplus(z))
    if activation is TelluricActivation.SIGMOID:
        return torch.sigmoid(z)
    return z


def clear_sky_raw_value(activation: TelluricActivation | str) -> float:
    """Raw value giving ``T ~ 0.98`` for bounded heads (1.0 for ``none``)."""
    return {
        TelluricActivation.OPTICAL_DEPTH: -4.0,   # exp(-softplus(-4)) ~ 0.98
        TelluricActivation.SIGMOID: 4.0,          # sigmoid(4) ~ 0.98
        TelluricActivation.NONE: 1.0,
    }[TelluricActivation(activation)]
