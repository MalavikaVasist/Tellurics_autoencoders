"""Shared plumbing of the single-frame telluric autoencoders.

Every single-frame model follows the same contract::

    observed (B, 1, N)  --flatten-->  x (B, N)  --reconstruct()-->  z (B, N)
    z + template (N,)   --activation-->  telluric_pred (B, 1, N)

* ``reconstruct`` is the architecture (CNN / transformer / SpecPT) and returns
  an unconstrained per-wavelength output ``z``.
* ``template`` is a learned per-wavelength offset: the mean telluric
  spectrum in the activation's raw space. Telluric lines sit at fixed
  wavelengths, so this costs only ``N`` parameters and lets the network spend
  its capacity on the exposure-to-exposure *variation*.
* the activation is the same as the whole-night head
  (:mod:`tellurics.models.activations`): ``optical_depth``
  gives ``T = exp(-softplus(z))`` in ``(0, 1]``.

The forward signature matches :class:`~tellurics.models.multiple_frames.predictor.NeuralTelluricPredictor`
(``stellar`` / ``metadata`` / ``time`` keywords) so the same training module
drives every model; single-frame autoencoders use the observed spectrum only
and ignore the other inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tellurics.config.model import TelluricActivation
from tellurics.models.activations import apply_telluric_activation, clear_sky_raw_value
from tellurics.models.output import ModelOutput

__all__ = ["SingleFrameTelluricModel"]


class SingleFrameTelluricModel(nn.Module):
    """Base class: flatten ``(B, 1, N)`` -> ``reconstruct`` -> ``(B, 1, N)``."""

    def __init__(self, n_wavelength: int, activation: TelluricActivation | str) -> None:
        super().__init__()
        self.n_wavelength = int(n_wavelength)
        self.activation = TelluricActivation(activation)

        # Learned mean telluric spectrum in raw (pre-activation) space; starts
        # at a near-transparent sky (T ~ 0.98) for the bounded heads.
        init = clear_sky_raw_value(self.activation)
        self.template = nn.Parameter(torch.full((self.n_wavelength,), init))

    # -- to implement ------------------------------------------------------- #
    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, N) -> (z (B, N), latent (B, ...))``.

        ``z`` is the raw output (before template + activation); ``latent`` is
        the bottleneck code, returned as ``ModelOutput.latent``.
        """
        raise NotImplementedError

    # -- shared ------------------------------------------------------------- #
    def flatten_input(self, observed: torch.Tensor) -> torch.Tensor:
        """``(B, 1, N)`` or ``(B, N)`` -> ``(B, N)``."""
        if observed.dim() == 3:
            if observed.size(1) != 1:
                raise ValueError(
                    f"{type(self).__name__} is a single-frame model and expects "
                    f"observed of shape (B, 1, N); got {tuple(observed.shape)}. "
                    "Train it with a single-frame data module."
                )
            observed = observed[:, 0]
        if observed.dim() != 2 or observed.size(-1) != self.n_wavelength:
            raise ValueError(
                f"expected (B, 1, {self.n_wavelength}) or (B, {self.n_wavelength}); "
                f"got {tuple(observed.shape)}"
            )
        return observed

    def constrain(self, z: torch.Tensor) -> torch.Tensor:
        """Raw output -> transmission."""
        return apply_telluric_activation(z, self.activation)

    def forward(
        self,
        observed: torch.Tensor,
        stellar: torch.Tensor | None = None,     # unused (observed-only AE)
        metadata: torch.Tensor | None = None,    # unused
        time: torch.Tensor | None = None,        # unused
    ) -> ModelOutput:
        """Predict the telluric transmission of single exposures.

        Args:
            observed: ``(B, 1, N)`` (or ``(B, N)``) observed spectra ``T * S``.

        Returns:
            ``ModelOutput(telluric=(B, 1, N), latent=bottleneck code)``.
        """
        x = self.flatten_input(observed)                      # (B, N)
        z, latent = self.reconstruct(x)                       # (B, N), (B, ...)
        assert z.shape == x.shape, (z.shape, x.shape)
        telluric = self.constrain(z + self.template)          # (B, N)
        return ModelOutput(
            telluric=telluric.unsqueeze(1),                   # (B, 1, N)
            latent=latent,
        )

    # -- reporting (same API as NeuralTelluricPredictor) -------------------- #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        """Trainable parameters per top-level child (+ the template)."""
        counts = {
            name: sum(p.numel() for p in child.parameters() if p.requires_grad)
            for name, child in self.named_children()
        }
        counts["template"] = self.template.numel()
        return counts
