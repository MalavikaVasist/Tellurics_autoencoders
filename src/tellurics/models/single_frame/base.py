"""Shared plumbing of the per-observation telluric autoencoders.

One observation in, one telluric spectrum out. A batch is a set of
observations (the exposures of all nights pooled, so nights may contribute
different numbers of frames)::

    observed   (B, N)  = T_tell * S
    metadata   (B, M)  e.g. airmass, PWV
    wavelength (B, N)  grid of observed (or (N,), shared)
    reconstruct(x, m, wl)  -->  z (B, N)
    z + template (N,)  --activation-->  telluric_pred (B, N)

* ``reconstruct`` is the architecture (CNN / transformer / SpecPT) and returns
  an unconstrained per-wavelength output ``z``. The CNN and transformer
  autoencoders join the metadata to the spectrum; the transformer also uses the wavelengths as positional encoding.
  SpecPT uses the spectrum only.
* ``template`` is a learned per-wavelength offset: the mean telluric
  spectrum in the activation's raw space. Telluric lines sit at fixed
  wavelengths, so this costs only ``N`` parameters and lets the network spend
  its capacity on the exposure-to-exposure *variation*.
* the activation (:mod:`tellurics.models.activations`): ``optical_depth``
  gives ``T = exp(-softplus(z))`` in ``(0, 1]``.

Every model is called as ``model(observed, metadata=..., wavelength=...)``
by the same training module. The stellar spectrum (if any) is already inside
``observed`` (built by the dataset).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tellurics.config.model import TelluricActivation
from tellurics.models.activations import apply_telluric_activation, clear_sky_raw_value
from tellurics.models.output import ModelOutput

__all__ = ["SingleFrameTelluricModel"]


class SingleFrameTelluricModel(nn.Module):
    """Base class: one observation per row, ``(B, N)`` -> ``reconstruct`` -> ``(B, N)``."""

    def __init__(self, n_wavelength: int, activation: TelluricActivation | str) -> None:
        super().__init__()
        self.n_wavelength = int(n_wavelength)
        self.activation = TelluricActivation(activation)

        # Learned mean telluric spectrum in raw (pre-activation) space; starts
        # at a near-transparent sky (T ~ 0.98) for the bounded heads.
        init = clear_sky_raw_value(self.activation)
        self.template = nn.Parameter(torch.full((self.n_wavelength,), init))

    # -- to implement ------------------------------------------------------- #
    def reconstruct(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        wavelength: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x (B, N), metadata (B, M), wavelength (B, N) -> (z (B, N), latent)``.

        ``z`` is the raw output (before template + activation); ``latent`` is
        the bottleneck code, returned as ``ModelOutput.latent``.
        """
        raise NotImplementedError

    # -- shared ------------------------------------------------------------- #
    def check_input(self, observed: torch.Tensor) -> torch.Tensor:
        """``observed`` must be ``(B, N)``: one observation per row."""
        if observed.dim() != 2 or observed.size(-1) != self.n_wavelength:
            raise ValueError(
                f"{type(self).__name__} takes one observation per row: expected "
                f"observed (B, {self.n_wavelength}); got {tuple(observed.shape)}"
            )
        return observed

    def check_metadata(self, metadata: torch.Tensor | None, batch: int) -> torch.Tensor | None:
        """``metadata`` must be ``(B, M)``; ``None`` stays ``None``."""
        if metadata is None:
            return None
        if metadata.dim() != 2 or metadata.size(0) != batch:
            raise ValueError(
                f"expected metadata ({batch}, M), one row per observation; "
                f"got {tuple(metadata.shape)}"
            )
        return metadata

    def check_wavelength(self, wavelength: torch.Tensor | None, batch: int) -> torch.Tensor | None:
        """``(N,)`` or ``(B, N)`` -> ``(B, N)``; ``None`` stays ``None``."""
        if wavelength is None:
            return None
        n = self.n_wavelength
        if wavelength.dim() == 1:
            wavelength = wavelength.expand(batch, n)
        if wavelength.shape != (batch, n):
            raise ValueError(f"expected wavelength ({batch}, {n}) or ({n},); got {tuple(wavelength.shape)}")
        return wavelength

    def constrain(self, z: torch.Tensor) -> torch.Tensor:
        """Raw output -> transmission."""
        return apply_telluric_activation(z, self.activation)

    def forward(
        self,
        observed: torch.Tensor,
        metadata: torch.Tensor | None = None,    # (B, M) CNN / transformer AE
        wavelength: torch.Tensor | None = None,  # (B, N) transformer AE
    ) -> ModelOutput:
        """Predict one telluric spectrum per observation.

        Args:
            observed: ``(B, N)`` observed spectra ``T_tell * S``, one
                observation per row.
            metadata: ``(B, M)`` metadata of each observation.
            wavelength: ``(B, N)`` (or ``(N,)``) wavelength grid of ``observed``.

        Returns:
            ``ModelOutput(telluric=(B, N), latent=(B, ...))``.
        """
        x = self.check_input(observed)                         # (B, N)
        m = self.check_metadata(metadata, x.size(0))           # (B, M) | None
        wl = self.check_wavelength(wavelength, x.size(0))      # (B, N) | None
        z, latent = self.reconstruct(x, m, wl)                 # (B, N), (B, ...)
        assert z.shape == x.shape, (z.shape, x.shape)
        return ModelOutput(
            telluric=self.constrain(z + self.template),        # (B, N)
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
