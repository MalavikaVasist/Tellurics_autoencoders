"""Per-exposure telluric-spectrum decoder (shared MLP + learned spectral basis)."""

from __future__ import annotations

import torch
import torch.nn as nn

from tellurics.config.model import TelluricActivation
from tellurics.models.activations import apply_telluric_activation, clear_sky_raw_value

__all__ = ["TelluricDecoder"]


class TelluricDecoder(nn.Module):
    """Shared per-exposure decoder: exposure code -> full telluric spectrum.

    Maps each ``d``-dim per-exposure code of ``H_T`` to the telluric
    transmission of that exposure on the full wavelength grid, so that the
    model returns the whole telluric time series of the night:

        H_T (B, T, d)  ->  telluric_pred (B, T, N)

    Structure (one instance, shared by every exposure)::

        code (d) --MLP--> h (hidden) --Linear(hidden -> N)--> z (N) --act--> T(l)

    The final ``Linear(hidden -> N)`` is a *learned spectral basis*: every
    output wavelength has its own weights, so the absorption lines can sit at
    fixed wavelengths (a purely convolutional upsampler is translation
    equivariant and cannot place sharp lines at absolute positions from a
    64-number code). It is the learned analogue of a PCA / template basis for
    tellurics.

    ``activation`` maps the raw output ``z`` to a transmission:

    * :attr:`TelluricActivation.OPTICAL_DEPTH` (default) --
      ``T = exp(-softplus(z))``, i.e. ``z`` parametrizes an optical depth
      ``tau >= 0`` and ``T`` lies in ``(0, 1]``. Line strengths add linearly in
      ``tau``, which is how column densities / airmass act physically;
    * :attr:`TelluricActivation.SIGMOID` -- ``T = sigmoid(z)`` in ``(0, 1)``;
    * :attr:`TelluricActivation.NONE` -- raw linear output.

    Forward accepts any leading dimensions: ``(..., d) -> (..., N)``.
    """

    def __init__(
        self,
        latent_dim: int,
        n_wavelength: int,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.0,
        activation: TelluricActivation | str = TelluricActivation.OPTICAL_DEPTH,
    ) -> None:
        super().__init__()
        assert latent_dim > 0 and n_wavelength > 0 and n_layers >= 1
        self.latent_dim = latent_dim
        self.n_wavelength = n_wavelength
        self.activation = TelluricActivation(activation)
        hidden = hidden_dim or 2 * latent_dim

        # Shared MLP on the exposure code: d -> hidden (n_layers - 1 GELU blocks)
        layers: list[nn.Module] = []
        cur = latent_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur, hidden), nn.GELU(), nn.Dropout(dropout)]
            cur = hidden
        self.mlp = nn.Sequential(*layers)          # identity if n_layers == 1

        # Learned spectral basis: hidden -> N (one weight row per wavelength)
        self.basis = nn.Linear(cur, n_wavelength)
        self._init_bias()

    def _init_bias(self) -> None:
        """Start near a fully transparent sky (T ~ 1) for bounded heads."""
        if self.activation is TelluricActivation.NONE:
            return
        with torch.no_grad():
            self.basis.bias.fill_(clear_sky_raw_value(self.activation))

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """Decode exposure codes. ``(..., d) -> (..., N)``."""
        lead = code.shape[:-1]
        x = code.reshape(-1, self.latent_dim)      # (M, d), M = B*T
        z = self.basis(self.mlp(x))                # (M, N)
        out = self._constrain(z)                   # (M, N) transmission
        return out.view(*lead, self.n_wavelength)

    def _constrain(self, z: torch.Tensor) -> torch.Tensor:
        """Map the raw basis output to a transmission."""
        return apply_telluric_activation(z, self.activation)
