"""Fusion modules for the whole-night telluric estimator.

:class:`FusionMLP` fuses the flattened Perceiver night summary with the compact
stellar code (see :mod:`tellurics.models.predictor`).
"""

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Concatenating whole-night fusion MLP.

    ``concat[h_X (q*d), h_S (d)]`` -> MLP -> ``h_fused (q*d)``.

    The output width equals ``q*d`` on purpose: it is reshaped to ``(Q, d)``
    later without any projection/dimension loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_layers >= 1
        layers: list[nn.Module] = []
        cur = input_dim
        for i in range(n_layers):
            out = output_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(cur, out))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            cur = out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in) -> (B, output_dim)``."""
        return self.net(x)
