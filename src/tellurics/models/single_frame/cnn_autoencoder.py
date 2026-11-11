"""Single-frame 1D-CNN autoencoder for telluric transmission.

Shape-annotated data flow (N = 51557, encoder_channels = [16, 32, 64, 96, 128],
stride = 2, bottleneck_channels = 8, latent_dim = 128)::

    observed (B, 1, N) -> flatten -> x (B, N) -> (B, 1, N)
        | encoder: 5 x [Conv1d(k, stride 2) + BatchNorm + GELU]
        v
    (B, 128, 1612)
        | 1x1 Conv squeeze -> (B, 8, 1612) -> flatten (B, 12896)
        | Linear -> latent (B, 128)                       <- bottleneck
        | Linear -> (B, 8 * 1612) -> reshape (B, 8, 1612)
        | 1x1 Conv expand -> (B, 128, 1612)
        | decoder: 5 x [ConvTranspose1d(k, stride 2) + BatchNorm + GELU]
        v
    (B, 16, 51584) -> crop to N -> Conv1d(k) -> z (B, N)
        | + template, activation
        v
    telluric_pred (B, 1, N)

The flatten + Linear bottleneck gives every position of the decoder its own
weights, so the decoder knows *where* on the wavelength grid it is (a purely
convolutional decoder is translation equivariant and could not place the
telluric lines at their fixed wavelengths from a global code).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tellurics.config.model import CNNAutoencoderConfig
from tellurics.models.single_frame.base import SingleFrameTelluricModel
from tellurics.utils.registry import ModelRegistry

__all__ = ["CNNAutoencoder"]


def _conv_out_len(length: int, kernel: int, stride: int, padding: int) -> int:
    return (length + 2 * padding - kernel) // stride + 1


@ModelRegistry.register("cnn_autoencoder")
class CNNAutoencoder(SingleFrameTelluricModel):
    """Strided Conv1d encoder -> dense bottleneck -> ConvTranspose1d decoder."""

    def __init__(self, config: CNNAutoencoderConfig) -> None:
        super().__init__(config.num_wavelength_bins, config.telluric_activation)
        k, s = config.kernel_size, config.stride
        if k % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {k}")
        pad = k // 2
        chans = list(config.encoder_channels)
        self.config = config

        # ---- encoder ------------------------------------------------------ #
        enc: list[nn.Module] = []
        c_in, length = 1, self.n_wavelength
        for c_out in chans:
            enc += [
                nn.Conv1d(c_in, c_out, k, stride=s, padding=pad),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]
            c_in, length = c_out, _conv_out_len(length, k, s, pad)
        self.encoder = nn.Sequential(*enc)
        self.enc_len = length                                   # e.g. 1612
        bc = config.bottleneck_channels

        # ---- dense bottleneck --------------------------------------------- #
        self.squeeze = nn.Conv1d(chans[-1], bc, kernel_size=1)
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Dropout(config.dropout),
            nn.Linear(bc * self.enc_len, config.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(config.latent_dim, bc * self.enc_len), nn.GELU(),
        )
        self.expand = nn.Sequential(nn.Conv1d(bc, chans[-1], kernel_size=1), nn.GELU())

        # ---- decoder (mirror) --------------------------------------------- #
        # ConvTranspose1d(k, s, padding=k//2, output_padding=s-1) maps L -> s*L.
        dec: list[nn.Module] = []
        dec_chans = list(reversed(chans)) + [chans[0]]          # 128,96,64,32,16,16
        for c_i, c_o in zip(dec_chans[:-1], dec_chans[1:]):
            dec += [
                nn.ConvTranspose1d(c_i, c_o, k, stride=s, padding=pad,
                                   output_padding=s - 1),
                nn.BatchNorm1d(c_o),
                nn.GELU(),
            ]
        self.decoder = nn.Sequential(*dec)
        self.dec_len = self.enc_len * s ** len(chans)           # >= N
        assert self.dec_len >= self.n_wavelength, (self.dec_len, self.n_wavelength)
        self.head = nn.Conv1d(chans[0], 1, k, padding=pad)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        h = self.encoder(x.unsqueeze(1))                        # (B, C, L_enc)
        latent = self.to_latent(self.squeeze(h))                # (B, latent_dim)
        h = self.from_latent(latent).view(b, -1, self.enc_len)  # (B, bc, L_enc)
        h = self.decoder(self.expand(h))                        # (B, C0, L_dec)
        h = h[..., : self.n_wavelength]                         # crop to N
        z = self.head(h).squeeze(1)                             # (B, N)
        return z, latent
