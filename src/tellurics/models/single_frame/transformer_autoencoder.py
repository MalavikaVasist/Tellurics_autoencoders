"""Single-frame transformer autoencoder for telluric transmission.

Shape-annotated data flow (N = 51557, patch_size = 128 -> 403 patches,
d_model = 256, num_latents = 16, latent_dim = 64)::

    observed (B, 1, N) -> flatten -> x (B, N) -> pad to 403 * 128
        | patchify                     (B, 403, 128)
        | Linear + learned pos-emb     (B, 403, 256)
        | TransformerEncoder           (B, 403, 256)
        | cross-attention from 16 learned latent queries
        |                              (B, 16, 256)
        | Linear -> latent             (B, 16, 64)        <- bottleneck
        | Linear                       (B, 16, 256)
        | TransformerDecoder: 403 learned per-patch queries attend to the
        |   latents (self-attention among patches + cross-attention)
        |                              (B, 403, 256)
        | LayerNorm + Linear           (B, 403, 128)
        v
    unpatchify + crop -> z (B, N) -> + template, activation
    telluric_pred (B, 1, N)

The only path from encoder to decoder is the ``num_latents x latent_dim``
bottleneck, so this is a genuine autoencoder (no skip connections).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.config.model import TransformerAutoencoderConfig
from tellurics.models.single_frame.base import SingleFrameTelluricModel
from tellurics.utils.registry import ModelRegistry

__all__ = ["TransformerAutoencoder"]


@ModelRegistry.register("transformer_autoencoder")
class TransformerAutoencoder(SingleFrameTelluricModel):
    """Patch tokens -> TransformerEncoder -> latent queries -> TransformerDecoder."""

    def __init__(self, config: TransformerAutoencoderConfig) -> None:
        super().__init__(config.num_wavelength_bins, config.telluric_activation)
        if config.d_model % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        p, d = config.patch_size, config.d_model
        self.patch_size = p
        self.n_patches = math.ceil(self.n_wavelength / p)
        self.pad = self.n_patches * p - self.n_wavelength

        def enc_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d, config.num_heads, config.dim_feedforward, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )

        def dec_layer() -> nn.TransformerDecoderLayer:
            return nn.TransformerDecoderLayer(
                d, config.num_heads, config.dim_feedforward, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )

        # ---- encoder ------------------------------------------------------ #
        self.patch_embed = nn.Linear(p, d)
        self.enc_pos = nn.Parameter(torch.zeros(1, self.n_patches, d))
        self.encoder = nn.TransformerEncoder(
            enc_layer(), config.num_encoder_layers, enable_nested_tensor=False,
        )
        self.enc_norm = nn.LayerNorm(d)

        # ---- bottleneck: learned queries cross-attend to the patch tokens - #
        self.latent_queries = nn.Parameter(torch.zeros(1, config.num_latents, d))
        self.compress = dec_layer()
        self.to_latent = nn.Linear(d, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, d)

        # ---- decoder: per-patch queries attend to the latents ------------- #
        self.dec_queries = nn.Parameter(torch.zeros(1, self.n_patches, d))
        self.decoder = nn.TransformerDecoder(dec_layer(), config.num_decoder_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, p)

        for param in (self.enc_pos, self.latent_queries, self.dec_queries):
            nn.init.trunc_normal_(param, std=0.02)

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        if self.pad:
            x = F.pad(x.unsqueeze(1), (0, self.pad), mode="replicate").squeeze(1)
        tokens = x.view(b, self.n_patches, self.patch_size)        # (B, P#, p)
        h = self.patch_embed(tokens) + self.enc_pos                # (B, P#, d)
        h = self.enc_norm(self.encoder(h))

        q = self.latent_queries.expand(b, -1, -1)                  # (B, Q, d)
        latent = self.to_latent(self.compress(q, h))               # (B, Q, latent_dim)
        memory = self.from_latent(latent)                          # (B, Q, d)

        dq = self.dec_queries.expand(b, -1, -1)                    # (B, P#, d)
        out = self.dec_norm(self.decoder(dq, memory))              # (B, P#, d)
        z = self.unpatch(out).reshape(b, -1)[:, : self.n_wavelength]  # (B, N)
        return z, latent.flatten(1)                                # latent (B, Q*latent_dim)
