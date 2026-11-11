"""Per-observation transformer autoencoder for telluric transmission.

Input: the observed spectrum ``X = T_tell * S`` (telluric transmission times
the raw stellar spectrum, built by the dataset), its wavelength grid, and the
exposure's metadata (``data.metadata_columns``, e.g. airmass + pwv).

* **Wavelengths as positional encoding.** Every flux patch is embedded and
  added to an encoding of *its own wavelengths*: the grid is normalised to
  ``[0, 1]`` per spectrum, each sample gets Fourier features
  ``sin / cos(2^k * pi * lambda)``, ``k = 0..F-1``, and a Linear maps the
  patch's features to ``d_model``. The decoder's per-patch queries are built
  from the same wavelength features, so both sides know *where on the
  spectrum* every token sits.
* **Metadata as a token.** As in the whole-night model, the metadata goes
  through the shared :class:`~tellurics.models.metadata.MetadataEncoder`
  (to ``metadata_enc_dim``, then a Linear to ``d_model``) and is concatenated
  to the sequence as one extra
  token: the encoder input is ``[patch tokens ; metadata token]``, the token
  form of ``[spectrum ; metadata]``.

Shape-annotated data flow (N = 2048, patch_size = 16 -> 128 patches,
d_model = 256, M = 2, num_latents = 16, latent_dim = 64)::

    observed (B, N)                                    wavelength (B, N)
        | patchify                     (B, 128, 16)    (B, 128, 16)
        | Linear(flux) + Linear(Fourier(lambda))       -> (B, 128, 256)
    metadata (B, 2) -> MetadataEncoder (B, 16) -> Linear -> metadata token (B, 1, 256)
        | concat                       (B, 129, 256)
        | TransformerEncoder           (B, 129, 256)
        | cross-attention from 16 learned latent queries
        |                              (B, 16, 256)
        | Linear -> latent             (B, 16, 64)        <- bottleneck
        | Linear                       (B, 16, 256)
        | TransformerDecoder: 128 wavelength-encoded per-patch queries attend
        |   to the latents             (B, 128, 256)
        | LayerNorm + Linear           (B, 128, 16)
        v
    unpatchify + crop -> z (B, N) -> + template, activation
    telluric_pred (B, N)

The only path from encoder to decoder is the ``num_latents x latent_dim``
bottleneck (the decoder queries carry wavelength positions, not flux).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.config.model import TransformerAutoencoderConfig
from tellurics.models.metadata import MetadataEncoder
from tellurics.models.single_frame.base import SingleFrameTelluricModel
from tellurics.utils.registry import ModelRegistry

__all__ = ["TransformerAutoencoder", "WavelengthEncoding"]


class WavelengthEncoding(nn.Module):
    """Positional encoding of a patch from the wavelengths of its samples.

    ``wavelength (B, n_patches, patch_size)`` (normalised to ``[0, 1]``) ->
    Fourier features ``(B, n_patches, patch_size * 2F)`` -> Linear ->
    ``(B, n_patches, d_model)``.
    """

    def __init__(self, patch_size: int, n_frequencies: int, d_model: int) -> None:
        super().__init__()
        freqs = (2.0 ** torch.arange(n_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)          # (F,)
        self.proj = nn.Linear(patch_size * 2 * n_frequencies, d_model)

    def forward(self, wl_patches: torch.Tensor) -> torch.Tensor:
        ang = wl_patches.unsqueeze(-1) * self.freqs                     # (B, P#, p, F)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)               # (B, P#, p, 2F)
        return self.proj(feats.flatten(2))                              # (B, P#, d)


@ModelRegistry.register("transformer_autoencoder")
class TransformerAutoencoder(SingleFrameTelluricModel):
    """Patch tokens (+ wavelength PE, + metadata token) -> encoder -> latents -> decoder."""

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

        # ---- tokens: flux patches + wavelength PE, and a metadata token --- #
        self.patch_embed = nn.Linear(p, d)
        self.enc_wavelength = WavelengthEncoding(p, config.wavelength_frequencies, d)
        self.metadata_encoder = MetadataEncoder(
            config.metadata_dim, config.metadata_enc_dim,
            config.metadata_enc_hidden, dropout=config.dropout,
        )
        self.metadata_to_token = nn.Linear(config.metadata_enc_dim, d)
        self.metadata_type = nn.Parameter(torch.zeros(1, 1, d))   # "this is metadata"

        # ---- encoder ------------------------------------------------------ #
        self.encoder = nn.TransformerEncoder(
            enc_layer(), config.num_encoder_layers, enable_nested_tensor=False,
        )
        self.enc_norm = nn.LayerNorm(d)

        # ---- bottleneck: learned queries cross-attend to the tokens ------- #
        self.latent_queries = nn.Parameter(torch.zeros(1, config.num_latents, d))
        self.compress = dec_layer()
        self.to_latent = nn.Linear(d, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, d)

        # ---- decoder: wavelength-encoded per-patch queries ---------------- #
        self.dec_wavelength = WavelengthEncoding(p, config.wavelength_frequencies, d)
        self.dec_query_base = nn.Parameter(torch.zeros(1, 1, d))
        self.decoder = nn.TransformerDecoder(dec_layer(), config.num_decoder_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, p)

        for param in (self.metadata_type, self.latent_queries, self.dec_query_base):
            nn.init.trunc_normal_(param, std=0.02)

    # ------------------------------------------------------------------ #
    def _patchify(self, v: torch.Tensor) -> torch.Tensor:
        """``(B, N) -> (B, n_patches, patch_size)`` (right-pad by replication)."""
        if self.pad:
            v = F.pad(v.unsqueeze(1), (0, self.pad), mode="replicate").squeeze(1)
        return v.reshape(v.size(0), self.n_patches, self.patch_size)

    @staticmethod
    def _normalise(wavelength: torch.Tensor) -> torch.Tensor:
        """Per-spectrum min-max normalisation of the grid to ``[0, 1]``."""
        lo = wavelength.amin(dim=-1, keepdim=True)
        hi = wavelength.amax(dim=-1, keepdim=True)
        return (wavelength - lo) / (hi - lo).clamp_min(1e-12)

    def reconstruct(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        wavelength: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata is None:
            raise ValueError("TransformerAutoencoder needs metadata (B, 1, M)")
        if wavelength is None:
            raise ValueError(
                "TransformerAutoencoder needs the wavelength grid (B, N) "
                "(batch['wavelength']) for its positional encoding"
            )
        b = x.size(0)
        wl = self._patchify(self._normalise(wavelength.to(x.dtype)))   # (B, P#, p)

        tokens = self.patch_embed(self._patchify(x)) + self.enc_wavelength(wl)  # (B, P#, d)
        m_code = self.metadata_encoder(metadata)                       # (B, C_meta)
        meta_tok = self.metadata_to_token(m_code).unsqueeze(1) + self.metadata_type  # (B, 1, d)
        h = torch.cat([tokens, meta_tok], dim=1)                       # (B, P#+1, d)
        h = self.enc_norm(self.encoder(h))

        q = self.latent_queries.expand(b, -1, -1)                      # (B, Q, d)
        latent = self.to_latent(self.compress(q, h))                   # (B, Q, latent_dim)
        memory = self.from_latent(latent)                              # (B, Q, d)

        dq = self.dec_query_base + self.dec_wavelength(wl)             # (B, P#, d)
        out = self.dec_norm(self.decoder(dq, memory))                  # (B, P#, d)
        z = self.unpatch(out).reshape(b, -1)[:, : self.n_wavelength]   # (B, N)
        return z, latent.flatten(1)                                    # latent (B, Q*latent_dim)
