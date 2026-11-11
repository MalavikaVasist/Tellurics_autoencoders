"""SpecPT adapted to per-observation telluric extraction.

Based on the SpecPT pre-training autoencoder (conv stem -> transformer
encoder/decoder -> linear reconstruction). Kept from the original:

* the conv stem: Conv1d(1->64, k=41, s=2) / Conv1d(64->128, k=21, s=2) /
  Conv1d(128->256, k=11, s=2), each + BatchNorm + ReLU, then MaxPool1d(3, 2);
* a TransformerEncoder followed by a TransformerDecoder that attends to the
  encoder output (``decoder(tgt=encoded, memory=encoded)``), post-norm, ReLU,
  ``d_model=512, nhead=8, 3 + 3 layers, dim_feedforward=2048, dropout=0.1``;
* a linear1 -> ReLU -> linear2 reconstruction head.

Changed, because the original does not scale to N = 51557 (it was built for
7781 samples):

1. **Tokens instead of one vector.** The original flattens the whole conv
   output into ``Linear(output_size, d_model)`` and runs the transformer on a
   sequence of length 1 -- attention over a single token is the identity, and
   at N = 51557 that Linear alone would need 256 * 3223 * 512 ~ 422M weights.
   Here a strided conv (``token_stride``) turns the stem output into a
   sequence of tokens (403 at N = 51557) with a learned positional
   embedding, so self-attention acts across wavelength.
2. **Per-token reconstruction.** The original ``linear2(output_size ->
   input_size)`` would need 825k * 51557 ~ 42 billion weights. Here linear1 /
   linear2 are applied to every token and each token reconstructs its own
   window of ``16 * token_stride`` wavelength samples (128 by default).
3. The input is padded to a multiple of the total downsampling factor so the
   tokens tile the spectrum exactly; the output is cropped back to N.

Shape-annotated data flow (defaults, N = 51557 -> padded 51584)::

    observed (B, N) -> pad (B, 1, N_pad)
        | stem 3 x [Conv1d s=2 + BN + ReLU], MaxPool s=2   (B, 256, 3224)
        | token Conv1d(256 -> 512, k=s=8)                   (B, 512, 403)
        | + pos-emb -> TransformerEncoder                   (B, 403, 512)
        | TransformerDecoder(tgt=encoded, memory=encoded)   (B, 403, 512)
        | linear1 (512 -> head_hidden) + ReLU + linear2 (-> 128)
        v                                                   (B, 403, 128)
    flatten + crop -> z (B, N) -> + template, activation -> telluric_pred (B, N)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tellurics.config.model import SpecPTConfig
from tellurics.models.single_frame.base import SingleFrameTelluricModel
from tellurics.utils.registry import ModelRegistry

__all__ = ["SpecPT"]


@ModelRegistry.register("specpt")
class SpecPT(SingleFrameTelluricModel):
    """SpecPT conv stem + tokenised transformer encoder/decoder."""

    def __init__(self, config: SpecPTConfig) -> None:
        super().__init__(config.num_wavelength_bins, config.telluric_activation)
        if len(config.stem_channels) != len(config.stem_kernels):
            raise ValueError("stem_channels and stem_kernels must have the same length")
        if any(k % 2 == 0 for k in config.stem_kernels):
            raise ValueError(f"stem_kernels must be odd, got {config.stem_kernels}")
        if config.d_model % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        d = config.d_model

        # ---- SpecPT conv stem (forward_conv in the original) -------------- #
        stem: list[nn.Module] = []
        c_in = 1
        for c_out, k in zip(config.stem_channels, config.stem_kernels):
            stem += [
                nn.Conv1d(c_in, c_out, k, stride=2, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            ]
            c_in = c_out
        stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)

        # ---- tokens (replaces flatten + proj_to_d_model) ------------------ #
        self.tokenize = nn.Conv1d(
            c_in, d, kernel_size=config.token_stride, stride=config.token_stride
        )
        self.downsample = 2 ** len(config.stem_channels) * 2 * config.token_stride
        self.n_tokens = -(-self.n_wavelength // self.downsample)      # ceil
        self.pad = self.n_tokens * self.downsample - self.n_wavelength
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # ---- transformer encoder & decoder (as SpecPT) -------------------- #
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d, config.num_heads, config.dim_feedforward, config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d, config.num_heads, config.dim_feedforward, config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_decoder_layers,
        )

        # ---- reconstruction head (linear1 / linear2, per token) ----------- #
        self.linear1 = nn.Linear(d, config.head_hidden)
        self.linear2 = nn.Linear(config.head_hidden, self.downsample)

    def reconstruct(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,     # unused: SpecPT is spectrum-only
        wavelength: torch.Tensor | None = None,   # unused
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        h = x.unsqueeze(1)                                          # (B, 1, N)
        if self.pad:
            h = F.pad(h, (0, self.pad), mode="replicate")           # (B, 1, N_pad)
        h = self.stem(h)                                            # (B, C, N_pad/16)
        h = self.tokenize(h).transpose(1, 2)                        # (B, n_tok, d)
        assert h.size(1) == self.n_tokens, (h.shape, self.n_tokens)
        h = h + self.pos

        encoded = self.transformer_encoder(h)                       # (B, n_tok, d)
        decoded = self.transformer_decoder(encoded, encoded)        # (B, n_tok, d)

        out = self.linear2(F.relu(self.linear1(decoded)))           # (B, n_tok, D)
        z = out.reshape(b, -1)[:, : self.n_wavelength]              # (B, N)
        return z, encoded.mean(dim=1)                               # latent (B, d)
