"""Model configuration: one config class per architecture.

The ``model:`` section of a manifest is validated against exactly one of the
classes below, selected by its ``architecture`` value (a pydantic
discriminated union, :data:`AnyModelConfig`). Each experiment therefore only
states the fields of its own model:

===========================  =====================================
``architecture``             config class
===========================  =====================================
``cnn_autoencoder``          :class:`CNNAutoencoderConfig`
``transformer_autoencoder``  :class:`TransformerAutoencoderConfig`
``specpt``                   :class:`SpecPTConfig`
===========================  =====================================

Every model takes one observation and returns one telluric spectrum:
``observed (B, N)`` (+ metadata ``(B, M)``) -> ``telluric (B, N)``.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures (``ModelRegistry`` ids)."""

    CNN_AUTOENCODER = "cnn_autoencoder"
    TRANSFORMER_AUTOENCODER = "transformer_autoencoder"
    SPECPT = "specpt"


class TelluricActivation(str, Enum):
    """Output activation of the telluric head (only the spectrum prediction).

    ``telluric_pred`` has shape ``(B, N)``: the telluric transmission of each
    observation on the full wavelength grid. Transmission
    is physical in ``[0, 1]`` and is never rescaled by the data pipeline:

    * ``OPTICAL_DEPTH`` -- ``T = exp(-softplus(z))``: the head predicts an
      optical depth ``tau >= 0``, so ``T`` lies in ``(0, 1]``. Recommended.
    * ``SIGMOID`` -- ``T = sigmoid(z)`` in ``(0, 1)``.
    * ``NONE`` -- linear head (unbounded).
    """

    OPTICAL_DEPTH = "optical_depth"
    SIGMOID = "sigmoid"
    NONE = "none"


# =========================================================================== #
# Single-frame models:  observed (B, N) -> telluric (B, N)
# =========================================================================== #
class _SingleFrameConfig(BaseModel):
    """Fields shared by every per-observation autoencoder.

    One observation in, one telluric spectrum out: the exposures of all
    nights are pooled (nights may contribute different numbers of frames), a
    batch is ``observed (B, N)`` + ``metadata (B, M)``, and the model returns
    ``telluric (B, N)`` -- one spectrum per observation.
    """

    num_wavelength_bins: int = Field(gt=0, description="Spectral samples N.")

    dropout: float = Field(ge=0.0, lt=1.0)
    telluric_activation: TelluricActivation = Field(
        description="Output activation of the telluric head "
                    "(see TelluricActivation).",
    )


class _MetadataConditionedConfig(_SingleFrameConfig):
    """Single-frame models whose input is ``observed = T * S`` + metadata.

    The metadata vector (``data.metadata_columns``, e.g. airmass + PWV) is
    encoded by the shared :class:`~tellurics.models.metadata.MetadataEncoder`
    and joined to the spectral input.
    """

    metadata_dim: int = Field(
        gt=0, description="Metadata scalars per exposure M; must equal "
                          "len(data.metadata_columns).",
    )
    metadata_enc_dim: int = Field(gt=0, description="MetadataEncoder code size C.")
    metadata_enc_hidden: int | None = Field(
        default=None, gt=0,
        description="MetadataEncoder hidden width (None -> auto).",
    )


class CNNAutoencoderConfig(_MetadataConditionedConfig):
    """Single-frame 1D-CNN autoencoder (:mod:`tellurics.models.single_frame.cnn_autoencoder`).

    ``observed (B, N) -> strided Conv1d stack -> 1x1 squeeze -> flatten
    -> concat MetadataEncoder(metadata) -> Linear -> latent (B, latent_dim)
    -> Linear -> ConvTranspose1d stack -> telluric (B, N)``.
    """

    architecture: Literal[ModelArchitecture.CNN_AUTOENCODER]
    encoder_channels: list[int] = Field(
        min_length=1, description="Channels of the strided encoder Conv1d stack; "
                                  "the decoder mirrors them.",
    )
    kernel_size: int = Field(gt=0, description="Conv kernel (odd).")
    stride: int = Field(gt=1, description="Down/up-sampling factor per layer.")
    bottleneck_channels: int = Field(
        gt=0, description="Channels kept by the 1x1 squeeze before flattening.",
    )
    latent_dim: int = Field(gt=0, description="Size of the latent code.")


class TransformerAutoencoderConfig(_MetadataConditionedConfig):
    """Single-frame transformer autoencoder (:mod:`tellurics.models.single_frame.transformer_autoencoder`).

    ``observed (B, N) -> patches (B, n_patches, patch_size) -> Linear
    + wavelength positional encoding -> [patch tokens ; metadata token]
    -> TransformerEncoder -> cross-attention onto num_latents learned queries
    (bottleneck: num_latents x latent_dim) -> TransformerDecoder from
    wavelength-encoded per-patch queries -> Linear -> telluric (B, N)``.
    """

    architecture: Literal[ModelArchitecture.TRANSFORMER_AUTOENCODER]
    patch_size: int = Field(gt=0, description="Wavelength samples per token.")
    d_model: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_encoder_layers: int = Field(gt=0)
    num_decoder_layers: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    num_latents: int = Field(gt=0, description="Learned bottleneck tokens.")
    latent_dim: int = Field(gt=0, description="Width of each bottleneck token.")
    wavelength_frequencies: int = Field(
        gt=0, description="Fourier frequencies of the wavelength positional "
                          "encoding (sin/cos of 2^k * pi * lambda_norm, "
                          "k = 0..F-1, for every sample of a patch).",
    )


class SpecPTConfig(_SingleFrameConfig):
    """SpecPT adapted to full-resolution single frames (:mod:`tellurics.models.single_frame.specpt`).

    SpecPT conv stem (Conv1d k=41/21/11, stride 2, BatchNorm, ReLU, MaxPool)
    -> strided token conv -> TransformerEncoder / TransformerDecoder (decoder
    attends to the encoder output, as in SpecPT) -> per-token linear1/ReLU/
    linear2 reconstruction of that token's wavelength window -> (B, N).
    """

    architecture: Literal[ModelArchitecture.SPECPT]
    stem_channels: list[int] = Field(
        min_length=1, description="SpecPT conv-stem channels (paper: 64, 128, 256).",
    )
    stem_kernels: list[int] = Field(
        min_length=1, description="SpecPT conv-stem kernels (paper: 41, 21, 11).",
    )
    token_stride: int = Field(
        gt=0, description="Extra downsampling of the stem output into tokens; "
                          "each token covers 2**len(stem)*2*token_stride samples.",
    )
    d_model: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_encoder_layers: int = Field(gt=0)
    num_decoder_layers: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    head_hidden: int = Field(gt=0, description="Hidden width of linear1 in the "
                                               "per-token reconstruction head.")


AnyModelConfig = Annotated[
    Union[
        CNNAutoencoderConfig,
        TransformerAutoencoderConfig,
        SpecPTConfig,
    ],
    Field(discriminator="architecture"),
]
"""The ``model:`` section: the config class is chosen by ``architecture``."""
