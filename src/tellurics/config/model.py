"""Model configuration: one config class per architecture.

The ``model:`` section of a manifest is validated against exactly one of the
classes below, selected by its ``architecture`` value (a pydantic
discriminated union, :data:`AnyModelConfig`). Each experiment therefore only
states the fields of its own model:

=============================  ==============================  ===============
``architecture``               config class                    sample
=============================  ==============================  ===============
``neural_telluric_predictor``  :class:`ModelConfig`            whole night
``cnn_autoencoder``            :class:`CNNAutoencoderConfig`   single frame
``transformer_autoencoder``    :class:`TransformerAutoencoderConfig`  single frame
``specpt``                     :class:`SpecPTConfig`           single frame
=============================  ==============================  ===============

``single_frame`` (a class constant) tells the training script which dataset
to build: whole nights ``(B, T, N)`` for the whole-night model, or single
observations ``(B, N)`` -- the exposures of all nights pooled -- for the
per-observation models.
"""

from enum import Enum
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures (``ModelRegistry`` ids)."""

    NEURAL_TELLURIC_PREDICTOR = "neural_telluric_predictor"
    CNN_AUTOENCODER = "cnn_autoencoder"
    TRANSFORMER_AUTOENCODER = "transformer_autoencoder"
    SPECPT = "specpt"


class TelluricActivation(str, Enum):
    """Output activation of the telluric head (only the spectrum prediction).

    ``telluric_pred`` has shape ``(B, T, N)`` and holds the telluric
    transmission of every exposure on the full wavelength grid. Transmission
    is physical in ``[0, 1]`` and is never rescaled by the data pipeline:

    * ``OPTICAL_DEPTH`` -- ``T = exp(-softplus(z))``: the head predicts an
      optical depth ``tau >= 0``, so ``T`` lies in ``(0, 1]``. Recommended.
    * ``SIGMOID`` -- ``T = sigmoid(z)`` in ``(0, 1)``.
    * ``NONE`` -- linear head (unbounded).
    """

    OPTICAL_DEPTH = "optical_depth"
    SIGMOID = "sigmoid"
    NONE = "none"


class ModelConfig(BaseModel):
    """Configuration for the whole-night telluric time-series estimator.

    No field carries a default: every value is a design or data decision and
    must be stated by the manifest (see ``experiments/multiple_frames/``).  All fields are
    consumed by :mod:`tellurics.models.multiple_frames.predictor`; see that module's docstring for
    the shape-annotated data flow.
    """

    single_frame: ClassVar[bool] = False

    architecture: Literal[ModelArchitecture.NEURAL_TELLURIC_PREDICTOR]

    # -- problem dimensions (must match the DataModule contract) ---------- #
    num_wavelength_bins: int = Field(gt=0)
    n_frames_per_series: int = Field(gt=0, description="Exposures per night (T).")
    metadata_dim: int = Field(
        gt=0,
        description="Per-exposure metadata width P (data contract: 3 -> "
                    "pressure, temperature, humidity).",
    )
    use_stellar: bool = Field(
        default=True,
        description="Feed the night's stellar spectrum through the "
                    "StellarEncoder into the fusion. False for observed-flux "
                    "data (data.observed_key set) and real observations, "
                    "which carry no separate stellar spectrum.",
    )

    # -- architecture geometry -------------------------------------------- #
    num_queries: int = Field(gt=0, description="Number of learned Perceiver latents (Q).")
    spectral_latent_dim: int = Field(
        gt=0, description="Dim of the per-exposure spectral code (d)."
    )
    num_heads: int = Field(
        gt=0, description="Heads of the temporal cross-attention."
    )
    dropout: float = Field(ge=0.0, lt=1.0)

    # -- shared strided 1D-CNN over each exposure of X -------------------- #
    x_encoder_channels: list[int] = Field(
        description="Channels of the shared strided 1D-CNN applied to each exposure of X.",
    )
    x_encoder_kernel: int = Field(gt=0)
    x_encoder_stride: int = Field(gt=0)
    s_encoder_channels: list[int] = Field(
        description="Channels of the independent stellar (S) 1D-CNN.",
    )
    s_encoder_kernel: int = Field(gt=0)
    s_encoder_stride: int = Field(gt=0)
    encoder_pool_bins: int = Field(
        gt=0,
        description="Adaptive spectral bins kept per channel before the code read-out.",
    )
    fusion_hidden: int = Field(
        ge=0,
        description="Fusion MLP width. 0 -> num_queries * spectral_latent_dim.",
    )
    fusion_layers: int = Field(gt=0)
    metadata_enc_dim: int = Field(
        gt=0,
        description="Width of the per-exposure metadata code (C), concatenated "
                    "with the spectral code before the temporal Perceiver.",
    )
    metadata_enc_hidden: int | None = Field(
        gt=0,
        description="Hidden width of the per-exposure metadata encoder MLP "
                    "(None -> auto).",
    )
    time_enc_dim: int = Field(
        gt=0,
        description="Width (k) of the per-exposure time code produced by the "
                    "TimeEncoder from continuous exposure times (time_hours), "
                    "concatenated with the spectral + metadata codes before "
                    "the temporal Perceiver.",
    )
    telluric_decoder_hidden: int | None = Field(
        gt=0,
        description="Hidden width of the per-exposure telluric-decoder MLP "
                    "(None -> 2 * spectral_latent_dim).",
    )
    telluric_activation: TelluricActivation = Field(
        description="Activation applied to the telluric head output only "
                    "(see TelluricActivation). 'optical_depth' keeps the "
                    "predicted transmission in (0, 1].",
    )


# =========================================================================== #
# Single-frame models:  observed (B, 1, N) -> flatten (B, N) -> telluric (B, 1, N)
# =========================================================================== #
class _SingleFrameConfig(BaseModel):
    """Fields shared by every per-observation autoencoder.

    One observation in, one telluric spectrum out: the exposures of all
    nights are pooled (nights may contribute different numbers of frames), a
    batch is ``observed (B, N)`` + ``metadata (B, M)``, and the model returns
    ``telluric (B, N)`` -- one spectrum per observation.
    """

    single_frame: ClassVar[bool] = True

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
    and joined to the spectral input, as in the whole-night model.
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
        ModelConfig,
        CNNAutoencoderConfig,
        TransformerAutoencoderConfig,
        SpecPTConfig,
    ],
    Field(discriminator="architecture"),
]
"""The ``model:`` section: the config class is chosen by ``architecture``."""
