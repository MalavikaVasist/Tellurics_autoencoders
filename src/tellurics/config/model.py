"""Model configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelArchitecture(str, Enum):
    """Available model architectures (``ModelRegistry`` ids)."""

    NEURAL_TELLURIC_PREDICTOR = "neural_telluric_predictor"


class ParamActivation(str, Enum):
    """Activation of the parameter head (only the parameter prediction).

    ``param_pred`` has shape ``(B, T, P)`` and holds ``P`` *heterogeneous*
    physical parameters (pressure, humidity, airmass, molecular abundances,
    ...). With min-max scaled targets (``data.scale_params``) every target lies
    in ``[0, 1]``, so the head has to be bounded as well:

    * ``NONE`` -- linear head, for raw (physical-unit) targets;
    * ``SIGMOID`` -- element-wise ``sigmoid``, i.e. every parameter is
      independently constrained to ``(0, 1)``. This is the correct partner of
      ``data.scale_params: true``.

    A ``softmax`` is deliberately not offered: normalizing over ``P`` would
    force pressure + humidity + airmass + ... to sum to 1 within one exposure,
    and normalizing over ``T`` would force each parameter's exposures to sum to
    1 over the night. Neither constraint holds for these parameters.
    """

    NONE = "none"
    SIGMOID = "sigmoid"


class ModelConfig(BaseModel):
    """Configuration for the whole-night telluric parameter estimator.

    No field carries a default: every value is a design or data decision and
    must be stated by the manifest (see ``experiments/``).  All fields are
    consumed by :mod:`tellurics.models.predictor`; see that module's docstring for
    the shape-annotated data flow.
    """

    architecture: ModelArchitecture

    # -- problem dimensions (must match the DataModule contract) ---------- #
    num_wavelength_bins: int = Field(gt=0)
    n_frames_per_series: int = Field(gt=0, description="Exposures per night (T).")
    metadata_dim: int = Field(
        gt=0,
        description="Per-exposure metadata width P (data contract: 3 -> "
                    "pressure, temperature, humidity).",
    )
    param_dim: int = Field(
        gt=0,
        description="Number of telluric/atmospheric parameters per exposure "
                    "(data contract: 16 = len(TARGET_COLUMNS)).",
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
    param_decoder_hidden: int | None = Field(
        gt=0,
        description="Hidden width of the per-exposure param-decoder MLP "
                    "(None -> auto).",
    )
    param_activation: ParamActivation = Field(
        description="Activation applied to the parameter head output only "
                    "(see ParamActivation). Use 'sigmoid' when "
                    "data.scale_params maps the targets onto [0, 1]; the "
                    "inverse transform back to physical units is "
                    "tellurics.data.scaling.ParameterScaler.inverse_params.",
    )
