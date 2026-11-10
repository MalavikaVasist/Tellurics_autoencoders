"""Tests for configuration validation."""

import pytest

from tellurics.config.parameters import (
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
)
from tellurics.config.data import DataConfig
from tellurics.config.model import ModelArchitecture, ModelConfig, ParamActivation
from tellurics.config.training import TrainingConfig
from tests._bounds import BOUNDS

# Every ModelConfig field, with the problem dims kept tiny.
MODEL_KWARGS = {
    "architecture": "neural_telluric_predictor",
    "num_wavelength_bins": 1024,
    "n_frames_per_series": 12,
    "metadata_dim": 3,
    "param_dim": 16,
    "num_queries": 8,
    "spectral_latent_dim": 32,
    "num_heads": 4,
    "dropout": 0.1,
    "x_encoder_channels": [8, 16, 32],
    "x_encoder_kernel": 7,
    "x_encoder_stride": 2,
    "s_encoder_channels": [8, 16, 32],
    "s_encoder_kernel": 7,
    "s_encoder_stride": 2,
    "encoder_pool_bins": 16,
    "fusion_hidden": 0,
    "fusion_layers": 2,
    "metadata_enc_dim": 16,
    "metadata_enc_hidden": None,
    "time_enc_dim": 16,
    "param_decoder_hidden": None,
    "param_activation": "sigmoid",
}


class TestModelConfig:
    def test_explicit_config(self) -> None:
        config = ModelConfig(**MODEL_KWARGS)
        assert (
            config.architecture is ModelArchitecture.NEURAL_TELLURIC_PREDICTOR
        )
        assert config.n_frames_per_series == 12
        assert config.param_dim == 16

    @pytest.mark.parametrize("field", sorted(MODEL_KWARGS))
    def test_no_field_has_a_default(self, field: str) -> None:
        """Omitting any field is an error: nothing is assumed implicitly."""
        with pytest.raises(ValueError):
            ModelConfig(**{k: v for k, v in MODEL_KWARGS.items() if k != field})

    def test_rejects_unknown_architecture(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(**{**MODEL_KWARGS, "architecture": "cnn"})

    def test_metadata_dim_must_be_positive(self) -> None:
        """metadata_dim=0 would leave the metadata encoder unbuildable."""
        with pytest.raises(ValueError):
            ModelConfig(**{**MODEL_KWARGS, "metadata_dim": 0})

    def test_data_contract_dimensions(self) -> None:
        """The three dims that must agree with the DataModule."""
        config = ModelConfig(**MODEL_KWARGS)
        assert config.metadata_dim == 3   # pressure, temperature, humidity
        assert config.param_dim == 16     # TARGET_COLUMNS
        assert config.num_wavelength_bins == 1024

    def test_param_activation_is_parsed(self) -> None:
        assert ModelConfig(**MODEL_KWARGS).param_activation is ParamActivation.SIGMOID
        assert (
            ModelConfig(**{**MODEL_KWARGS, "param_activation": "none"})
            .param_activation
            is ParamActivation.NONE
        )

    def test_rejects_unknown_activation(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(**{**MODEL_KWARGS, "param_activation": "softmax"})


class TestDataConfigScaling:
    """The [0, 1] scaling switches and their authoritative bounds."""

    def test_scaling_flags_default_to_on(self) -> None:
        config = DataConfig(param_bounds=BOUNDS)
        assert config.scale_metadata is True
        assert config.scale_params is True
        assert config.scale_time is True

    def test_flags_are_independent(self) -> None:
        config = DataConfig(
            scale_metadata=False,
            scale_params=False,
            scale_time=False,
            param_bounds=BOUNDS,
        )
        assert not (config.scale_metadata or config.scale_params or config.scale_time)

    def test_bounds_cover_the_whole_schema(self) -> None:
        bounds = DataConfig(param_bounds=BOUNDS).param_bounds.to_mapping()
        assert tuple(bounds) == PHYSICAL_COLUMNS
        assert set(TARGET_COLUMNS) <= set(bounds)
        assert bounds[TIME_COLUMN] == (0.0, 16.0)

    def test_bounds_are_declared_by_the_manifest(self) -> None:
        config = DataConfig(param_bounds={**BOUNDS, "pressure": (700.0, 780.0)})
        assert config.param_bounds.pressure == (700.0, 780.0)
        # The other columns keep whatever the manifest declared.
        assert config.param_bounds.airmass == (1.0, 2.0)

    def test_missing_bounds_are_rejected(self) -> None:
        """Every bound must be stated: Parameters carries no defaults."""
        incomplete = {k: v for k, v in BOUNDS.items() if k != "pressure"}
        with pytest.raises(ValueError):
            DataConfig(param_bounds=incomplete)

    def test_degenerate_bounds_are_rejected(self) -> None:
        """A constant parameter (min == max) carries no scale and is refused."""
        with pytest.raises(ValueError):
            DataConfig(param_bounds={**BOUNDS, "co": (0.14, 0.14)})

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError):
            DataConfig(param_bounds={**BOUNDS, "pressure": (900.0, 600.0)})

    def test_rejects_non_finite_bounds(self) -> None:
        with pytest.raises(ValueError):
            DataConfig(param_bounds={**BOUNDS, "humidity": (0.0, float("nan"))})


class TestTrainingConfig:
    def test_default_config(self) -> None:
        config = TrainingConfig()
        assert config.max_epochs == 100
        assert config.optimizer.learning_rate == 1e-4
