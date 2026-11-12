"""Tests for configuration validation."""

import pytest
from pydantic import ValidationError

from tellurics.config import CNNAutoencoderConfig, DataConfig, ExperimentConfig, TelluricActivation
from tellurics.config.training import TrainingConfig
from tests._bounds import BOUNDS

CNN_KWARGS = {
    "architecture": "cnn_autoencoder",
    "num_wavelength_bins": 256,
    "encoder_channels": [8, 16],
    "kernel_size": 7,
    "stride": 2,
    "bottleneck_channels": 4,
    "metadata_dim": 2,
    "metadata_enc_dim": 8,
    "latent_dim": 16,
    "dropout": 0.1,
    "telluric_activation": "optical_depth",
}


class TestModelConfig:
    def test_explicit_config(self) -> None:
        config = CNNAutoencoderConfig(**CNN_KWARGS)
        assert config.num_wavelength_bins == 256
        assert config.telluric_activation is TelluricActivation.OPTICAL_DEPTH

    def test_rejects_unknown_activation(self) -> None:
        with pytest.raises(ValidationError):
            CNNAutoencoderConfig(**{**CNN_KWARGS, "telluric_activation": "relu"})

    def test_rejects_unknown_architecture(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TELLURICS_DATA", str(tmp_path))
        with pytest.raises(ValidationError):
            ExperimentConfig(model={**CNN_KWARGS, "architecture": "neural_telluric_predictor"},
                             data={"metadata_bounds": BOUNDS})

    def test_metadata_dim_must_match_metadata_columns(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TELLURICS_DATA", str(tmp_path))
        with pytest.raises(ValidationError, match="metadata_dim"):
            ExperimentConfig(model={**CNN_KWARGS, "metadata_dim": 3},
                             data={"metadata_bounds": BOUNDS})


class TestDataConfig:
    @pytest.fixture(autouse=True)
    def _root(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("TELLURICS_DATA", str(tmp_path))

    def test_defaults(self) -> None:
        config = DataConfig(metadata_bounds=BOUNDS)
        assert config.metadata_columns == ["airmass", "PWV"]
        assert config.target_key == "flux" and config.observed_key is None
        assert config.scale_metadata and config.noise_level_mean == 0.0

    def test_metadata_bounds_are_required(self) -> None:
        with pytest.raises(ValidationError, match="metadata_bounds"):
            DataConfig()

    def test_every_metadata_column_needs_bounds(self) -> None:
        with pytest.raises(ValidationError, match="PWV"):
            DataConfig(metadata_bounds={"airmass": (1.0, 2.0)})

    def test_bounds_must_be_ordered_and_finite(self) -> None:
        with pytest.raises(ValidationError, match="min < max"):
            DataConfig(metadata_bounds={**BOUNDS, "PWV": (25.0, 0.0)})
        with pytest.raises(ValidationError, match="min < max"):
            DataConfig(metadata_bounds={**BOUNDS, "PWV": (0.0, float("inf"))})

    def test_duplicate_metadata_columns_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            DataConfig(metadata_columns=["airmass", "airmass"], metadata_bounds=BOUNDS)

    def test_split_fractions(self) -> None:
        with pytest.raises(ValidationError, match="less than 1"):
            DataConfig(metadata_bounds=BOUNDS, train_fraction=0.9, val_fraction=0.1)


class TestTrainingConfig:
    def test_defaults_construct(self) -> None:
        assert TrainingConfig().max_epochs > 0
