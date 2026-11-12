"""Tests for model architectures."""

import torch

from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class TestModelOutput:
    def test_default_output_is_empty(self) -> None:
        output = ModelOutput()
        assert output.telluric is None
        assert output.latent is None
        assert output.attention_weights is None
        assert output.intermediate_features is None

    def test_full_output(self) -> None:
        output = ModelOutput(
            telluric=torch.randn(4, 256),
            latent=torch.randn(4, 128),
            attention_weights=torch.randn(4, 16, 8),
            intermediate_features={"tokens": torch.randn(4, 8, 64)},
        )
        assert output.telluric is not None
        assert output.telluric.shape == (4, 256)
        assert output.latent is not None
        assert output.intermediate_features is not None


class TestModelRegistry:
    def test_registered_models(self) -> None:
        assert sorted(ModelRegistry.list_models()) == [
            "cnn_autoencoder",
            "specpt",
            "transformer_autoencoder",
        ]

    def test_get_model(self) -> None:
        from tellurics.models import CNNAutoencoder

        assert ModelRegistry.get("cnn_autoencoder") is CNNAutoencoder

    def test_get_unknown_model(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")
