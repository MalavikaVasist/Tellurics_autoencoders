"""Tests for model architectures."""

import torch

from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


class TestModelOutput:
    def test_default_output_is_empty(self) -> None:
        output = ModelOutput()
        assert output.params is None
        assert output.latent is None
        assert output.attention_weights is None
        assert output.intermediate_features is None

    def test_full_output(self) -> None:
        output = ModelOutput(
            params=torch.randn(4, 8, 16),
            latent=torch.randn(4, 128),
            attention_weights=torch.randn(4, 16, 8),
            intermediate_features={"tokens": torch.randn(4, 8, 64)},
        )
        assert output.params is not None
        assert output.params.shape == (4, 8, 16)
        assert output.latent is not None
        assert output.intermediate_features is not None


class TestModelRegistry:
    def test_registered_models(self) -> None:
        assert ModelRegistry.list_models() == ["neural_telluric_predictor"]

    def test_get_model(self) -> None:
        from tellurics.models.predictor import NeuralTelluricPredictor

        assert (
            ModelRegistry.get("neural_telluric_predictor")
            is NeuralTelluricPredictor
        )

    def test_get_unknown_model(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            ModelRegistry.get("nonexistent_model")
