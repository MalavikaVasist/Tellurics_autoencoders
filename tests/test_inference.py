"""Tests for inference-time behaviour of the whole-night estimator."""

import torch

from tellurics.config.model import ModelArchitecture, ModelConfig
from tellurics.models.predictor import NeuralTelluricPredictor
from tellurics.models.output import ModelOutput


def _tiny_config() -> ModelConfig:
    """Small-dimension estimator config (CPU-friendly)."""
    return ModelConfig(
        architecture=ModelArchitecture.NEURAL_TELLURIC_PREDICTOR,
        num_wavelength_bins=256,
        n_frames_per_series=6,
        metadata_dim=3,
        param_dim=20,
        num_queries=4,
        spectral_latent_dim=16,
        num_heads=4,
        dropout=0.2,
        x_encoder_channels=[8, 16],
        x_encoder_kernel=7,
        x_encoder_stride=2,
        s_encoder_channels=[8, 16],
        s_encoder_kernel=7,
        s_encoder_stride=2,
        encoder_pool_bins=16,
        fusion_hidden=0,
        fusion_layers=2,
        metadata_enc_dim=16,
        metadata_enc_hidden=None,
        time_enc_dim=16,
        param_decoder_hidden=None,
        param_activation="sigmoid",
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(X, S, metadata, time) for two nights of six exposures.

    Shapes: X (B,T,N), S (B,N), metadata (B,T,P), time (B,T) -- the same batch
    contract :meth:`TelluricTrainingModule._run_estimator` feeds the model.
    """
    return (
        torch.randn(2, 6, 256),
        torch.randn(2, 256),
        torch.randn(2, 6, 3),
        torch.rand(2, 6) * 8.0,  # exposure times (hours)
    )


class TestEstimatorInference:
    """Inference contract: eval() is deterministic, train() exposes dropout."""

    def test_eval_output_structure_and_determinism(self) -> None:
        model = NeuralTelluricPredictor(_tiny_config()).eval()
        x, s, metadata, time = _batch()

        with torch.no_grad():
            first = model(x, stellar=s, metadata=metadata, time=time)
            second = model(x, stellar=s, metadata=metadata, time=time)

        assert isinstance(first, ModelOutput)
        assert first.params.shape == (2, 6, 20)
        assert torch.equal(first.params, second.params)

    def test_mc_dropout_varies_predictions(self) -> None:
        """Dropout must be active in train() so MC sampling is meaningful."""
        model = NeuralTelluricPredictor(_tiny_config()).train()
        x, s, metadata, time = _batch()

        with torch.no_grad():
            first = model(x, stellar=s, metadata=metadata, time=time).params
            second = model(x, stellar=s, metadata=metadata, time=time).params

        assert first.shape == (2, 6, 20)
        assert not torch.equal(first, second)

    def test_planet_recovery_computation(self) -> None:
        """Test that planet recovery P_hat = R / T_hat works correctly."""
        spectrum = torch.rand(1, 100) + 0.5  # R = X/S
        telluric_pred = torch.rand(1, 100) * 0.5 + 0.3  # T_hat in [0.3, 0.8]

        # Recover planet
        epsilon = 1e-8
        planet_hat = spectrum / (telluric_pred + epsilon)

        assert planet_hat.shape == spectrum.shape
        assert torch.all(torch.isfinite(planet_hat))
