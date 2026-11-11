"""Tests for the whole-night telluric time-series estimator (NeuralTelluricPredictor)."""

import os

import pytest
import torch
import torch.nn.functional as F

from tellurics.config.model import ModelConfig, TelluricActivation
from tellurics.models.multiple_frames.decoders import TelluricDecoder
from tellurics.models.multiple_frames.predictor import NeuralTelluricPredictor
from tellurics.models.output import ModelOutput
from tellurics.utils.registry import ModelRegistry


def _config(
    n_wave: int = 1024,
    t: int = 12,
    q: int = 8,
    d: int = 32,
    metadata_dim: int = 3,
    heads: int = 4,
    telluric_activation: str = "optical_depth",
) -> ModelConfig:
    """Small dims so the tests stay fast on CPU (a repo-wide ModelConfig).

    ``ModelConfig`` has no defaults, so every field is stated here.
    """
    return ModelConfig(
        architecture="neural_telluric_predictor",
        num_wavelength_bins=n_wave,
        n_frames_per_series=t,
        metadata_dim=metadata_dim,
        num_queries=q,
        spectral_latent_dim=d,
        num_heads=heads,
        dropout=0.0,
        x_encoder_channels=[8, 16, 32],
        x_encoder_kernel=7,
        x_encoder_stride=2,
        s_encoder_channels=[8, 16, 32],
        s_encoder_kernel=7,
        s_encoder_stride=2,
        encoder_pool_bins=16,
        fusion_hidden=0,
        fusion_layers=2,
        metadata_enc_dim=16,
        metadata_enc_hidden=None,
        time_enc_dim=16,
        telluric_decoder_hidden=None,
        telluric_activation=telluric_activation,
    )


class TestNeuralTelluricPredictorShapes:
    def test_shapes(self) -> None:
        cfg = _config()
        model = NeuralTelluricPredictor(cfg)

        observed = torch.randn(2, cfg.n_frames_per_series, cfg.num_wavelength_bins)
        stellar = torch.randn(2, cfg.num_wavelength_bins)
        metadata = torch.randn(2, cfg.n_frames_per_series, cfg.metadata_dim)
        time = torch.randn(2, cfg.n_frames_per_series)

        out = model(observed, stellar=stellar, metadata=metadata, time=time)

        assert isinstance(out, ModelOutput)
        assert out.telluric is not None
        assert out.telluric.shape == (
            2, cfg.n_frames_per_series, cfg.num_wavelength_bins,
        )
        assert out.latent.shape == (2, cfg.num_queries * cfg.spectral_latent_dim)
        assert out.attention_weights.shape == (
            2, cfg.num_queries, cfg.n_frames_per_series,
        )

        # intermediate features have exactly the documented shapes
        feats = out.intermediate_features
        assert feats["z"].shape == (
            2, cfg.n_frames_per_series, cfg.spectral_latent_dim,
        )
        assert feats["metadata_code"].shape == (
            2, cfg.n_frames_per_series, cfg.metadata_enc_dim,
        )
        assert feats["time_code"].shape == (
            2, cfg.n_frames_per_series, cfg.time_enc_dim,
        )
        assert feats["tokens"].shape == (
            2, cfg.n_frames_per_series,
            cfg.spectral_latent_dim + cfg.metadata_enc_dim + cfg.time_enc_dim,
        )
        assert feats["l"].shape == (
            2, cfg.num_queries, cfg.spectral_latent_dim,
        )
        assert feats["h_x"].shape == (
            2, cfg.num_queries * cfg.spectral_latent_dim,
        )
        assert feats["h_s"].shape == (2, cfg.spectral_latent_dim)
        assert feats["fusion"].shape == (
            2,
            cfg.num_queries * cfg.spectral_latent_dim + cfg.spectral_latent_dim,
        )
        assert feats["h_fused"].shape == (
            2, cfg.num_queries * cfg.spectral_latent_dim,
        )
        assert feats["latent"].shape == (
            2, cfg.num_queries, cfg.spectral_latent_dim,
        )
        assert feats["temporal"].shape == (
            2, cfg.n_frames_per_series, cfg.spectral_latent_dim,
        )
        assert feats["telluric_pred"].shape == (
            2, cfg.n_frames_per_series, cfg.num_wavelength_bins,
        )

    def test_missing_inputs_raise(self) -> None:
        """stellar, metadata and time are required: there is no optional path."""
        cfg = _config()
        model = NeuralTelluricPredictor(cfg)
        observed = torch.randn(2, cfg.n_frames_per_series, cfg.num_wavelength_bins)
        stellar = torch.randn(2, cfg.num_wavelength_bins)
        metadata = torch.randn(2, cfg.n_frames_per_series, cfg.metadata_dim)
        hours = torch.rand(2, cfg.n_frames_per_series)

        with pytest.raises(AttributeError):  # no time -> TimeEncoder(None)
            model(observed, stellar=stellar, metadata=metadata)
        with pytest.raises(NameError):       # no metadata -> code left unbound
            model(observed, stellar=stellar, time=hours)
        with pytest.raises(AttributeError):  # no stellar -> None.shape
            model(observed, metadata=metadata, time=hours)

    def test_unbatched_stellar_raises(self) -> None:
        """S must be (B, N); a bare (N,) spectrum is not broadcast."""
        cfg = _config()
        model = NeuralTelluricPredictor(cfg)
        observed = torch.randn(4, cfg.n_frames_per_series, cfg.num_wavelength_bins)
        with pytest.raises(ValueError):
            model(
                observed,
                stellar=torch.randn(cfg.num_wavelength_bins),  # (N,)
                metadata=torch.randn(4, 12, cfg.metadata_dim),
                time=torch.rand(4, 12),
            )

    def test_dimension_mismatch_raises(self) -> None:
        model = NeuralTelluricPredictor(_config(t=12))
        with pytest.raises(ValueError):
            model(torch.randn(2, 10, 1024))  # wrong number of exposures

    def test_arbitrary_batch_sizes(self) -> None:
        cfg = _config()
        model = NeuralTelluricPredictor(cfg)
        for batch in (1, 2, 5):
            out = model(
                torch.randn(batch, 12, 1024),
                stellar=torch.randn(batch, 1024),
                metadata=torch.randn(batch, 12, cfg.metadata_dim),
                time=torch.rand(batch, 12),
            )
            assert out.telluric.shape == (batch, 12, cfg.num_wavelength_bins)

    @pytest.mark.skipif(
        os.environ.get("TELLURIC_SLOW_TESTS") != "1",
        reason="single full-resolution (N=51556, T=73) forward is slow; "
               "set TELLURIC_SLOW_TESTS=1",
    )
    def test_full_problem_shape(self) -> None:
        """Sanity shape at the real problem dims (small batch, single pass)."""
        cfg = _config(n_wave=51556, t=73, q=16, d=64, metadata_dim=3, heads=4)
        model = NeuralTelluricPredictor(cfg)
        batch = 1
        out = model(
            torch.randn(batch, 73, 51556),
            stellar=torch.randn(batch, 51556),
            metadata=torch.randn(batch, 73, 3),
            time=torch.rand(batch, 73),
        )
        assert out.telluric.shape == (batch, 73, 51556)
        assert out.latent.shape == (batch, 1024)
        assert out.attention_weights.shape == (batch, 16, 73)


class TestGradientFlow:
    def test_gradient_flow_through_all_parameters(self) -> None:
        cfg = _config()
        model = NeuralTelluricPredictor(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        observed = torch.randn(2, 12, 1024)
        stellar = torch.randn(2, 1024)
        metadata = torch.randn(2, 12, cfg.metadata_dim)
        hours = torch.rand(2, 12)                   # exposure times
        target = torch.rand(2, 12, cfg.num_wavelength_bins)  # random transmission

        opt.zero_grad()
        out = model(observed, stellar=stellar, metadata=metadata, time=hours)
        loss = F.mse_loss(out.telluric, target)
        loss.backward()
        opt.step()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert len(grads) > 0
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert torch.isfinite(loss)

    def test_parameter_counts_and_components(self) -> None:
        model = NeuralTelluricPredictor(_config())
        counts = model.parameter_counts()
        assert set(counts) == {
            "SpectralEncoder (X)", "MetadataEncoder", "TimeEncoder",
            "PreTemporal", "TemporalPerceiver", "StellarEncoder (S)",
            "FusionMLP", "TemporalDecoder", "TelluricDecoder",
        }
        assert all(v > 0 for v in counts.values())
        assert sum(counts.values()) == model.count_parameters()


def _synthetic_night(n_wave: int, n_frames: int, rng: torch.Generator):
    """One smooth synthetic night: X = T * S with airmass-scaled tellurics.

    Returns (observed, stellar, telluric, metadata) where ``metadata`` is the
    per-exposure airmass code ``(T, 3)`` -- the physical driver of the drift.
    """
    device = "cpu"
    w = torch.linspace(0.0, 1.0, n_wave, device=device).unsqueeze(0)  # (1, N)
    noise = torch.sin(2 * torch.pi * torch.arange(n_wave, dtype=torch.float32, device=device).unsqueeze(0) * 3.0 / n_wave)
    stellar = 1.0 + 0.4 * noise + 0.2 * torch.cos(2 * torch.pi * w * 6.0)
    stellar = stellar / stellar.mean()

    tau = 0.05 + 0.25 * (0.5 + 0.5 * torch.sin(2 * torch.pi * w * 8.0))
    tau = tau * (0.7 + 0.3 * torch.sin(2 * torch.pi * w * 37.0))
    airm = torch.linspace(1.05, 2.0, n_frames, device=device).unsqueeze(1)  # (T,1)
    tell = torch.exp(-tau * airm)                                        # (T,N)
    observed = stellar * tell                                             # (T,N)
    airm_flat = airm.squeeze(1)                                           # (T,)
    metadata = torch.stack(
        [airm_flat, airm_flat ** 2, torch.ones_like(airm_flat)], dim=-1   # (T,3)
    )
    return observed, stellar, tell, metadata


class TestOverfitTiny:
    def test_fits_a_tiny_telluric_reconstruction(self) -> None:
        """The estimator must reduce MSE(T_tell, telluric_pred) on a tiny set."""
        cfg = _config(n_wave=512, t=12, q=8, d=32, heads=4)
        model = NeuralTelluricPredictor(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)

        nights = 4
        xs, ss, tells, metas = [], [], [], []
        g = torch.Generator().manual_seed(0)
        for _ in range(nights):
            x, s, tell, meta = _synthetic_night(
                cfg.num_wavelength_bins, cfg.n_frames_per_series, g
            )
            xs.append(x); ss.append(s.squeeze(0)); tells.append(tell)
            metas.append(meta)
        x = torch.stack(xs)               # (B, T, N)
        s = torch.stack(ss)               # (B, N)
        metadata = torch.stack(metas)     # (B, T, P) per-exposure metadata
        telluric = torch.stack(tells)     # (B, T, N) true transmission
        hours = torch.rand(nights, cfg.n_frames_per_series)

        losses = []
        for _ in range(120):
            opt.zero_grad()
            out = model(x, stellar=s, metadata=metadata, time=hours)
            loss = F.mse_loss(out.telluric, telluric)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


class TestTelluricActivation:
    """The telluric head is the only bounded output of the model."""

    @pytest.mark.parametrize("activation", ["optical_depth", "sigmoid"])
    def test_bounded_heads_stay_within_the_unit_interval(self, activation) -> None:
        """Even with saturating inputs, the predicted transmission is in [0, 1]."""
        cfg = _config(telluric_activation=activation)
        model = NeuralTelluricPredictor(cfg).eval()
        with torch.no_grad():
            out = model(
                torch.randn(2, 12, 1024) * 25.0,
                stellar=torch.randn(2, 1024),
                metadata=torch.randn(2, 12, cfg.metadata_dim) * 25.0,
                time=torch.rand(2, 12) * 25.0,
            )
        assert float(out.telluric.min()) >= 0.0
        assert float(out.telluric.max()) <= 1.0

    def test_head_matches_the_documented_transform(self) -> None:
        """exp(-softplus(raw)), sigmoid(raw) or raw, depending on the head."""
        code = torch.randn(3, 8)
        tau_head = TelluricDecoder(8, 64, activation="optical_depth")
        sigmoid_head = TelluricDecoder(8, 64, activation="sigmoid")
        linear_head = TelluricDecoder(8, 64, activation="none")

        def raw(head):
            return head.basis(head.mlp(code))

        assert torch.allclose(tau_head(code), torch.exp(-F.softplus(raw(tau_head))))
        assert torch.allclose(sigmoid_head(code), torch.sigmoid(raw(sigmoid_head)))
        assert torch.allclose(linear_head(code), raw(linear_head))
        assert bool(((tau_head(code) > 0) & (tau_head(code) <= 1)).all())

    def test_default_head_is_optical_depth(self) -> None:
        assert TelluricDecoder(8, 64).activation is TelluricActivation.OPTICAL_DEPTH

    def test_any_leading_dims(self) -> None:
        head = TelluricDecoder(8, 64)
        assert head(torch.randn(2, 5, 8)).shape == (2, 5, 64)
        assert head(torch.randn(7, 8)).shape == (7, 64)

    def test_initial_prediction_is_near_transparent(self) -> None:
        """Bounded heads start near T ~ 1 (a clear sky), not at T = 0.5."""
        for activation in ("optical_depth", "sigmoid"):
            head = TelluricDecoder(8, 64, activation=activation)
            assert float(head(torch.zeros(1, 8)).mean()) > 0.9


def _repo_model_config(n_wave: int = 1024, t: int = 12) -> "ModelConfig":
    """Small repo-wide ModelConfig targeting the neural_telluric_predictor arch."""
    return _config(n_wave=n_wave, t=t, q=8, d=32, metadata_dim=3, heads=4)


class TestRepoIntegration:
    """Wiring of NeuralTelluricPredictor into ModelConfig / ModelRegistry."""

    def test_registry(self) -> None:
        assert "neural_telluric_predictor" in ModelRegistry.list_models()
        assert ModelRegistry.get("neural_telluric_predictor") is NeuralTelluricPredictor

    def test_construct_from_repo_model_config(self) -> None:
        cfg = _repo_model_config()
        model = NeuralTelluricPredictor(cfg)  # ModelConfig path
        out = model(
            torch.randn(2, 12, 1024),
            stellar=torch.randn(2, 1024),
            metadata=torch.randn(2, 12, 3),
            time=torch.rand(2, 12),
        )
        assert out.telluric.shape == (2, 12, cfg.num_wavelength_bins)


class TestNightLevelSplit:
    def test_split_is_on_nights_only(self) -> None:
        from tellurics.data.splits import split_night_indices

        train, val, test = split_night_indices(100, 0.8, 0.1, seed=7)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        # Disjoint sets of night ids.
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
