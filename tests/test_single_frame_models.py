"""Tests for the single-frame autoencoders and the single-frame data path.

Covers the three single-frame architectures (``cnn_autoencoder``,
``transformer_autoencoder``, ``specpt``), their per-architecture configs, the
:class:`TelluricFrameDataset` and one Lightning train/val step per model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch.utils.data import DataLoader

from tellurics.config import (
    CNNAutoencoderConfig,
    ExperimentConfig,
    ModelConfig,
    SpecPTConfig,
    TransformerAutoencoderConfig,
    load_config,
)
from tellurics.config.training import TrainingConfig
from tellurics.data.datasets import TelluricFrameDataset, TelluricTimeseriesDataset
from tellurics.models import CNNAutoencoder, SpecPT, TransformerAutoencoder
from tellurics.training.module import TelluricTrainingModule
from tellurics.utils.registry import ModelRegistry
from tests.test_scaling import N_FRAMES, N_WAVE, _FakeStellarPool, _write_night_file

REPO = Path(__file__).resolve().parents[1]
N = 1000   # small, deliberately not a power of two (exercises padding/cropping)


# --------------------------------------------------------------------------- #
# Tiny configs
# --------------------------------------------------------------------------- #
def _cnn(n: int = N, **kw) -> CNNAutoencoderConfig:
    return CNNAutoencoderConfig(**{
        "architecture": "cnn_autoencoder", "num_wavelength_bins": n,
        "encoder_channels": [4, 8, 8], "kernel_size": 5, "stride": 2,
        "bottleneck_channels": 2, "latent_dim": 16, "dropout": 0.0,
        "telluric_activation": "optical_depth", **kw,
    })


def _tae(n: int = N, **kw) -> TransformerAutoencoderConfig:
    return TransformerAutoencoderConfig(**{
        "architecture": "transformer_autoencoder", "num_wavelength_bins": n,
        "patch_size": 32, "d_model": 32, "num_heads": 4,
        "num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 64,
        "num_latents": 4, "latent_dim": 8, "dropout": 0.0,
        "telluric_activation": "optical_depth", **kw,
    })


def _specpt(n: int = N, **kw) -> SpecPTConfig:
    return SpecPTConfig(**{
        "architecture": "specpt", "num_wavelength_bins": n,
        "stem_channels": [4, 8, 8], "stem_kernels": [9, 5, 3], "token_stride": 2,
        "d_model": 32, "num_heads": 4, "num_encoder_layers": 1,
        "num_decoder_layers": 1, "dim_feedforward": 64, "head_hidden": 32,
        "dropout": 0.0, "telluric_activation": "optical_depth", **kw,
    })


BUILDERS = {
    "cnn_autoencoder": (_cnn, CNNAutoencoder),
    "transformer_autoencoder": (_tae, TransformerAutoencoder),
    "specpt": (_specpt, SpecPT),
}
ARCHS = list(BUILDERS)


def _model(arch: str, n: int = N, **kw):
    make_cfg, cls = BUILDERS[arch]
    return cls(make_cfg(n, **kw))


# --------------------------------------------------------------------------- #
# Configs: one class per architecture, selected by ``architecture``
# --------------------------------------------------------------------------- #
class TestConfigs:
    @pytest.mark.parametrize(
        "fname, cls, single_frame",
        [
            ("multiple_frames/experiment.yaml", ModelConfig, False),
            ("single_frame/experiment_cnn_ae.yaml", CNNAutoencoderConfig, True),
            ("single_frame/experiment_transformer_ae.yaml", TransformerAutoencoderConfig, True),
            ("single_frame/experiment_specpt.yaml", SpecPTConfig, True),
        ],
    )
    def test_each_manifest_selects_its_own_model_config(
        self, fname, cls, single_frame
    ) -> None:
        cfg = load_config(REPO / "experiments" / fname)
        assert type(cfg.model) is cls
        assert cfg.model.single_frame is single_frame
        assert cfg.model.num_wavelength_bins == 51557

    def test_manifest_folders_mirror_the_model_packages(self) -> None:
        """experiments/multiple_frames/ -> whole night, single_frame/ -> one exposure."""
        for folder, single in (("multiple_frames", False), ("single_frame", True)):
            manifests = sorted((REPO / "experiments" / folder).glob("*.yaml"))
            assert manifests, folder
            for path in manifests:
                assert load_config(path).model.single_frame is single, path.name

    def test_runs_do_not_share_a_run_dir(self) -> None:
        names = ["multiple_frames/experiment", "single_frame/experiment_cnn_ae",
                 "single_frame/experiment_transformer_ae", "single_frame/experiment_specpt"]
        dirs = {load_config(REPO / "experiments" / f"{n}.yaml").run_dir for n in names}
        assert len(dirs) == 4

    def test_fields_of_another_architecture_are_rejected(self) -> None:
        """A CNN config must not silently accept transformer fields."""
        raw = load_config(REPO / "experiments" / "single_frame" / "experiment_cnn_ae.yaml").model_dump(mode="json")
        raw["model"]["patch_size"] = 128          # transformer-only field
        raw["model"].pop("latent_dim")            # and a missing CNN field
        with pytest.raises(ValidationError):
            ExperimentConfig(**raw)

    def test_unknown_architecture_is_rejected(self) -> None:
        raw = load_config(REPO / "experiments" / "single_frame" / "experiment_specpt.yaml").model_dump(mode="json")
        raw["model"]["architecture"] = "unet"
        with pytest.raises(ValidationError):
            ExperimentConfig(**raw)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_registered(self, arch) -> None:
        assert ModelRegistry.get(arch) is BUILDERS[arch][1]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TestSingleFrameModels:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_shape_bounds_and_inputs(self, arch) -> None:
        model = _model(arch).eval()
        x = torch.rand(3, 1, N) + 0.5
        with torch.no_grad():
            out = model(x, stellar=torch.rand(3, N),
                        metadata=torch.rand(3, 1, 3), time=torch.rand(3, 1))
            flat = model(x[:, 0])                 # (B, N) is accepted too
        assert out.telluric.shape == (3, 1, N)
        assert torch.allclose(out.telluric, flat.telluric)
        assert bool(((out.telluric > 0) & (out.telluric <= 1)).all())
        assert out.latent is not None and out.latent.shape[0] == 3

    @pytest.mark.parametrize("arch", ARCHS)
    def test_rejects_whole_nights(self, arch) -> None:
        with pytest.raises(ValueError, match="single-frame"):
            _model(arch)(torch.rand(2, 5, N))

    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_parameter_gets_a_gradient(self, arch) -> None:
        model = _model(arch)
        loss = F.mse_loss(model(torch.rand(2, 1, N)).telluric, torch.rand(2, 1, N))
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    @pytest.mark.parametrize("arch", ARCHS)
    def test_batch_items_are_independent(self, arch) -> None:
        """Eval mode: changing one spectrum must not change another's output."""
        model = _model(arch).eval()
        x = torch.rand(2, 1, N)
        x2 = x.clone(); x2[1] = torch.rand(1, N)
        with torch.no_grad():
            a, b = model(x).telluric, model(x2).telluric
        assert torch.allclose(a[0], b[0], atol=1e-6)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_fits_a_tiny_set(self, arch) -> None:
        """MSE(T_tell, telluric_pred) must drop well below its start."""
        torch.manual_seed(0)
        n = 512
        wl = torch.linspace(0, 1, n)
        lines = torch.zeros(n)
        for c in (0.13, 0.31, 0.47, 0.62, 0.88):
            lines += torch.exp(-0.5 * ((wl - c) / 0.004) ** 2)
        airmass = torch.linspace(1.0, 2.0, 8)[:, None]
        tell = torch.exp(-1.5 * airmass * lines)                  # (8, n)
        star = 1.0 + 0.2 * torch.sin(6 * wl)
        x = (tell * star).unsqueeze(1)                            # (8, 1, n)
        target = tell.unsqueeze(1)

        model = _model(arch, n)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(150):
            opt.zero_grad()
            loss = F.mse_loss(model(x).telluric, target)
            loss.backward(); opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])

    def test_specpt_tokens_tile_the_spectrum(self) -> None:
        m = _model("specpt")
        assert m.downsample == 2 ** 3 * 2 * 2
        assert m.n_tokens * m.downsample >= N
        assert m.n_tokens * m.downsample - N < m.downsample

    def test_specpt_full_resolution_geometry(self) -> None:
        """The repo manifest gives 403 tokens of 128 samples at N = 51557."""
        cfg = load_config(REPO / "experiments" / "single_frame" / "experiment_specpt.yaml").model
        m = SpecPT(cfg)
        assert (m.n_tokens, m.downsample, m.pad) == (403, 128, 27)

    def test_cnn_decoder_covers_the_grid(self) -> None:
        m = _model("cnn_autoencoder")
        assert m.dec_len >= N
        cfg = load_config(REPO / "experiments" / "single_frame" / "experiment_cnn_ae.yaml").model
        full = CNNAutoencoder(cfg)
        assert full.enc_len == 1612 and full.dec_len == 51584


# --------------------------------------------------------------------------- #
# Single-frame data path
# --------------------------------------------------------------------------- #
@pytest.fixture()
def night_file(tmp_path):
    path = tmp_path / "nights.h5"
    _write_night_file(path)
    return path


def _frames(path, nights) -> TelluricFrameDataset:
    return TelluricFrameDataset(
        night_ids=nights, h5_path=path, pool=_FakeStellarPool(),
        star_assignment=np.zeros(8, dtype=int), n_frames=N_FRAMES,
    )


class TestFrameDataset:
    def test_one_sample_per_exposure_of_the_split_nights(self, night_file) -> None:
        ds = _frames(night_file, [0, 1])
        assert len(ds) == 2 * N_FRAMES
        seen = {(ds[i]["night_id"], ds[i]["frame_id"]) for i in range(len(ds))}
        assert seen == {(n, f) for n in (0, 1) for f in range(N_FRAMES)}

    def test_shapes_have_a_length_one_frame_axis(self, night_file) -> None:
        sample = _frames(night_file, [0])[1]
        assert sample["observed"].shape == (1, N_WAVE)
        assert sample["telluric"].shape == (1, N_WAVE)
        assert sample["stellar"].shape == (N_WAVE,)
        assert sample["theta"]["time"].shape == (1,)
        assert sample["theta"]["metadata"].shape == (1, 3)
        assert torch.allclose(
            sample["observed"], sample["telluric"] * sample["stellar"][None]
        )

    def test_matches_the_corresponding_night_frame(self, night_file) -> None:
        night = TelluricTimeseriesDataset(
            [1], night_file, _FakeStellarPool(), np.zeros(8, dtype=int)
        )[0]
        frame = _frames(night_file, [1])[2]
        for key in ("observed", "telluric"):
            assert torch.equal(frame[key][0], night[key][2])
        assert torch.equal(frame["theta"]["metadata"][0], night["theta"]["metadata"][2])

    @pytest.mark.parametrize("arch", ARCHS)
    def test_one_lightning_train_and_val_step(self, night_file, arch) -> None:
        loader = DataLoader(_frames(night_file, [0, 1]), batch_size=4)
        module = TelluricTrainingModule(
            _tiny_for_n_wave(arch), TrainingConfig(max_epochs=1, precision="32"),
        )
        trainer = pl.Trainer(
            fast_dev_run=True, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            accelerator="cpu", devices=1,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        assert trainer.state.status == "finished"
        with torch.no_grad():
            out = module.eval()._run_estimator(next(iter(loader)))
        assert out.telluric.shape == (4, 1, N_WAVE)


def _tiny_for_n_wave(arch: str):
    """Configs that fit the 8-sample synthetic HDF5 of the scaling tests."""
    if arch == "cnn_autoencoder":
        return _cnn(N_WAVE, encoder_channels=[4, 4], kernel_size=3)
    if arch == "transformer_autoencoder":
        return _tae(N_WAVE, patch_size=2)
    return _specpt(N_WAVE, stem_channels=[4], stem_kernels=[3], token_stride=1)
