"""Tests for the per-observation autoencoders and their data path.

One observation in, one telluric spectrum out: a batch is ``observed (B, N)``
+ ``metadata (B, M)`` (+ ``wavelength (B, N)``) and the models return
``telluric (B, N)``. The exposures of all nights are pooled into observations.

Covers the three single-frame architectures (``cnn_autoencoder``,
``transformer_autoencoder``, ``specpt``), their per-architecture configs, the
metadata (airmass, pwv) and wavelength inputs of the CNN / transformer
autoencoders, the :class:`TelluricFrameDataset` and one Lightning train/val
step per model.
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
M = 2      # metadata: airmass, pwv


# --------------------------------------------------------------------------- #
# Tiny configs
# --------------------------------------------------------------------------- #
def _cnn(n: int = N, **kw) -> CNNAutoencoderConfig:
    return CNNAutoencoderConfig(**{
        "architecture": "cnn_autoencoder", "num_wavelength_bins": n,
        "encoder_channels": [4, 8, 8], "kernel_size": 5, "stride": 2,
        "bottleneck_channels": 2, "latent_dim": 16, "dropout": 0.0,
        "metadata_dim": M, "metadata_enc_dim": 4,
        "telluric_activation": "optical_depth", **kw,
    })


def _tae(n: int = N, **kw) -> TransformerAutoencoderConfig:
    return TransformerAutoencoderConfig(**{
        "architecture": "transformer_autoencoder", "num_wavelength_bins": n,
        "patch_size": 32, "d_model": 32, "num_heads": 4,
        "num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 64,
        "num_latents": 4, "latent_dim": 8, "dropout": 0.0,
        "metadata_dim": M, "metadata_enc_dim": 4, "wavelength_frequencies": 4,
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


def _inputs(b: int, n: int = N, seed: int = 0):
    """observed (B, n), metadata (B, M), wavelength (B, n)."""
    g = torch.Generator().manual_seed(seed)
    observed = torch.rand(b, n, generator=g) + 0.5
    metadata = torch.rand(b, M, generator=g)
    wavelength = torch.linspace(2000.0, 2100.0, n).expand(b, n)
    return observed, metadata, wavelength


def _run(model, observed, metadata, wavelength):
    return model(observed, metadata=metadata, wavelength=wavelength)


# --------------------------------------------------------------------------- #
# Configs: one class per architecture, selected by ``architecture``
# --------------------------------------------------------------------------- #
class TestConfigs:
    @pytest.mark.parametrize(
        "fname, cls, per_frame",
        [
            ("multiple_frames/experiment.yaml", ModelConfig, False),
            ("single_frame/experiment_cnn_ae.yaml", CNNAutoencoderConfig, True),
            ("single_frame/experiment_transformer_ae.yaml", TransformerAutoencoderConfig, True),
            ("single_frame/experiment_specpt.yaml", SpecPTConfig, True),
        ],
    )
    def test_each_manifest_selects_its_own_model_config(
        self, fname, cls, per_frame
    ) -> None:
        cfg = load_config(REPO / "experiments" / fname)
        assert type(cfg.model) is cls
        # whole nights for the whole-night model, single observations otherwise
        assert cfg.model.single_frame is per_frame
        # whole night: full 51557 grid; per-observation models: 2048-sample data
        assert cfg.model.num_wavelength_bins == (2048 if per_frame else 51557)

    @pytest.mark.parametrize(
        "fname", ["single_frame/experiment_cnn_ae.yaml",
                  "single_frame/experiment_transformer_ae.yaml"],
    )
    def test_autoencoders_take_airmass_and_pwv(self, fname) -> None:
        cfg = load_config(REPO / "experiments" / fname)
        assert cfg.data.metadata_columns == ["airmass", "pwv"]
        assert cfg.model.metadata_dim == 2
        assert cfg.data.param_bounds.pwv is not None

    def test_whole_night_keeps_its_three_metadata_columns(self) -> None:
        cfg = load_config(REPO / "experiments" / "multiple_frames" / "experiment.yaml")
        assert cfg.data.metadata_columns == ["pressure", "temperature", "humidity"]
        assert cfg.model.metadata_dim == 3

    def test_metadata_dim_must_match_metadata_columns(self) -> None:
        raw = load_config(
            REPO / "experiments" / "single_frame" / "experiment_cnn_ae.yaml"
        ).model_dump(mode="json")
        raw["data"]["metadata_columns"] = ["airmass"]
        with pytest.raises(ValidationError, match="metadata_dim"):
            ExperimentConfig(**raw)

    def test_metadata_columns_need_bounds(self) -> None:
        raw = load_config(
            REPO / "experiments" / "single_frame" / "experiment_cnn_ae.yaml"
        ).model_dump(mode="json")
        raw["data"]["param_bounds"].pop("pwv")
        with pytest.raises(ValidationError, match="pwv"):
            ExperimentConfig(**raw)

    def test_manifest_folders_mirror_the_model_packages(self) -> None:
        """multiple_frames/ -> whole-night model, single_frame/ -> per-frame models."""
        for folder, per_frame in (("multiple_frames", False), ("single_frame", True)):
            manifests = sorted((REPO / "experiments" / folder).glob("*.yaml"))
            assert manifests, folder
            for path in manifests:
                assert load_config(path).model.single_frame is per_frame, path.name

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
    def test_one_observation_in_one_telluric_out(self, arch) -> None:
        model = _model(arch).eval()
        x, m, wl = _inputs(3)
        with torch.no_grad():
            out = model(x, stellar=torch.rand(3, N), metadata=m,
                        time=torch.rand(3), wavelength=wl)
            shared = model(x, metadata=m, wavelength=wl[0])        # (N,) grid
        assert out.telluric.shape == (3, N)                         # (B, N)
        assert torch.allclose(out.telluric, shared.telluric, atol=1e-6)
        assert bool(((out.telluric > 0) & (out.telluric <= 1)).all())
        assert out.latent is not None and out.latent.shape[0] == 3

    @pytest.mark.parametrize("arch", ARCHS)
    def test_rejects_anything_but_one_observation_per_row(self, arch) -> None:
        model = _model(arch)
        x, m, wl = _inputs(2)
        with pytest.raises(ValueError, match="one observation per row"):
            model(x.unsqueeze(1), metadata=m, wavelength=wl)        # (B, 1, N)
        with pytest.raises(ValueError, match="one observation per row"):
            model(torch.rand(2, 5, N), metadata=m, wavelength=wl)   # (B, T, N)

    @pytest.mark.parametrize("arch", ["cnn_autoencoder", "transformer_autoencoder"])
    def test_metadata_must_be_one_row_per_observation(self, arch) -> None:
        x, m, wl = _inputs(2)
        with pytest.raises(ValueError, match="metadata"):
            _model(arch)(x, metadata=torch.rand(3, M), wavelength=wl)

    @pytest.mark.parametrize("arch", ["cnn_autoencoder", "transformer_autoencoder"])
    def test_metadata_is_required_and_used(self, arch) -> None:
        model = _model(arch).eval()
        x, m, wl = _inputs(2)
        with pytest.raises(ValueError, match="metadata"):
            model(x, wavelength=wl)
        with torch.no_grad():
            a = _run(model, x, m, wl).telluric
            b = _run(model, x, m + 1.0, wl).telluric
        # untrained heads sit near the clear-sky template, so the change is
        # small; it must be non-zero.
        assert float((a - b).abs().max()) > 0, "output must depend on the metadata"

    def test_transformer_requires_and_uses_wavelength(self) -> None:
        model = _model("transformer_autoencoder").eval()
        x, m, wl = _inputs(2)
        with pytest.raises(ValueError, match="wavelength"):
            model(x, metadata=m)
        shifted = torch.linspace(2000.0, 2100.0, N) ** 2          # non-linear regrid
        with torch.no_grad():
            a = _run(model, x, m, wl).telluric
            b = _run(model, x, m, shifted.expand(2, N)).telluric
        assert float((a - b).abs().max()) > 0, "output must depend on the wavelengths"

    def test_transformer_wavelength_encoding_is_unit_free(self) -> None:
        """nm vs Angstrom grids give the same output (per-spectrum normalisation)."""
        model = _model("transformer_autoencoder").eval()
        x, m, wl = _inputs(2)
        with torch.no_grad():
            a = _run(model, x, m, wl).telluric
            b = _run(model, x, m, wl * 10.0).telluric
        assert torch.allclose(a, b, atol=1e-5)

    def test_specpt_ignores_metadata_and_wavelength(self) -> None:
        model = _model("specpt").eval()
        x, m, wl = _inputs(2)
        with torch.no_grad():
            a = model(x).telluric
            b = _run(model, x, m, wl).telluric
        assert torch.allclose(a, b)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_every_parameter_gets_a_gradient(self, arch) -> None:
        model = _model(arch)
        x, m, wl = _inputs(2)
        loss = F.mse_loss(_run(model, x, m, wl).telluric, torch.rand(2, N))
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    @pytest.mark.parametrize("arch", ARCHS)
    def test_batch_items_are_independent(self, arch) -> None:
        """Eval mode: changing one exposure must not change another's output."""
        model = _model(arch).eval()
        x, m, wl = _inputs(2)
        x2, m2 = x.clone(), m.clone()
        x2[1] = torch.rand(N); m2[1] = torch.rand(M)
        with torch.no_grad():
            a = _run(model, x, m, wl).telluric
            b = _run(model, x2, m2, wl).telluric
        assert torch.allclose(a[0], b[0], atol=1e-6)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_fits_a_tiny_set(self, arch) -> None:
        """MSE(T_tell, telluric_pred) must drop well below its start.

        The telluric depth is set by airmass * pwv, which the models get as
        metadata, and the observed spectrum is T * S.
        """
        torch.manual_seed(0)
        n, b = 512, 8
        wl = torch.linspace(0, 1, n)
        lines = torch.zeros(n)
        for c in (0.13, 0.31, 0.47, 0.62, 0.88):
            lines += torch.exp(-0.5 * ((wl - c) / 0.004) ** 2)
        airmass = torch.linspace(1.0, 2.0, b)[:, None]
        pwv = torch.linspace(0.5, 1.5, b).flip(0)[:, None]
        tell = torch.exp(-1.5 * airmass * pwv * lines)            # (b, n)
        star = 1.0 + 0.2 * torch.sin(6 * wl)
        x = tell * star                                           # (b, n)
        meta = torch.cat([airmass, pwv], dim=1)                   # (b, 2)
        target = tell
        grid = (2000.0 + 100.0 * wl).expand(b, n)

        model = _model(arch, n)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(150):
            opt.zero_grad()
            loss = F.mse_loss(_run(model, x, meta, grid).telluric, target)
            loss.backward(); opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.2 * losses[0], (losses[0], losses[-1])

    def test_specpt_tokens_tile_the_spectrum(self) -> None:
        m = _model("specpt")
        assert m.downsample == 2 ** 3 * 2 * 2
        assert m.n_tokens * m.downsample >= N
        assert m.n_tokens * m.downsample - N < m.downsample

    def test_repo_manifest_geometries_at_2048(self) -> None:
        """The single-frame manifests tile the 2048-sample spectrum exactly."""
        sf = REPO / "experiments" / "single_frame"
        spec = SpecPT(load_config(sf / "experiment_specpt.yaml").model)
        assert (spec.n_tokens, spec.downsample, spec.pad) == (128, 16, 0)
        cnn = CNNAutoencoder(load_config(sf / "experiment_cnn_ae.yaml").model)
        assert (cnn.enc_len, cnn.dec_len) == (64, 2048)
        tae = TransformerAutoencoder(load_config(sf / "experiment_transformer_ae.yaml").model)
        assert (tae.n_patches, tae.pad) == (128, 0)


# --------------------------------------------------------------------------- #
# Single-frame data path
# --------------------------------------------------------------------------- #
@pytest.fixture()
def night_file(tmp_path):
    path = tmp_path / "nights.h5"
    _write_night_file(path)
    return path


def _frames(path, nights, metadata_columns=("airmass", "pwv")) -> TelluricFrameDataset:
    return TelluricFrameDataset(
        night_ids=nights, h5_path=path, pool=_FakeStellarPool(),
        star_assignment=np.zeros(8, dtype=int), n_frames=N_FRAMES,
        metadata_columns=metadata_columns,
    )


class TestFrameDataset:
    def test_one_sample_per_exposure_of_the_split_nights(self, night_file) -> None:
        ds = _frames(night_file, [0, 1])
        assert len(ds) == 2 * N_FRAMES
        seen = {(ds[i]["night_id"], ds[i]["frame_id"]) for i in range(len(ds))}
        assert seen == {(n, f) for n in (0, 1) for f in range(N_FRAMES)}

    def test_one_observation_per_sample_without_frame_axis(self, night_file) -> None:
        sample = _frames(night_file, [0])[1]
        assert sample["observed"].shape == (N_WAVE,)
        assert sample["telluric"].shape == (N_WAVE,)
        assert sample["stellar"].shape == (N_WAVE,)
        assert sample["theta"]["time"].shape == ()
        assert sample["theta"]["metadata"].shape == (M,)              # airmass, pwv
        assert sample["wavelength"].shape == (N_WAVE,)
        assert torch.allclose(sample["observed"], sample["telluric"] * sample["stellar"])

    def test_matches_the_corresponding_night_frame(self, night_file) -> None:
        night = TelluricTimeseriesDataset(
            [1], night_file, _FakeStellarPool(), np.zeros(8, dtype=int),
            metadata_columns=("airmass", "pwv"),
        )[0]
        frame = _frames(night_file, [1])[2]
        for key in ("observed", "telluric"):
            assert torch.equal(frame[key], night[key][2])
        assert torch.equal(frame["theta"]["metadata"], night["theta"]["metadata"][2])
        assert torch.equal(frame["wavelength"], night["wavelength"])

    def test_metadata_columns_are_selectable(self, night_file) -> None:
        """airmass, pwv by name -- and the default three still work."""
        sample = _frames(night_file, [0])[0]
        assert sample["theta"]["metadata"].shape == (2,)
        default = _frames(
            night_file, [0], metadata_columns=("pressure", "temperature", "humidity")
        )[0]
        assert default["theta"]["metadata"].shape == (3,)
        with pytest.raises(KeyError, match="seeing"):
            _frames(night_file, [0], metadata_columns=("airmass", "seeing"))

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
        batch = next(iter(loader))
        assert batch["observed"].shape == (4, N_WAVE)                # (B, N)
        assert out.telluric.shape == (4, N_WAVE)                     # one spectrum each
        assert batch["wavelength"].shape == (4, N_WAVE)


def _tiny_for_n_wave(arch: str):
    """Configs that fit the 8-sample synthetic HDF5 of the scaling tests."""
    if arch == "cnn_autoencoder":
        return _cnn(N_WAVE, encoder_channels=[4, 4], kernel_size=3)
    if arch == "transformer_autoencoder":
        return _tae(N_WAVE, patch_size=2)
    return _specpt(N_WAVE, stem_channels=[4], stem_kernels=[3], token_stride=1)
