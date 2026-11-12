"""End-to-end data scenarios (single frames), every supported combination of

* simulations, tellurics x star (scaled PHOENIX star)   noisy | noiseless
* simulations, observed flux stored in the file         noisy | noiseless
* real observations (no target, no noise)  -> predictions after training

for the CNN and transformer autoencoders, flat and night-major files.

Each scenario builds a synthetic HDF5, runs the real ``TelluricDataModule``
setup, one Lightning train/val step, and the real-observation prediction.
The PHOENIX pool is replaced by a small fake one (no FITS files needed).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl
import torch

import tellurics.data.datamodule as dm_module
from tellurics.config import CNNAutoencoderConfig, TransformerAutoencoderConfig
from tellurics.config.data import DataConfig
from tellurics.config.training import TrainingConfig
from tellurics.data.datamodule import TelluricDataModule
from tellurics.training.module import TelluricTrainingModule
from tellurics.training.predict import predict_real_observations

N = 64           # wavelength samples
T = 4            # frames per night (night-major files)
NIGHTS = 10
FULL_BOUNDS = {"airmass": (1.0, 2.0), "pwv": (0.0, 25.0), "pressure": (600.0, 850.0)}


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
class _FakePool:
    """Stand-in for StellarPool: star k is a constant spectrum k + 2."""

    def __init__(self, phoenix_dir=None, scale=True) -> None:
        self.names = [f"star{k}" for k in range(3)]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        return np.full(N, index + 2.0, np.float32), None


@pytest.fixture(autouse=True)
def _fake_stellar_pool(monkeypatch):
    monkeypatch.setattr(dm_module, "StellarPool", _FakePool)


def _labels(n_rows: int, rng) -> tuple[list[str], np.ndarray]:
    """A few label columns, values inside FULL_BOUNDS."""
    cols = list(FULL_BOUNDS)
    lab = np.empty((n_rows, len(cols)), np.float32)
    for j, c in enumerate(cols):
        lo, hi = FULL_BOUNDS[c]
        lab[:, j] = lo + (hi - lo) * rng.uniform(0.2, 0.8, n_rows)
    return cols, lab


def _write(path, *, flat: bool, observed: bool, target: bool = True, seed: int = 0) -> None:
    """Flat (n_obs, N) with series_index, or night-major (NIGHTS, T, N)."""
    rng = np.random.default_rng(seed)
    if flat:
        frames = rng.integers(1, 6, NIGHTS)               # unequal frames per night
        series = np.repeat(np.arange(100, 100 + NIGHTS), frames)
        shape = (len(series), N)
        cols, lab = _labels(len(series), rng)
        cols, lab = ["series_index", *cols], np.column_stack([series, lab]).astype(np.float32)
    else:
        shape = (NIGHTS, T, N)
        cols, lab2 = _labels(NIGHTS * T, rng)
        lab = lab2.reshape(NIGHTS, T, -1)
    trans = (0.6 + 0.3 * rng.random(shape)).astype(np.float32)
    with h5py.File(path, "w") as f:
        if target:
            f.create_dataset("flux", data=trans)
        if observed:
            f.create_dataset("observed", data=(trans * 3.0).astype(np.float32))
        f.create_dataset("wavelength", data=np.linspace(2000.0, 2100.0, N))
        f.create_dataset("labels", data=lab).attrs["columns"] = np.array(cols, dtype="S")


def _single_frame_model(kind: str):
    common = dict(num_wavelength_bins=N, dropout=0.0, telluric_activation="optical_depth",
                  metadata_dim=2, metadata_enc_dim=4)
    if kind == "cnn":
        return CNNAutoencoderConfig(architecture="cnn_autoencoder", encoder_channels=[4, 4],
                                    kernel_size=3, stride=2, bottleneck_channels=2,
                                    latent_dim=8, **common)
    return TransformerAutoencoderConfig(architecture="transformer_autoencoder", patch_size=8,
                                        d_model=16, num_heads=2, num_encoder_layers=1,
                                        num_decoder_layers=1, dim_feedforward=32,
                                        num_latents=2, latent_dim=4,
                                        wavelength_frequencies=3, **common)


def _run(model_cfg, data_cfg, tmp_path):
    """setup -> one train/val step -> predict data.real."""
    dm = TelluricDataModule(data_cfg, run_dir=tmp_path, batch_size=2, num_workers=0)
    dm.setup()
    module = TelluricTrainingModule(model_cfg, TrainingConfig(max_epochs=1, precision="32"))
    trainer = pl.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         accelerator="cpu", devices=1)
    trainer.fit(module, datamodule=dm)
    assert trainer.state.status == "finished"
    return dm, module, predict_real_observations(module, dm, tmp_path)


# --------------------------------------------------------------------------- #
# simulations: tellurics x star | observed  x  noisy | noiseless;  + real
# --------------------------------------------------------------------------- #
META = dict(metadata_columns=["airmass", "pwv"],
            metadata_bounds={"airmass": FULL_BOUNDS["airmass"], "pwv": FULL_BOUNDS["pwv"]})
NOISY = dict(noise_level_mean=0.01, noise_level_std=0.001, noise_repeats=2)


@pytest.mark.parametrize("kind", ["cnn", "tae"])
@pytest.mark.parametrize("observed", [False, True], ids=["tellurics_x_star", "observed"])
@pytest.mark.parametrize("noisy", [True, False], ids=["noisy", "noiseless"])
@pytest.mark.parametrize("flat", [True, False], ids=["flat", "night_major"])
def test_frame_scenarios(tmp_path, kind, observed, noisy, flat) -> None:
    sim, real = tmp_path / "sim.h5", tmp_path / "real.h5"
    _write(sim, flat=flat, observed=observed)
    _write(real, flat=flat, observed=True, target=False, seed=1)
    data = DataConfig(
        night_h5=sim, observed_key="observed" if observed else None, simulated=True,
        real=[{"name": "obs1", "night_h5": real, "observed_key": "observed"}],
        **META, **(NOISY if noisy else {}),
    )
    dm, module, written = _run(_single_frame_model(kind), data, tmp_path)

    x = dm.train_ds[0]
    assert x["observed"].shape == (N,) and x["telluric"].shape == (N,)
    assert set(x["theta"]) == {"metadata"}
    assert ("stellar" in x) is (not observed)                 # stars only for tellurics
    assert (tmp_path / "stellar_assignment.csv").exists() is (not observed)
    assert ("noise_level" in x) is noisy
    assert len(dm.train_ds) == dm.train_ds._n_base() * (2 if noisy else 1)

    # noiseless: the input is exactly flux x star, or the stored observed flux
    if not noisy:
        expected = x["telluric"] * x["stellar"] if not observed else x["telluric"] * 3.0
        assert torch.allclose(x["observed"], expected)

    # real observations: every one predicted, no target, no noise, no star
    r = dm.real_ds["obs1"][0]
    assert "telluric" not in r and "stellar" not in r and "noise_level" not in r
    with h5py.File(written[0]) as f, h5py.File(real) as src:
        n_obs = int(np.prod(src["observed"].shape[:-1]))
        assert f["telluric_pred"].shape == (n_obs, N)
        assert f["frame_id"].shape == (n_obs,)


# --------------------------------------------------------------------------- #
# Noise: simulations only; random in training, fixed in validation / test
# --------------------------------------------------------------------------- #
def _noise_dm(tmp_path, simulated=True, mean=0.01, std=0.001, repeats=1, real=False):
    sim, rl = tmp_path / "sim.h5", tmp_path / "real.h5"
    _write(sim, flat=True, observed=True)
    _write(rl, flat=True, observed=True, target=False, seed=1)
    data = DataConfig(
        night_h5=sim, observed_key="observed", simulated=simulated,
        noise_level_mean=mean, noise_level_std=std, noise_repeats=repeats,
        metadata_columns=["airmass", "pwv"],
        metadata_bounds={"airmass": FULL_BOUNDS["airmass"], "pwv": FULL_BOUNDS["pwv"]},
        real=[{"name": "obs1", "night_h5": rl}] if real else [],
    )
    dm = TelluricDataModule(data, run_dir=tmp_path, num_workers=0)
    dm.setup()
    return dm, sim, rl


def _stored(path, ds, i):
    night, row, _ = ds._index[i]
    with h5py.File(path) as f:
        return f["observed"][row]


def test_noise_level_is_drawn_per_observation(tmp_path) -> None:
    """level ~ N(1%, 0.1%) per observation; noise = level * observed * N(0, 1)."""
    dm, sim, _ = _noise_dm(tmp_path, mean=0.01, std=0.001)
    ds = dm.train_ds
    levels, stds = [], []
    for _ in range(20):
        for i in range(len(ds)):
            x = ds[i]
            clean = _stored(sim, ds, i)
            rel = (x["observed"].numpy() - clean) / clean
            levels.append(float(x["noise_level"]))
            stds.append(rel.std() / float(x["noise_level"]))
    levels = np.array(levels)
    assert abs(levels.mean() - 0.01) < 0.0005 and abs(levels.std() - 0.001) < 0.0003
    assert abs(np.mean(stds) - 1.0) < 0.05             # per-pixel noise has std = level


def test_noise_repeats_multiply_train_and_val_not_test(tmp_path) -> None:
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    dm1, _, _ = _noise_dm(tmp_path / "a", repeats=1)
    dm3, _, _ = _noise_dm(tmp_path / "b", repeats=3)
    assert len(dm3.train_ds) == 3 * len(dm1.train_ds)
    assert len(dm3.val_ds) == 3 * len(dm1.val_ds)
    assert len(dm3.test_ds) == len(dm1.test_ds)
    n = dm3.val_ds._n_base()
    a, b = dm3.val_ds[0], dm3.val_ds[n]                # same observation, next repeat
    assert (a["night_id"], a["frame_id"]) == (b["night_id"], b["frame_id"])
    assert not torch.equal(a["observed"], b["observed"])   # different noise realization
    assert torch.equal(a["observed"], dm3.val_ds[0]["observed"])   # but fixed per epoch


def test_repeats_without_noise_are_rejected(tmp_path) -> None:
    sim = tmp_path / "sim.h5"
    _write(sim, flat=True, observed=True)
    with pytest.raises(ValueError, match="noise_repeats"):
        DataConfig(night_h5=sim, observed_key="observed", noise_repeats=2,
                   metadata_columns=["airmass", "pwv"],
                   metadata_bounds={"airmass": (1, 2), "pwv": (0, 25)})


def test_target_key_selects_the_hdf5_dataset(tmp_path) -> None:
    """Default 'flux'; an older file with 'transmission' works via target_key."""
    sim = tmp_path / "sim.h5"
    _write(sim, flat=True, observed=True)
    with h5py.File(sim, "a") as f:
        f["transmission"] = f["flux"][:]
        del f["flux"]
    kw = dict(night_h5=sim, observed_key="observed", metadata_columns=["airmass", "pwv"],
              metadata_bounds={"airmass": (1, 2), "pwv": (0, 25)})
    dm = TelluricDataModule(DataConfig(**kw), run_dir=tmp_path, num_workers=0)
    with pytest.raises(KeyError, match="target_key"):
        dm.setup()
    dm = TelluricDataModule(DataConfig(target_key="transmission", **kw), run_dir=tmp_path,
                            num_workers=0)
    dm.setup()
    assert dm.train_ds[0]["telluric"].shape == (N,)


def test_training_noise_is_fresh_validation_noise_is_fixed(tmp_path) -> None:
    dm, _, _ = _noise_dm(tmp_path)
    a, b = dm.train_ds[0]["observed"], dm.train_ds[0]["observed"]
    assert not torch.equal(a, b)                                   # new draw every read
    v1, v2 = dm.val_ds[0]["observed"], dm.val_ds[0]["observed"]
    assert torch.equal(v1, v2)                                     # same every epoch
    t1, t2 = dm.test_ds[0]["observed"], dm.test_ds[0]["observed"]
    assert torch.equal(t1, t2)


def test_real_observations_never_get_noise(tmp_path) -> None:
    dm, _, rl = _noise_dm(tmp_path, real=True)
    ds = dm.real_ds["obs1"]
    for i in range(len(ds)):
        assert np.array_equal(ds[i]["observed"].numpy(), _stored(rl, ds, i))
        assert "telluric" not in ds[i]                             # no target


def test_no_noise_when_not_simulated(tmp_path) -> None:
    dm, sim, _ = _noise_dm(tmp_path, simulated=False, mean=0.0, std=0.0)
    ds = dm.train_ds
    assert np.array_equal(ds[0]["observed"].numpy(), _stored(sim, ds, 0))


def test_noise_on_non_simulated_data_is_rejected(tmp_path) -> None:
    sim = tmp_path / "sim.h5"
    _write(sim, flat=True, observed=True)
    with pytest.raises(ValueError, match="simulated"):
        DataConfig(night_h5=sim, observed_key="observed", simulated=False, noise_level_mean=0.01,
                   metadata_columns=["airmass", "pwv"],
                   metadata_bounds={"airmass": (1, 2), "pwv": (0, 25)})


def test_real_file_must_match_the_wavelength_grid(tmp_path) -> None:
    sim, rl = tmp_path / "sim.h5", tmp_path / "real.h5"
    _write(sim, flat=True, observed=True)
    with h5py.File(rl, "w") as f:
        f.create_dataset("observed", data=np.ones((3, N + 1), np.float32))
        f.create_dataset("wavelength", data=np.arange(N + 1.0))
        cols = ["series_index", "airmass", "pwv"]
        f.create_dataset("labels", data=np.array([[1, 1.5, 5]] * 3, np.float32)).attrs["columns"] = \
            np.array(cols, dtype="S")
    data = DataConfig(night_h5=sim, observed_key="observed", metadata_columns=["airmass", "pwv"],
                      metadata_bounds={"airmass": (1, 2), "pwv": (0, 25)},
                      real=[{"name": "bad", "night_h5": rl}])
    dm = TelluricDataModule(data, run_dir=tmp_path, num_workers=0)
    with pytest.raises(ValueError, match="wavelength samples"):
        dm.setup()
