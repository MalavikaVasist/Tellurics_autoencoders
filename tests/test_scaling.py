"""Tests for the [0, 1] label scaling driven by the declared bounds.

Covers the three ``data.scale_*`` switches, the min-max and inverse transforms,
rejection of degenerate (``min == max``) bounds, rejection of out-of-bounds
values and the split-independence of the mapping. The tests use a tiny synthetic
night file, so they never touch the multi-GB production HDF5.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tellurics.config.parameters import (
    METADATA_COLUMNS,
    PHYSICAL_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
)
from tellurics.config.model import ModelConfig
from tellurics.config.training import TrainingConfig
from tellurics.data.datamodule import TelluricDataModule, build_scaler
from tellurics.data.datasets.timeseries import TelluricTimeseriesDataset
from tellurics.data.scaling import MinMax, ParameterScaler
from tellurics.training.module import TelluricTrainingModule

# ``DataConfig`` below is ``TestDataConfig``: the scaling tests exercise the
# transform math, not the (manifest-declared) bounds plumbing, so it pins the
# canonical table while keeping the ``DataConfig(...)`` call sites unchanged.
from tests._bounds import BOUNDS, TestDataConfig as DataConfig

N_WAVE = 8
N_FRAMES = 4

# Per-frame physical values of the synthetic nights. Frames 0..3 walk from the
# declared minimum towards the maximum, so the expected scaled values are exact.
TIME_HOURS = (0.0, 4.0, 8.0, 12.0)        # / 16          -> 0, .25, .50, .75
PRESSURE = (600.0, 700.0, 800.0, 850.0)   # [600, 850]    -> 0, .4, .8, 1
HUMIDITY = (0.0, 25.0, 50.0, 100.0)       # [0, 100]      -> 0, .25, .5, 1
AIRMASS = (1.0, 1.5, 1.75, 2.0)           # [1, 2]        -> 0, .5, .75, 1
CO2 = (420.0, 425.0, 430.0, 435.0)        # [420, 435]    -> 0, 1/3, 2/3, 1


class _FakeStellarPool:
    """Stand-in for ``StellarPool``: only ``pool[i] -> (spectrum, meta)`` is used."""

    def __init__(self, n_wavelength: int = N_WAVE) -> None:
        self.spectrum = np.linspace(1.0, 2.0, n_wavelength).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, None]:
        return self.spectrum, None


def _write_night_file(path, n_nights: int = 2) -> None:
    """Write a minimal night-major HDF5 with the real label schema."""
    columns = ["flat_index", "series_index", "frame_index", *PHYSICAL_COLUMNS]
    labels = np.zeros((n_nights, N_FRAMES, len(columns)), np.float32)
    for night in range(n_nights):
        # Night 1 is constant (a "different split" with a different spread).
        labels[night, :, columns.index(TIME_COLUMN)] = (
            TIME_HOURS if night == 0 else 6.0
        )
        labels[night, :, columns.index("pressure")] = (
            PRESSURE if night == 0 else 725.0
        )
        labels[night, :, columns.index("temperature")] = 280.0
        labels[night, :, columns.index("humidity")] = (
            HUMIDITY if night == 0 else 10.0
        )
        labels[night, :, columns.index("angle")] = 30.0
        labels[night, :, columns.index("airmass")] = (
            AIRMASS if night == 0 else 1.25
        )
        labels[night, :, columns.index("co2")] = CO2 if night == 0 else 427.0
        # Everything else sits on its declared minimum.
        for name in TARGET_COLUMNS:
            if name not in ("pressure", "temperature", "humidity", "angle",
                            "airmass", "co2"):
                labels[night, :, columns.index(name)] = getattr(
                    DataConfig().param_bounds, name
                )[0]

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "transmission",
            data=np.full((n_nights, N_FRAMES, N_WAVE), 0.5, np.float32),
        )
        f.create_dataset("wavelength", data=np.linspace(2000.0, 2100.0, N_WAVE))
        labels_ds = f.create_dataset("labels", data=labels)
        labels_ds.attrs["columns"] = np.array(columns, dtype="S")


def _dataset(
    path, night_ids, config: DataConfig, *, clip: bool = False
) -> TelluricTimeseriesDataset:
    """Dataset over ``night_ids`` with the scaler described by ``config``."""
    return TelluricTimeseriesDataset(
        night_ids=list(night_ids),
        h5_path=path,
        pool=_FakeStellarPool(),
        star_assignment=np.zeros(4, dtype=int),
        scaler=build_scaler(config, clip=clip),
    )


def _scaled(scaler: ParameterScaler, column: str, value: float) -> float:
    """Scaled value of ``value`` for one target column (all other columns 0)."""
    index = scaler.params.columns.index(column)
    row = np.zeros((1, len(scaler.params.columns)), dtype=np.float32)
    row[0, index] = value
    return float(scaler.params.transform(row)[0, index])


@pytest.fixture()
def night_file(tmp_path):
    path = tmp_path / "nights.h5"
    _write_night_file(path)
    return path


# --------------------------------------------------------------------------- #
# The three switches
# --------------------------------------------------------------------------- #
class TestScalingSwitches:
    def test_all_switches_on_by_default(self) -> None:
        scaler = build_scaler(DataConfig())
        assert scaler.scale_metadata and scaler.scale_params and scaler.scale_time

    def test_scale_params_false_leaves_targets_in_physical_units(
        self, night_file
    ) -> None:
        ds = _dataset(night_file, [0], DataConfig(scale_params=False))
        params = ds[0]["theta"]["output_params"].numpy()
        # Column 0 of TARGET_COLUMNS is pressure, returned raw.
        assert params[:, TARGET_COLUMNS.index("pressure")] == pytest.approx(PRESSURE)

    def test_scale_params_true_normalizes_targets(self, night_file) -> None:
        ds = _dataset(night_file, [0], DataConfig(scale_params=True))
        params = ds[0]["theta"]["output_params"].numpy()
        assert params[:, TARGET_COLUMNS.index("pressure")] == pytest.approx(
            [0.0, 0.4, 0.8, 1.0]
        )
        assert params[:, TARGET_COLUMNS.index("co2")] == pytest.approx(
            [0.0, 1 / 3, 2 / 3, 1.0]
        )

    def test_scale_time_false_keeps_hours(self, night_file) -> None:
        ds = _dataset(night_file, [0], DataConfig(scale_time=False))
        assert ds[0]["theta"]["time"].numpy() == pytest.approx(TIME_HOURS)

    def test_scale_time_true_divides_by_the_bound(self, night_file) -> None:
        """Default bound is [0, 16] h, i.e. exactly ``time_hours / 16``."""
        ds = _dataset(night_file, [0], DataConfig(scale_time=True))
        assert ds[0]["theta"]["time"].numpy() == pytest.approx(
            [h / 16.0 for h in TIME_HOURS]
        )

    def test_scale_metadata_is_independent_of_scale_params(self, night_file) -> None:
        config = DataConfig(scale_metadata=False, scale_params=True)
        scaler = build_scaler(config)
        assert scaler.metadata is None and scaler.params is not None

        meta = _dataset(night_file, [0], config)[0]["theta"]["metadata"].numpy()
        params = _dataset(night_file, [0], config)[0]["theta"]["output_params"].numpy()
        assert meta[:, METADATA_COLUMNS.index("pressure")] == pytest.approx(PRESSURE)
        assert params[:, TARGET_COLUMNS.index("pressure")] == pytest.approx(
            [0.0, 0.4, 0.8, 1.0]
        )

    def test_scale_metadata_true_normalizes_metadata(self, night_file) -> None:
        ds = _dataset(night_file, [0], DataConfig(scale_metadata=True))
        meta = ds[0]["theta"]["metadata"].numpy()
        assert meta[:, METADATA_COLUMNS.index("pressure")] == pytest.approx(
            [0.0, 0.4, 0.8, 1.0]
        )
        assert meta[:, METADATA_COLUMNS.index("humidity")] == pytest.approx(
            [0.0, 0.25, 0.5, 1.0]
        )

    def test_no_scaling_at_all_returns_identity(self, night_file) -> None:
        config = DataConfig(scale_metadata=False, scale_params=False, scale_time=False)
        scaler = build_scaler(config)
        assert not any(
            (scaler.scale_metadata, scaler.scale_params, scaler.scale_time)
        )

        sample = _dataset(night_file, [0], config)[0]
        assert sample["theta"]["time"].numpy() == pytest.approx(TIME_HOURS)
        assert sample["theta"]["output_params"].numpy()[
            :, TARGET_COLUMNS.index("pressure")
        ] == pytest.approx(PRESSURE)


# --------------------------------------------------------------------------- #
# The transforms themselves
# --------------------------------------------------------------------------- #
class TestMinMaxTransform:
    def test_maps_the_bounds_onto_the_unit_interval(self) -> None:
        group = MinMax.from_bounds(
            ("airmass", "co2"), {"airmass": (1.0, 2.0), "co2": (420.0, 435.0)}
        )
        scaled = group.transform(np.array([[1.0, 420.0], [2.0, 435.0]]))
        np.testing.assert_allclose(scaled, [[0.0, 0.0], [1.0, 1.0]])

    def test_formula_is_min_max(self) -> None:
        group = MinMax.from_bounds(("pressure",), {"pressure": (600.0, 850.0)})
        value = 725.0
        expected = (value - 600.0) / (850.0 - 600.0)
        assert group.transform(np.array([[value]])) == pytest.approx([[expected]])

    def test_inverse_is_the_exact_round_trip(self) -> None:
        group = MinMax.from_bounds(
            ("pressure", "airmass"),
            {"pressure": (600.0, 850.0), "airmass": (1.0, 2.0)},
        )
        physical = np.array(
            [[600.0, 1.0], [725.0, 1.5], [850.0, 2.0]], dtype=np.float32
        )
        assert group.inverse(group.transform(physical)) == pytest.approx(physical)

    def test_dtype_is_preserved(self) -> None:
        group = MinMax.from_bounds(("pressure",), {"pressure": (600.0, 850.0)})
        assert group.transform(np.array([[700.0]], np.float32)).dtype == np.float32

    def test_single_column_group_requires_a_two_dimensional_input(self) -> None:
        """The time group is (T, 1); a bare (T,) is refused."""
        group = MinMax.from_bounds((TIME_COLUMN,), {TIME_COLUMN: (0.0, 16.0)})
        assert group.transform(
            np.array([[0.0], [8.0], [16.0]])
        ) == pytest.approx([[0.0], [0.5], [1.0]])
        with pytest.raises(ValueError, match="expected a last axis of 1"):
            group.transform(np.array([0.0, 8.0, 16.0]))

    def test_values_outside_the_bounds_raise(self) -> None:
        group = MinMax.from_bounds(("o3",), {"o3": (0.018, 0.042)})
        # 0.0428 is the largest stored o3 value (the generator perturbs nightly
        # abundances by up to +/-2 %, so a few frames exceed the envelope).
        values = np.array([[0.017], [0.0428]])
        with pytest.raises(ValueError, match="outside the declared"):
            group.transform(values)
        assert group.count_out_of_bounds(values) == 2
        assert group.count_out_of_bounds(np.array([[0.02], [0.042]])) == 0

    def test_degenerate_bounds_are_rejected(self) -> None:
        """``min == max`` has no scale, so it must not build a group."""
        with pytest.raises(ValueError, match="min < max"):
            MinMax.from_bounds(("co",), {"co": (0.14, 0.14)})

    def test_missing_bounds_raise(self) -> None:
        with pytest.raises(KeyError):
            MinMax.from_bounds(("pressure",), {"airmass": (1.0, 2.0)})

    def test_misaligned_last_axis_raises(self) -> None:
        """A 1-D array must not be broadcast against a multi-column group."""
        group = MinMax.from_bounds(
            ("pressure", "airmass"), {"pressure": (600.0, 850.0), "airmass": (1.0, 2.0)}
        )
        with pytest.raises(ValueError, match="expected a last axis of 2"):
            group.transform(np.array([700.0, 1.5, 750.0]))
        with pytest.raises(ValueError, match="expected a last axis of 2"):
            group.count_out_of_bounds(np.array([700.0, 1.5]))

    def test_inverted_bounds_raise(self) -> None:
        with pytest.raises(ValueError):
            MinMax(("pressure",), np.array([900.0]), np.array([600.0]))


class TestMinMaxClipping:
    """``clip`` lets the forward transform absorb out-of-range values."""

    def test_clip_keeps_scaled_values_in_the_unit_interval(self) -> None:
        group = MinMax.from_bounds(
            ("o3",), {"o3": (0.018, 0.0429)}, clip=True
        )
        scaled = group.transform(np.array([[0.017], [0.045]]))
        np.testing.assert_allclose(scaled, [[0.0], [1.0]])

    def test_clip_is_off_by_default(self) -> None:
        group = MinMax.from_bounds(("o3",), {"o3": (0.018, 0.0429)})
        assert not group.clip
        with pytest.raises(ValueError, match="outside the declared"):
            group.transform(np.array([[0.045]]))

    def test_from_bounds_carries_the_clip_flag(self) -> None:
        group = MinMax.from_bounds(
            ("o3",), {"o3": (0.018, 0.0429)}, clip=True
        )
        assert group.clip
        np.testing.assert_array_equal(group.minimum, np.array([0.018]))

    def test_clip_tolerates_float32_boundary_rounding(self) -> None:
        """Bounds stored as float32 land just outside the float64 bounds."""
        group = MinMax.from_bounds(
            ("o3",), {"o3": (0.018, 0.0429)}, clip=True
        )
        scaled = group.transform(np.array([[0.018], [0.0429]], np.float32))
        np.testing.assert_allclose(scaled, [[0.0], [1.0]])

    def test_clipping_preserves_the_dtype(self) -> None:
        group = MinMax.from_bounds(
            ("o3",), {"o3": (0.018, 0.0429)}, clip=True
        )
        assert group.transform(np.array([[0.05]], np.float32)).dtype == np.float32

    def test_inverse_still_extrapolates(self) -> None:
        """Clipping is forward-only; inversion stays a plain linear map."""
        group = MinMax.from_bounds(
            ("o3",), {"o3": (0.018, 0.0429)}, clip=True
        )
        assert float(group.inverse(np.array([[1.5]]))[0, 0]) > 0.0429


class TestTestSplitClipping:
    """Only the test split absorbs values outside the declared bounds."""

    def test_boundary_rounding_is_rejected_by_the_strict_scaler(
        self, night_file
    ) -> None:
        ds = _dataset(night_file, [0], DataConfig())
        with pytest.raises(ValueError, match="outside the declared"):
            ds[0]

    def test_clipped_scaler_loads_the_same_night(self, night_file) -> None:
        ds = _dataset(night_file, [0], DataConfig(), clip=True)
        params = ds[0]["theta"]["output_params"].numpy()
        assert bool(((params >= 0.0) & (params <= 1.0)).all())

    def test_data_module_clips_only_the_test_split(
        self, night_file, tmp_path
    ) -> None:
        config = DataConfig(night_h5=night_file)
        module = TelluricDataModule(config, run_dir=tmp_path)
        module.scaler = build_scaler(config)
        module.stellar_pool = _FakeStellarPool()
        module.stellar_assignment = np.zeros(4, dtype=int)
        module.train_ids = [0]
        module.val_ids = [1]
        module.test_ids = [0]
        module._make_datasets()

        assert not module.train_ds.scaler.params.clip
        assert not module.val_ds.scaler.params.clip
        assert module.test_ds.scaler.params.clip
        assert module.test_ds[0]["theta"]["output_params"].shape == (
            N_FRAMES,
            len(TARGET_COLUMNS),
        )


# --------------------------------------------------------------------------- #
# Determinism / split independence / dataset wiring
# --------------------------------------------------------------------------- #
class TestSplitIndependence:
    def test_the_scaler_is_rebuildable_from_the_config_alone(self) -> None:
        first = build_scaler(DataConfig())
        second = build_scaler(DataConfig())          # e.g. at inference time
        assert np.array_equal(first.params.minimum, second.params.minimum)
        assert np.array_equal(first.params.maximum, second.params.maximum)
        assert np.array_equal(first.metadata.minimum, second.metadata.minimum)
        assert first.time.columns == second.time.columns

    def test_bounds_come_from_the_config_not_from_the_nights(self) -> None:
        """A split containing a constant night must not change the mapping."""
        scaler = build_scaler(DataConfig())
        # (725 - 600) / 250, whatever nights the file happens to contain.
        assert _scaled(scaler, "pressure", 725.0) == pytest.approx(0.5)

    def test_different_bounds_give_different_scaling(self) -> None:
        """700 hPa sits at 0.4 of [600, 850] but at 0.0 of [700, 750]."""
        wide = build_scaler(DataConfig())
        narrow = build_scaler(
            DataConfig(param_bounds={**BOUNDS, "pressure": (700.0, 750.0)})
        )
        assert _scaled(wide, "pressure", 700.0) == pytest.approx(0.4)
        assert _scaled(narrow, "pressure", 700.0) == pytest.approx(0.0)

    def test_same_physical_value_scales_the_same_in_any_split(
        self, night_file
    ) -> None:
        """A night present in "train" and alone in "val" scales identically."""
        config = DataConfig()
        train = _dataset(night_file, [0, 1], config)
        val = _dataset(night_file, [1], config)
        # Night 1 has constant pressure 725 hPa; (725-600)/250 = 0.5 either way.
        expected = 0.5
        assert train[1]["theta"]["output_params"].numpy()[
            :, TARGET_COLUMNS.index("pressure")
        ] == pytest.approx(expected)
        assert val[0]["theta"]["output_params"].numpy()[
            :, TARGET_COLUMNS.index("pressure")
        ] == pytest.approx(expected)

    def test_dataset_rejects_a_mismatched_scaler(self, night_file) -> None:
        """A scaler built for other columns would silently mis-scale."""
        scaler = ParameterScaler(
            params=MinMax.from_bounds(
                ("airmass",), {"airmass": (1.0, 2.0)}
            )
        )
        with pytest.raises(ValueError, match="scaler.params is built for columns"):
            TelluricTimeseriesDataset(
                night_ids=[0],
                h5_path=night_file,
                pool=_FakeStellarPool(),
                star_assignment=np.zeros(4, dtype=int),
                scaler=scaler,
            )

    def test_dataset_without_a_scaler_returns_physical_units(self, night_file) -> None:
        ds = TelluricTimeseriesDataset(
            night_ids=[0],
            h5_path=night_file,
            pool=_FakeStellarPool(),
            star_assignment=np.zeros(4, dtype=int),
        )
        assert ds.scaler is None
        assert ds[0]["theta"]["time"].numpy() == pytest.approx(TIME_HOURS)


# --------------------------------------------------------------------------- #
# The full scaler as assembled from a manifest
# --------------------------------------------------------------------------- #
class TestParameterScalerGroups:
    def test_groups_are_aligned_with_the_schema(self) -> None:
        scaler = build_scaler(DataConfig())
        assert scaler.metadata.columns == METADATA_COLUMNS
        assert scaler.params.columns == TARGET_COLUMNS
        assert scaler.time.columns == (TIME_COLUMN,)

    def test_time_bound_is_the_night_length(self) -> None:
        scaler = build_scaler(DataConfig())
        assert scaler.time.minimum[0] == 0.0 and scaler.time.maximum[0] == 16.0

    def test_dataset_tensors_are_float32_and_finite(self, night_file) -> None:
        ds = _dataset(night_file, [0], DataConfig())
        sample = ds[0]
        for key in ("time", "metadata", "output_params"):
            tensor = sample["theta"][key]
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()

    def test_scaled_targets_stay_within_the_unit_interval(self, night_file) -> None:
        """The sigmoid head can only reach such targets."""
        sample = _dataset(night_file, [0], DataConfig())[0]
        params = sample["theta"]["output_params"]
        assert bool(((params >= 0.0) & (params <= 1.0)).all())
        assert bool(
            ((sample["theta"]["metadata"] >= 0.0)
             & (sample["theta"]["metadata"] <= 1.0)).all()
        )

    def test_inverse_params_recovers_physical_units_torch(self, night_file) -> None:
        scaler = build_scaler(DataConfig())
        ds = _dataset(night_file, [0], DataConfig())
        params = ds[0]["theta"]["output_params"]
        physical = scaler.inverse_params_torch(params)
        assert physical.numpy()[
            :, TARGET_COLUMNS.index("pressure")
        ] == pytest.approx(PRESSURE)
        assert physical.dtype == params.dtype

    def test_inverse_params_is_identity_when_disabled(self) -> None:
        scaler = build_scaler(DataConfig(scale_params=False))
        values = np.array([[0.3, 0.7]], np.float32)
        assert scaler.inverse_params(values) is values
        tensor = torch.tensor([[0.3, 0.7]])
        assert scaler.inverse_params_torch(tensor) is tensor


# --------------------------------------------------------------------------- #
# The training / loss pipeline on scaled targets
# --------------------------------------------------------------------------- #
def _tiny_model_config() -> ModelConfig:
    """Small ``ModelConfig`` matching the synthetic nights of this module."""
    return ModelConfig(
        architecture="neural_telluric_predictor",
        num_wavelength_bins=N_WAVE,
        n_frames_per_series=N_FRAMES,
        metadata_dim=len(METADATA_COLUMNS),
        param_dim=len(TARGET_COLUMNS),
        num_queries=2,
        spectral_latent_dim=8,
        num_heads=2,
        dropout=0.0,
        x_encoder_channels=[4, 8],
        x_encoder_kernel=3,
        x_encoder_stride=2,
        s_encoder_channels=[4, 8],
        s_encoder_kernel=3,
        s_encoder_stride=2,
        encoder_pool_bins=2,
        fusion_hidden=0,
        fusion_layers=1,
        metadata_enc_dim=4,
        metadata_enc_hidden=None,
        time_enc_dim=4,
        param_decoder_hidden=None,
        param_activation="sigmoid",
    )


def _training_module(scaler: ParameterScaler | None = None) -> TelluricTrainingModule:
    """Tiny LightningModule, optionally wired to a scaler."""
    return TelluricTrainingModule(
        _tiny_model_config(),
        TrainingConfig(max_epochs=1, precision="32"),
        scaler=scaler,
    )


def _batch(night_file) -> dict:
    """One collated batch (2 nights) of the synthetic file."""
    loader = DataLoader(_dataset(night_file, [0, 1], DataConfig()), batch_size=2)
    return next(iter(loader))


class TestTrainingPipelineWithScaledTargets:
    """``MSE(param_pred, params)`` with both sides on ``[0, 1]`` still trains."""

    def test_loss_is_finite_and_both_sides_are_bounded(self, night_file) -> None:
        module = _training_module()
        batch = _batch(night_file)
        pred = module._run_estimator(batch).params
        target = batch["theta"]["output_params"]

        assert bool(((pred >= 0.0) & (pred <= 1.0)).all())
        assert bool(((target >= 0.0) & (target <= 1.0)).all())
        loss = F.mse_loss(pred, target)
        assert torch.isfinite(loss)

    def test_one_lightning_train_and_val_step_completes(self, night_file) -> None:
        loader = DataLoader(_dataset(night_file, [0, 1], DataConfig()), batch_size=2)
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(
            _training_module(),
            train_dataloaders=loader,
            val_dataloaders=loader,
        )
        assert trainer.state.status == "finished"


class TestPredictPhysical:
    """``predict_physical`` is the inverse of the dataset's forward scaling."""

    def test_predict_step_stays_normalized(self, night_file) -> None:
        module = _training_module(build_scaler(DataConfig())).eval()
        with torch.no_grad():
            params = module.predict_step(_batch(night_file), 0).params
        assert bool(((params >= 0.0) & (params <= 1.0)).all())

    def test_predict_physical_inverts_the_scaling(self, night_file) -> None:
        scaler = build_scaler(DataConfig())
        module = _training_module(scaler).eval()
        batch = _batch(night_file)

        with torch.no_grad():
            normalized = module.predict_step(batch, 0)
            physical = module.predict_physical(batch)

        assert torch.allclose(
            physical.params, scaler.inverse_params_torch(normalized.params)
        )

    def test_predictions_land_inside_the_declared_physical_range(
        self, night_file
    ) -> None:
        """A bounded head + the inverse bound every physical prediction."""
        scaler = build_scaler(DataConfig())
        with torch.no_grad():
            params = _training_module(scaler).eval().predict_physical(
                _batch(night_file)
            ).params

        low = torch.as_tensor(scaler.params.minimum, dtype=params.dtype)
        high = torch.as_tensor(scaler.params.maximum, dtype=params.dtype)
        assert bool((params >= low).all())
        assert bool((params <= high).all())
        # e.g. pressure comes back in hPa, not as a 0-1 fraction.
        pressure = TARGET_COLUMNS.index("pressure")
        assert float(params[..., pressure].min()) >= 600.0
        assert float(params[..., pressure].max()) <= 850.0

    def test_non_parameter_fields_pass_through(self, night_file) -> None:
        module = _training_module(build_scaler(DataConfig())).eval()
        batch = _batch(night_file)
        with torch.no_grad():
            normalized = module.predict_step(batch, 0)
            physical = module.to_physical(normalized)

        assert torch.equal(physical.latent, normalized.latent)
        assert torch.equal(physical.attention_weights, normalized.attention_weights)
        assert physical.intermediate_features is normalized.intermediate_features

    def test_identity_without_a_scaler(self, night_file) -> None:
        module = _training_module(None).eval()
        batch = _batch(night_file)
        with torch.no_grad():
            assert torch.equal(
                module.predict_physical(batch).params,
                module.predict_step(batch, 0).params,
            )

    def test_identity_when_scale_params_is_false(self, night_file) -> None:
        scaler = build_scaler(DataConfig(scale_params=False))
        module = _training_module(scaler).eval()
        batch = _batch(night_file)
        assert not scaler.scale_params
        with torch.no_grad():
            assert torch.equal(
                module.predict_physical(batch).params,
                module.predict_step(batch, 0).params,
            )

    def test_scaler_is_not_written_into_the_hparams(self) -> None:
        """Checkpoints stay scaler-free: it is rebuilt from config.yaml."""
        scaler = build_scaler(DataConfig())
        module = _training_module(scaler)
        assert module.scaler is scaler
        assert "scaler" not in module.hparams
