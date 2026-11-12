"""
Lightning DataModule for whole-night telluric training.

Combines the shared wavelength grid, PHOENIX stellar spectra, night-level
train/validation/test splits, and the dataset. A sample is either one
complete night ``(T, N)`` (whole-night models) or, with ``single_frame=True``,
one observation ``(N,)`` (per-observation models; the exposures of all
nights pooled). The split is night-level in
both cases.

Each night is paired with one fixed, seeded stellar spectrum:
    observed = flux * stellar_flux
(``flux``: the telluric spectra, HDF5 dataset ``data.target_key``.)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from tellurics.config.data import DataConfig
from tellurics.data.datasets.frames import TelluricFrameDataset
from tellurics.data.datasets.timeseries import (
    read_label_columns,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    TIME_COLUMN,
    TelluricTimeseriesDataset,
)
from tellurics.data.scaling import ParameterScaler
from tellurics.data.splits import split_night_indices, split_night_indices_order
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def build_scaler(config: DataConfig, *, clip: bool = False) -> ParameterScaler:
    """Build the ``[0, 1]`` label scaler described by a ``data:`` section.

    A pure function of the declared bounds (``data.param_bounds`` or
    ``data.metadata_bounds``) and the ``data.scale_*``
    flags: no statistic is read off the data and no split is involved, so the
    scaler is identical for train/val/test and can be rebuilt from a run's
    ``config.yaml`` alone, e.g. to map ``theta["output_params"]`` back to
    physical units::

        scaler = build_scaler(load_config("runs/<run>/config.yaml").data)
        physical = scaler.inverse_params(batch["theta"]["output_params"])

    With ``clip`` the forward transform clamps physical values to the declared
    bounds instead of raising. The data module enables it for the test split
    only, where ``float32`` storage can put a value marginally outside the
    declared envelope.
    """
    return ParameterScaler.from_bounds(
        config.bounds,
        metadata_columns=tuple(config.metadata_columns),
        target_columns=config.resolved_target_columns,
        time_column=config.resolved_time_column,
        scale_metadata=config.scale_metadata,
        scale_params=config.scale_params,
        scale_time=config.scale_time,
        clip=clip,
    )



def read_layout(path: Path, shape_key: str) -> dict:
    """Layout of a telluric HDF5, from the dataset ``shape_key``.

    Returns ``flat`` (one row per observation), ``n_nights``, and either
    ``rows`` / ``series`` (flat: rows per night, grouped by ``series_index``;
    nights may have different numbers of frames) or ``n_frames`` (night-major
    ``(n_nights, T, N)``), plus the wavelength grid ``wavelength`` / ``unit``.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if shape_key not in f:
            raise KeyError(f"{path} has no '{shape_key}' dataset; datasets: {list(f.keys())}")
        shape = f[shape_key].shape
        out: dict = {"flat": len(shape) == 2, "n_samples": int(shape[-1])}
        if out["flat"]:
            columns = read_label_columns(path)
            if "series_index" not in columns:
                raise KeyError(
                    f"flat HDF5 {path} needs a 'series_index' label column "
                    f"to group observations into nights; columns: {columns}"
                )
            series = np.asarray(f["labels"][:, columns.index("series_index")]).astype(np.int64)
            ids, inverse = np.unique(series, return_inverse=True)
            order = np.argsort(inverse, kind="stable")
            bounds = np.cumsum(np.bincount(inverse, minlength=len(ids)))[:-1]
            out.update(rows=np.split(order, bounds), series=ids, n_nights=len(ids))
        else:
            out.update(n_nights=int(shape[0]), n_frames=int(shape[1]))
        out["wavelength"] = np.asarray(f["wavelength"][:], np.float64)
        out["unit"] = str(f["wavelength"].attrs.get("unit", "nm"))
    return out


class TelluricDataModule(pl.LightningDataModule):
    """Whole-night DataModule for training the :class:`NeuralTelluricPredictor`.

    Args:
        config: Input paths and night-level split settings (the manifest's
            ``data:`` section).
        run_dir: This run's artefact directory; the night -> star audit CSV is
            written to ``<run_dir>/stellar_assignment.csv``.
        batch_size: Samples per batch -- whole nights, or single exposures
            with ``single_frame`` (the manifest's ``training:`` section).
        num_workers: DataLoader workers (the manifest's ``training:`` section).
        single_frame: Serve single observations ``(N,)`` instead of whole
            nights (set from the model config's ``single_frame``).
    """

    def __init__(
        self,
        config: DataConfig,
        run_dir: str | Path,
        batch_size: int = 4,
        num_workers: int = 2,
        single_frame: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.run_dir = Path(run_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.single_frame = single_frame

    # -- attributes set by setup() ------------------------------------------ #
    wavegrid: Wavegrid | None = None
    stellar_pool: StellarPool | None = None
    stellar_assignment: np.ndarray | None = None      # (n_nights,) star indices
    scaler: ParameterScaler | None = None             # [0,1] label scaling
    n_nights: int = 0
    n_frames: int = 0                                 # night-major layout only
    night_rows: dict[int, np.ndarray] | None = None   # flat layout: night -> rows
    night_series: np.ndarray | None = None            # flat layout: night -> series_index
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    train_ds: TelluricTimeseriesDataset | None = None
    val_ds: TelluricTimeseriesDataset | None = None
    test_ds: TelluricTimeseriesDataset | None = None
    real_ds: dict[str, TelluricTimeseriesDataset] = {}   # data.real, prediction only

    # -- setup -------------------------------------------------------------- #
    def setup(self, stage: str | None = None) -> None:
        """Prepare metadata, splits, star assignment and datasets (once)."""
        if self.train_ds is not None:
            return

        self._load_metadata()  # shapes + wavelength grid
        self._make_scaler()    # [0,1] scaling from the declared bounds
        self._make_splits()    # night-level train/val/test
        if self.observed_input:
            # The file already holds the observed flux: no stellar spectra.
            logger.info(
                f"data.observed_key = '{self.config.observed_key}': reading the "
                "observed flux from the file; no stellar spectra assigned"
            )
        else:
            self._assign_stars()   # fixed stellar spectrum per night + audit CSV
        self._make_datasets()  # one dataset per split
        self._make_real_datasets()  # data.real: prediction-only observations

    @property
    def observed_input(self) -> bool:
        """``data.observed_key`` set -- the HDF5 already holds the observed flux."""
        return self.config.observed_key is not None

    def _load_metadata(self) -> None:
        """Read the shapes and the wavelength grid."""
        if self.single_frame and self.config.param_bounds is not None:
            raise ValueError(
                "single-frame models read only metadata: give data.metadata_bounds "
                "instead of data.param_bounds"
            )
        night_h5 = Path(self.config.night_h5)
        if not night_h5.exists():
            raise FileNotFoundError(
                f"night-major HDF5 not found at {night_h5}. It is resolved "
                "relative to $TELLURICS_DATA -- check that the variable points "
                "at the data directory and that the file exists. Create it "
                "once with tests/testing_dataset_reshape.ipynb."
            )

        import h5py

        target = self.config.target_key
        with h5py.File(night_h5, "r") as f:
            if target not in f:
                raise KeyError(
                    f"data.target_key is '{target}' but {night_h5} has no "
                    f"'{target}' dataset; datasets: {list(f.keys())}"
                )
            if self.observed_input:
                key = self.config.observed_key
                if key not in f:
                    raise KeyError(
                        f"data.observed_key is '{key}' but {night_h5} has no "
                        f"'{key}' dataset; datasets: {list(f.keys())}"
                    )
                if f[key].shape != f[target].shape:
                    raise ValueError(
                        f"'{key}' {f[key].shape} must have the shape of "
                        f"'{target}' {f[target].shape} (the target)"
                    )
        layout = read_layout(night_h5, target)
        flat = layout["flat"]
        available_nights = layout["n_nights"]
        if flat:
            rows = layout["rows"]
            self.night_series = layout["series"]   # night position -> series_index
        else:
            self.n_frames = layout["n_frames"]
        wl, unit = layout["wavelength"], layout["unit"]

        # Debug/smoke runs may cap the nights used (the first N); the split and
        # the star assignment then only cover those nights.
        self.n_nights = available_nights
        cap = self.config.max_nights
        if cap is not None and cap < available_nights:
            self.n_nights = cap
            logger.warning(
                f"data.max_nights={cap}: using only the first {cap} of "
                f"{available_nights} nights (smoke/debug subset)."
            )

        self.wavegrid = Wavegrid(wavelength=wl, unit=unit)
        if flat:
            if not self.single_frame:
                raise ValueError(
                    f"{night_h5} is flat (one row per observation): only the "
                    "single-frame models can use it; the whole-night model "
                    "needs the night-major (n_nights, T, N) layout"
                )
            self.night_rows = {i: rows[i] for i in range(self.n_nights)}
            counts = np.array([len(rows[i]) for i in range(self.n_nights)])
            logger.info(
                f"{self.n_nights} nights, {counts.sum()} observations "
                f"({counts.min()}-{counts.max()} frames per night) from "
                f"flat {night_h5}"
            )
        else:
            logger.info(
                f"{self.n_nights} nights x {self.n_frames} frames from {night_h5}"
            )


    def _make_scaler(self) -> None:
        """Build the ``[0, 1]`` scaler from the manifest's declared bounds.
        """
        self.scaler = build_scaler(self.config)
        if not any(
            (self.scaler.scale_metadata, self.scaler.scale_params,
             self.scaler.scale_time)
        ):
            logger.info(
                "data.scale_metadata/scale_params/scale_time are all false: "
                "labels are fed to the model in physical units."
            )

    def _make_splits(self) -> None:
        """Night-level train/val/test split (no exposure crosses splits)."""
        self.train_ids, self.val_ids, self.test_ids = split_night_indices(
            self.n_nights,
            self.config.train_fraction,
            self.config.val_fraction,
            self.config.seed,
        )

    def _assign_stars(self) -> None:
        """Assign a fixed stellar spectrum per night and write the audit CSV."""
        self.stellar_pool = StellarPool(
            self.config.phoenix_dir, scale=self.config.stellar_scale
        )
        self.stellar_assignment = assign_stellar_per_night(
            self.n_nights, self.stellar_pool, seed=self.config.seed
        )
        audit_csv = self.run_dir / "stellar_assignment.csv"
        save_stellar_assignment(
            self.stellar_assignment,
            self.stellar_pool,
            {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids},
            audit_csv,
            series=(
                None if self.night_series is None
                else self.night_series[: self.n_nights]
            ),
        )
        logger.info(
            f"train {len(self.train_ids)} / val {len(self.val_ids)} / "
            f"test {len(self.test_ids)}; stellar audit -> {audit_csv}"
        )

  
    def _make_datasets(self) -> None:
        """Wrap each split in a :class:`TelluricTimeseriesDataset`.

        The test split gets a clipping scaler: labels are stored as
        ``float32``, so a frame on a declared bound can scale marginally
        outside ``[0, 1]`` and would otherwise raise. Train and validation
        stay strict so a mis-declared bound still fails loudly.
        """
        common = dict(
            h5_path=self.config.night_h5,
            pool=self.stellar_pool,                 # None for data.input: observed
            star_assignment=self.stellar_assignment,
            observed_key=self.config.observed_key if self.observed_input else None,
            metadata_columns=tuple(self.config.metadata_columns),
            target_key=self.config.target_key,
        )
        logger.info(f"metadata columns: {list(self.config.metadata_columns)}")
        dataset_cls = TelluricTimeseriesDataset
        if not self.single_frame:
            # whole night: also exposure times (and parameters, analysis only)
            common["target_columns"] = self.config.resolved_target_columns
            common["time_column"] = self.config.resolved_time_column
        else:
            # single frames read metadata only: no time, no parameters
            dataset_cls = TelluricFrameDataset
            if self.night_rows is not None:
                common["night_rows"] = self.night_rows      # flat: any frames per night
            else:
                common["n_frames"] = self.n_frames
                logger.info(
                    f"single-frame samples: {self.n_frames} exposures per night"
                )
        test_scaler = (
            None if self.scaler is None else build_scaler(self.config, clip=True)
        )
        # Gaussian noise on simulated inputs only: fresh every read for
        # training, fixed (seeded) for validation / test so their losses are
        # comparable from epoch to epoch. Each observation draws its own
        # relative level ~ N(noise_level_mean, noise_level_std^2).
        cfg = self.config
        noisy = cfg.simulated and cfg.noise_level_mean > 0
        noise = dict(
            noise_level_mean=cfg.noise_level_mean if noisy else 0.0,
            noise_level_std=cfg.noise_level_std if noisy else 0.0,
        )
        # Augmentation: train and validation are served noise_repeats times per
        # epoch, each pass with a different noise realization; test once.
        repeats = cfg.noise_repeats if noisy else 1
        if noisy:
            logger.info(
                f"Gaussian input noise (simulations): level ~ N({cfg.noise_level_mean:.3%}, "
                f"{cfg.noise_level_std:.3%}) of each pixel per observation; "
                f"train/val repeated x{repeats} per epoch with new noise"
            )
        seed = cfg.seed
        self.train_ds = dataset_cls(
            night_ids=self.train_ids, scaler=self.scaler,
            noise_seed=None, repeats=repeats, **noise, **common
        )
        self.val_ds = dataset_cls(
            night_ids=self.val_ids, scaler=self.scaler,
            noise_seed=seed + 1, repeats=repeats, **noise, **common
        )
        self.test_ds = dataset_cls(
            night_ids=self.test_ids, scaler=test_scaler,
            noise_seed=seed + 2, repeats=1, **noise, **common
        )
        logger.info(
            f"samples per epoch: train {len(self.train_ds)}, val {len(self.val_ds)}; "
            f"test {len(self.test_ds)}"
        )

    def _make_real_datasets(self) -> None:
        """One prediction-only dataset per ``data.real`` file.

        Real observations: observed flux as stored, no target, no noise, no
        stellar spectra. Every observation of the file is used. Labels are
        scaled with the training bounds, clipped (real values may sit outside).
        """
        self.real_ds = {}
        if not self.config.real:
            return
        scaler = None
        if self.scaler is not None:
            # real files carry no telluric parameters: drop that scaler group
            scaler = dataclasses.replace(build_scaler(self.config, clip=True), params=None)
        for source in self.config.real:
            layout = read_layout(source.night_h5, source.observed_key)
            if layout["n_samples"] != self.wavegrid.wavelength.size:
                raise ValueError(
                    f"data.real '{source.name}' has {layout['n_samples']} "
                    f"wavelength samples; the training data has "
                    f"{self.wavegrid.wavelength.size}"
                )
            common = dict(
                night_ids=list(range(layout["n_nights"])),
                h5_path=source.night_h5, pool=None, star_assignment=None,
                scaler=scaler, observed_key=source.observed_key, has_target=False,
                metadata_columns=tuple(self.config.metadata_columns),
            )
            if self.single_frame:
                if layout["flat"]:
                    common["night_rows"] = dict(enumerate(layout["rows"]))
                else:
                    common["n_frames"] = layout["n_frames"]
                ds = TelluricFrameDataset(**common)
            else:
                if layout["flat"]:
                    raise ValueError(
                        f"data.real '{source.name}' is flat: the whole-night "
                        "model needs the night-major (n_nights, T, N) layout"
                    )
                ds = TelluricTimeseriesDataset(
                    **common, target_columns=(),
                    time_column=self.config.resolved_time_column,
                )
            self.real_ds[source.name] = ds
            logger.info(f"real observations '{source.name}': {len(ds)} samples")

    # -- loaders ------------------------------------------------------------ #
    def _loader(self, ds: TelluricTimeseriesDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,  # drop partial training batch only
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def real_dataloaders(self) -> dict[str, DataLoader]:
        """``{name: loader}`` over the ``data.real`` observations (no shuffle)."""
        return {name: self._loader(ds, shuffle=False) for name, ds in self.real_ds.items()}
