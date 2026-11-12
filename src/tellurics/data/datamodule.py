"""
Lightning DataModule for single-observation telluric training.

Combines the shared wavelength grid, the night-level train/validation/test
splits, the optional PHOENIX stellar spectra, and the dataset. A sample is one
observation ``(N,)``; the exposures of all nights of a split are pooled, but
no exposure of a validation or test night is ever seen in training.

Input: with ``data.observed_key: null`` each night is paired with one fixed,
seeded stellar spectrum, ``observed = flux * star``; with ``observed_key``
set the observed flux is read from the file as stored (no stars).
(``flux``: the telluric spectra, HDF5 dataset ``data.target_key``.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tellurics.config.data import DataConfig
from tellurics.data.datasets.frames import TelluricFrameDataset, read_label_columns
from tellurics.data.scaling import MinMax
from tellurics.data.splits import split_night_indices
from tellurics.data.stellar import (
    StellarPool,
    assign_stellar_per_night,
    save_stellar_assignment,
)
from tellurics.data.wavegrid import Wavegrid
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def build_scaler(config: DataConfig, *, clip: bool = False) -> MinMax | None:
    """The ``[0, 1]`` metadata scaler described by a ``data:`` section.

    A pure function of ``data.metadata_bounds`` (no statistic is read off the
    data), so it is identical for every split and can be rebuilt from a run's
    ``config.yaml`` alone. ``None`` with ``data.scale_metadata: false``.

    With ``clip`` the transform clamps values to the bounds instead of
    raising: used for the test split (``float32`` storage can put a value
    marginally outside) and for real observations.
    """
    if not config.scale_metadata:
        return None
    return MinMax.from_bounds(config.metadata_columns, config.metadata_bounds, clip=clip)


def read_layout(path: Path, shape_key: str) -> dict:
    """Layout of a telluric HDF5, from the dataset ``shape_key``.

    Returns ``flat`` (one row per observation), ``n_nights``, and either
    ``rows`` / ``series`` (flat: rows per night, grouped by ``series_index``;
    nights may have different numbers of frames) or ``n_frames`` (night-major
    ``(n_nights, T, N)``), plus the wavelength grid ``wavelength`` / ``unit``
    and the number of spectral samples ``n_samples``.
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
    """Serves single observations ``(N,)`` with a night-level split.

    Args:
        config: The manifest's ``data:`` section.
        run_dir: This run's artefact directory; the night -> star audit CSV is
            written to ``<run_dir>/stellar_assignment.csv`` (stars only).
        batch_size: Observations per batch.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        config: DataConfig,
        run_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.run_dir = Path(run_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    # -- attributes set by setup() ------------------------------------------ #
    wavegrid: Wavegrid | None = None
    stellar_pool: StellarPool | None = None
    stellar_assignment: np.ndarray | None = None      # (n_nights,) star indices
    scaler: MinMax | None = None                      # [0,1] metadata scaling
    n_nights: int = 0
    n_frames: int = 0                                 # night-major layout only
    night_rows: dict[int, np.ndarray] | None = None   # flat layout: night -> rows
    night_series: np.ndarray | None = None            # flat layout: night -> series_index
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    train_ds: TelluricFrameDataset | None = None
    val_ds: TelluricFrameDataset | None = None
    test_ds: TelluricFrameDataset | None = None
    real_ds: dict[str, TelluricFrameDataset] = {}     # data.real, prediction only

    # -- setup -------------------------------------------------------------- #
    def setup(self, stage: str | None = None) -> None:
        """Prepare layout, splits, (stars) and datasets (once)."""
        if self.train_ds is not None:
            return

        self._load_metadata()  # layout + wavelength grid
        self.scaler = build_scaler(self.config)
        self._make_splits()    # night-level train/val/test
        if self.observed_input:
            logger.info(
                f"data.observed_key = '{self.config.observed_key}': reading the "
                "observed flux from the file; no stellar spectra"
            )
        else:
            self._assign_stars()   # fixed stellar spectrum per night + audit CSV
        self._make_datasets()      # one dataset per split
        self._make_real_datasets() # data.real: prediction-only observations

    @property
    def observed_input(self) -> bool:
        """``data.observed_key`` set -- the HDF5 already holds the observed flux."""
        return self.config.observed_key is not None

    def _load_metadata(self) -> None:
        """Read the layout and the wavelength grid."""
        night_h5 = Path(self.config.night_h5)
        if not night_h5.exists():
            raise FileNotFoundError(
                f"HDF5 not found at {night_h5}. It is resolved relative to "
                "$TELLURICS_DATA -- check that the variable points at the data "
                "directory and that the file exists."
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

        # Debug/smoke runs may cap the nights used (the first N); the split and
        # the star assignment then only cover those nights.
        available = layout["n_nights"]
        self.n_nights = available
        cap = self.config.max_nights
        if cap is not None and cap < available:
            self.n_nights = cap
            logger.warning(
                f"data.max_nights={cap}: using only the first {cap} of "
                f"{available} nights (smoke/debug subset)."
            )

        self.wavegrid = Wavegrid(wavelength=layout["wavelength"], unit=layout["unit"])
        if layout["flat"]:
            rows = layout["rows"]
            self.night_series = layout["series"]   # night position -> series_index
            self.night_rows = {i: rows[i] for i in range(self.n_nights)}
            counts = np.array([len(rows[i]) for i in range(self.n_nights)])
            logger.info(
                f"{self.n_nights} nights, {counts.sum()} observations "
                f"({counts.min()}-{counts.max()} frames per night) from "
                f"flat {night_h5}"
            )
        else:
            self.n_frames = layout["n_frames"]
            logger.info(
                f"{self.n_nights} nights x {self.n_frames} frames from {night_h5}"
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
            f"test {len(self.test_ids)} nights; stellar audit -> {audit_csv}"
        )

    def _layout_kwargs(self) -> dict:
        """``night_rows`` (flat) or ``n_frames`` (night-major) for the datasets."""
        if self.night_rows is not None:
            return {"night_rows": self.night_rows}
        return {"n_frames": self.n_frames}

    def _make_datasets(self) -> None:
        """One :class:`TelluricFrameDataset` per split.

        The test split gets a clipping scaler: labels are stored as
        ``float32``, so a frame on a declared bound can scale marginally
        outside ``[0, 1]`` and would otherwise raise. Train and validation
        stay strict so a mis-declared bound still fails loudly.
        """
        cfg = self.config
        common = dict(
            h5_path=cfg.night_h5,
            pool=self.stellar_pool,                     # None with observed_key
            star_assignment=self.stellar_assignment,
            observed_key=cfg.observed_key,
            target_key=cfg.target_key,
            metadata_columns=tuple(cfg.metadata_columns),
            **self._layout_kwargs(),
        )
        logger.info(f"metadata columns: {list(cfg.metadata_columns)}")

        # Gaussian noise on simulated inputs only: fresh every read for
        # training, fixed (seeded) for validation / test so their losses are
        # comparable from epoch to epoch. Each observation draws its own
        # relative level ~ N(noise_level_mean, noise_level_std^2).
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
        self.train_ds = TelluricFrameDataset(
            night_ids=self.train_ids, scaler=self.scaler,
            noise_seed=None, repeats=repeats, **noise, **common
        )
        self.val_ds = TelluricFrameDataset(
            night_ids=self.val_ids, scaler=self.scaler,
            noise_seed=seed + 1, repeats=repeats, **noise, **common
        )
        self.test_ds = TelluricFrameDataset(
            night_ids=self.test_ids, scaler=build_scaler(cfg, clip=True),
            noise_seed=seed + 2, repeats=1, **noise, **common
        )
        logger.info(
            f"samples per epoch: train {len(self.train_ds)}, val {len(self.val_ds)}; "
            f"test {len(self.test_ds)}"
        )

    def _make_real_datasets(self) -> None:
        """One prediction-only dataset per ``data.real`` file.

        Real observations: observed flux as stored, no target, no noise, no
        stellar spectra. Every observation of the file is used. The metadata
        is scaled with the training bounds (as the model was trained), and
        clipped since real values may sit outside them.
        """
        self.real_ds = {}
        for source in self.config.real:
            layout = read_layout(source.night_h5, source.observed_key)
            if layout["n_samples"] != self.wavegrid.wavelength.size:
                raise ValueError(
                    f"data.real '{source.name}' has {layout['n_samples']} "
                    f"wavelength samples; the training data has "
                    f"{self.wavegrid.wavelength.size}"
                )
            layout_kw = (
                {"night_rows": dict(enumerate(layout["rows"]))} if layout["flat"]
                else {"n_frames": layout["n_frames"]}
            )
            ds = TelluricFrameDataset(
                night_ids=list(range(layout["n_nights"])),
                h5_path=source.night_h5,
                scaler=build_scaler(self.config, clip=True),
                metadata_columns=tuple(self.config.metadata_columns),
                observed_key=source.observed_key,
                has_target=False,
                **layout_kw,
            )
            self.real_ds[source.name] = ds
            logger.info(f"real observations '{source.name}': {len(ds)} samples")

    # -- loaders ------------------------------------------------------------ #
    def _loader(self, ds: TelluricFrameDataset, shuffle: bool) -> DataLoader:
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
