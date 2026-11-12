from pathlib import Path

import math

from pydantic import BaseModel, Field, field_validator, model_validator

from tellurics.config.paths import data_root

# Default metadata fed to the models (override with data.metadata_columns).
METADATA_COLUMNS = ("airmass", "PWV")


class RealSource(BaseModel):
    """A file of real observations: observed flux, no telluric target.

    Never trained on and never noised: after training, the best model
    predicts its tellurics into ``<run_dir>/predictions/<name>.h5``.
    """

    name: str = Field(min_length=1, description="Name of the output file.")
    night_h5: Path = Field(description="HDF5 (relative to data.root): the "
                                       "observed flux, wavelength, labels.")
    observed_key: str = "observed"           # HDF5 dataset of the observed flux


class DataConfig(BaseModel):
    """The ``data:`` section: one observation in, one telluric spectrum out."""

    root: Path = Field(default_factory=data_root)
    night_h5: Path = Path("telluric_timeseries/telluric_templates_nights.h5")
    phoenix_dir: Path = Path("phoenix/convolved")

    # HDF5 dataset holding the telluric spectra: the training target, and
    # (with observed_key: null) multiplied by the star to build the input.
    target_key: str = Field(default="flux", min_length=1)

    # Model input of night_h5 (target_key is always the target):
    #   null  -- night_h5 holds pure tellurics: every night gets a fixed
    #            PHOENIX star and observed = flux * star
    #   "<name>" -- night_h5 already holds the observed flux in that dataset:
    #            no stellar spectra are loaded, assigned or multiplied
    observed_key: str | None = None

    # Is night_h5 simulated? Only simulated inputs get noise.
    simulated: bool = True
    # Gaussian noise added to simulated inputs, relative per pixel:
    #   level ~ N(noise_level_mean, noise_level_std^2), one draw per
    #           observation (clipped at 0)
    #   observed + level * observed * N(0, 1)   (independent per pixel)
    # noise_level_mean 0 -> no noise. Training draws fresh noise on every
    # read; validation/test use fixed, seeded noise.
    noise_level_mean: float = Field(default=0.0, ge=0.0)
    noise_level_std: float = Field(default=0.0, ge=0.0)
    # Data augmentation: every epoch goes over the training and validation
    # observations this many times, each pass with a different noise
    # realization (test: once). Needs noise; 1 -> each observation once.
    noise_repeats: int = Field(default=1, ge=1)

    # Real observations (observed flux, no telluric target): not trained on,
    # not noised; predicted after training.
    real: list[RealSource] = Field(default_factory=list)

    # PHOENIX stars normalized to unit mean flux (observed_key: null only;
    # stored observed flux is never rescaled).
    stellar_scale: bool = True

    # Label columns fed to the model as metadata (order = metadata vector
    # order), and their [min, max] bounds used to scale them to [0, 1].
    metadata_columns: list[str] = Field(
        default_factory=lambda: list(METADATA_COLUMNS), min_length=1,
    )
    metadata_bounds: dict[str, tuple[float, float]]
    scale_metadata: bool = True

    pin_memory: bool = True

    # Night-level split (test = 1 - train - val).
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 42
    max_nights: int | None = Field(default=None, gt=0)

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        return root

    @model_validator(mode="after")
    def resolve_paths(self) -> "DataConfig":
        self.night_h5 = self._resolve(self.night_h5)
        self.phoenix_dir = self._resolve(self.phoenix_dir)
        for source in self.real:
            source.night_h5 = self._resolve(source.night_h5)
        names = [r.name for r in self.real]
        if len(set(names)) != len(names):
            raise ValueError(f"data.real names must be unique: {names}")
        if self.noise_level_mean > 0 and not self.simulated:
            raise ValueError(
                "data.noise_level_mean > 0 but data.simulated is false: noise "
                "is only added to simulations"
            )
        if self.noise_repeats > 1 and self.noise_level_mean == 0:
            raise ValueError(
                "data.noise_repeats > 1 needs noise (data.noise_level_mean > 0): "
                "without noise every repeat would be the same observation"
            )

        for name, (low, high) in self.metadata_bounds.items():
            if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
                raise ValueError(
                    f"data.metadata_bounds.{name} must be finite with min < max, "
                    f"got {[low, high]}"
                )
        columns = self.metadata_columns
        if len(set(columns)) != len(columns):
            raise ValueError(f"data.metadata_columns has duplicates: {columns}")
        missing = [c for c in columns if c not in self.metadata_bounds]
        if missing:
            raise ValueError(
                f"data.metadata_columns {missing} need bounds in data.metadata_bounds"
            )

        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must be less than 1"
            )

        return self

    def _resolve(self, path: Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path
