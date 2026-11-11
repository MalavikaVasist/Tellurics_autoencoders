from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from tellurics.config.parameters import (
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    Parameters,
)
from tellurics.config.paths import data_root


class DataConfig(BaseModel):
    root: Path = Field(default_factory=data_root)
    night_h5: Path = Path("telluric_timeseries/telluric_templates_nights.h5")
    phoenix_dir: Path = Path("phoenix/convolved")

    stellar_scale: bool = True
    scale_metadata: bool = True
    scale_params: bool = True
    scale_time: bool = True
    param_bounds: Parameters
    # Label columns fed to the model as metadata (order = metadata vector
    # order). Default: pressure, temperature, humidity; the single-frame
    # experiments use [airmass, pwv].
    metadata_columns: list[str] = Field(
        default_factory=lambda: list(METADATA_COLUMNS), min_length=1,
    )
    # Label columns returned as theta["output_params"] (analysis only, not a
    # training target). None -> the default TARGET_COLUMNS. Every listed
    # column must exist in the HDF5 file.
    target_columns: list[str] | None = None

    pin_memory: bool = True

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

        bounded = self.param_bounds.to_mapping()
        for group, columns in (
            ("metadata_columns", self.metadata_columns),
            ("target_columns", self.target_columns or []),
        ):
            if len(set(columns)) != len(columns):
                raise ValueError(f"data.{group} has duplicates: {columns}")
            missing = [c for c in columns if c not in bounded]
            if missing:
                raise ValueError(
                    f"data.{group} {missing} need bounds in data.param_bounds"
                )

        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must be less than 1"
            )

        return self

    @property
    def resolved_target_columns(self) -> tuple[str, ...]:
        """``target_columns`` or the default ``TARGET_COLUMNS``."""
        return tuple(self.target_columns) if self.target_columns else TARGET_COLUMNS

    def _resolve(self, path: Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path