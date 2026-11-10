from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from tellurics.config.parameters import Parameters
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

        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must be less than 1"
            )

        return self

    def _resolve(self, path: Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path