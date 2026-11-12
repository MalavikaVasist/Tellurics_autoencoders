"""Top-level experiment manifest: the ``model:`` / ``data:`` / ``training:`` sections."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from tellurics.config.data import DataConfig
from tellurics.config.model import AnyModelConfig
from tellurics.config.paths import expand_env_vars
from tellurics.config.training import TrainingConfig
from tellurics.config.wandb import WandbConfig
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentConfig(BaseModel):
    """A complete experiment: what to build, on what data, how to train it.

    Artefacts are written under ``run_dir``::

        runs/temporal_conv_001/
        |-- config.yaml               # resolved copy of this manifest
        |-- stellar_assignment.csv    # night -> star audit (from the DataModule)
        |-- logs/                     # logger output
        +-- checkpoints/              # ModelCheckpoint output
    """

    run_dir: Path = Field(
        default=Path("runs/default"),
        description="Where this run's artefacts are written: the resolved "
                    "config.yaml, the stellar-assignment audit, logs/ and "
                    "checkpoints/. Overridden by nothing; set it per manifest.",
    )
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    model: AnyModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


    @model_validator(mode="after")
    def _metadata_dim_matches_columns(self) -> "ExperimentConfig":
        """``model.metadata_dim`` must equal ``len(data.metadata_columns)``."""
        dim = getattr(self.model, "metadata_dim", None)
        if dim is not None and dim != len(self.data.metadata_columns):
            raise ValueError(
                f"model.metadata_dim={dim} but data.metadata_columns has "
                f"{len(self.data.metadata_columns)} entries "
                f"{self.data.metadata_columns}"
            )
        return self

    @model_validator(mode="after")
    def _whole_night_model_needs_time(self) -> "ExperimentConfig":
        """The whole-night model's TimeEncoder needs a time column."""
        if not self.model.single_frame and self.data.resolved_time_column is None:
            raise ValueError(
                f"model {self.model.architecture.value} uses exposure times: "
                "give data.param_bounds and data.time_column (e.g. time_hours)"
            )
        return self

    @model_validator(mode="after")
    def _whole_night_model_needs_stars(self) -> "ExperimentConfig":
        """The whole-night model takes the stellar spectrum as an input."""
        if not self.model.single_frame and self.data.input == "observed":
            raise ValueError(
                f"model {self.model.architecture.value} needs the stellar "
                "spectrum as an input: use data.input: tellurics"
            )
        return self


def load_config(path: str | Path | None) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from a YAML file.

        Environment variables (``${VAR}`` or ``$VAR``) are expanded before
        validation. Omitted sections use their default configuration values.

        Args:
            path: YAML manifest path, or ``None`` to use all defaults.

        Returns:
            A validated ``ExperimentConfig`` instance.

        Raises:
            FileNotFoundError: If the manifest does not exist.
            ValueError: If the YAML content is not a mapping.
            RuntimeError: If a referenced environment variable is unset.
    """
    if path is None:
        return ExperimentConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"experiment manifest not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping with 'model', 'data' and/or "
            "'training' sections"
        )
    # YAML has no variable expansion; do it here so manifests can use
    # ``data.root: "${TELLURICS_DATA}"`` and stay machine-independent.
    raw = expand_env_vars(raw)
    return ExperimentConfig(**raw)
