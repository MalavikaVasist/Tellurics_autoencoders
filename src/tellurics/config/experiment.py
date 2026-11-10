"""Top-level experiment manifest: the ``model:`` / ``data:`` / ``training:`` sections."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from tellurics.config.data import DataConfig
from tellurics.config.model import ModelConfig, ParamActivation
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
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def _warn_on_scaling_mismatch(self) -> ExperimentConfig:
        """Flag a mismatch between the target scaling and the head activation.

        ``MSE(param_pred, params)`` is only well-posed when both sides live on
        the same scale, so ``data.scale_params`` and
        ``model.param_activation`` have to agree. This is a warning, not an
        error, because the run is still trainable -- just not sensibly.
        """
        sigmoid = self.model.param_activation is ParamActivation.SIGMOID
        if self.data.scale_params and not sigmoid:
            logger.warning(
                "data.scale_params=true puts the targets in [0, 1] but "
                "model.param_activation='none' lets the head output any real "
                "value; the MSE floor is then unreachable. Set "
                "model.param_activation: sigmoid."
            )
        elif not self.data.scale_params and sigmoid:
            logger.warning(
                "model.param_activation='sigmoid' bounds the head to (0, 1) "
                "but data.scale_params=false keeps the targets in physical "
                "units; the targets are unreachable. Set "
                "data.scale_params: true or model.param_activation: none."
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
