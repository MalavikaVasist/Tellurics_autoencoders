"""Configuration models using Pydantic v2."""

from tellurics.config.data import METADATA_COLUMNS, DataConfig, RealSource
from tellurics.config.experiment import ExperimentConfig, load_config
from tellurics.config.model import (
    AnyModelConfig,
    CNNAutoencoderConfig,
    ModelArchitecture,
    SpecPTConfig,
    TelluricActivation,
    TransformerAutoencoderConfig,
)
from tellurics.config.training import OptimizerConfig, SchedulerConfig, TrainingConfig
from tellurics.config.wandb import WandbConfig

__all__ = [
    "AnyModelConfig",
    "CNNAutoencoderConfig",
    "TransformerAutoencoderConfig",
    "SpecPTConfig",
    "ModelArchitecture",
    "TelluricActivation",
    "DataConfig",
    "RealSource",
    "METADATA_COLUMNS",
    "ExperimentConfig",
    "WandbConfig",
    "load_config",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
]
