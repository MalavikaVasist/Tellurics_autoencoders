"""Training configuration."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class OptimizerType(str, Enum):
    """Available optimizers."""

    ADAM = "adam"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Available learning rate schedulers."""

    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.9, ge=0.0, description="For SGD only")


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = Field(default=5, ge=0)
    min_lr: float = Field(default=1e-7, ge=0.0)
    gamma: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="For ReduceOnPlateau (factor)"
    )
    patience: int = Field(default=10, gt=0, description="For ReduceOnPlateau")


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    max_epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=4, gt=0, description="Whole nights per batch.")
    num_workers: int = Field(default=2, ge=0, description="DataLoader workers.")
    gradient_clip_val: float | None = Field(default=1.0, ge=0.0)
    accumulate_grad_batches: int = Field(default=1, gt=0)
    precision: str = Field(default="16-mixed")
    early_stopping_patience: int = Field(default=15, gt=0)
    monitor_metric: str = Field(
        default="validation_loss",
        description="Metric monitored by ModelCheckpoint, EarlyStopping and "
                    "ReduceLROnPlateau. Must match a name the module logs.",
    )
    save_every_n_epochs: int = Field(
        default=1,
        gt=0,
        description="Save a checkpoint every N epochs (ModelCheckpoint). "
                    "Set to max_epochs to keep only the final epoch.",
    )
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Checkpoint folder, relative to the run directory.",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Logger folder, relative to the run directory.",
    )
    log_every_n_steps: int = Field(default=10, gt=0)
    val_check_interval: float | int = 1.0
