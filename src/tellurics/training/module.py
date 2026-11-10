"""PyTorch Lightning module for training the whole-night telluric predictor.

The model predicts telluric parameters for each exposure in a night.
Training uses MSE between the predicted and target parameters.

Inputs:
    observed: (B, T, N) observed spectra.
    stellar: (B, N) stellar spectrum used to create the observations.
    metadata: (B, T, 3) pressure, temperature, and humidity.
    time: (B, T) exposure times.

The model predicts:
    params: (B, T, P) telluric parameters.

When parameter scaling is enabled, targets and predictions are in [0, 1].
Use ``predict_physical()`` to convert predictions back to physical units.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from tellurics.config.model import ModelConfig
from tellurics.config.training import OptimizerType, SchedulerType, TrainingConfig
from tellurics.data.scaling import ParameterScaler
from tellurics.models.output import ModelOutput
from tellurics.utils.logging import get_logger
from tellurics.utils.registry import ModelRegistry


logger = get_logger(__name__)


class TelluricTrainingModule(pl.LightningModule):
    """Train the whole-night telluric parameter predictor.

    Uses MSE between predicted and target telluric parameters.
    Also configures the optimizer and learning-rate scheduler.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        scaler: ParameterScaler | None = None,
    ) -> None:
        """Initialize the training module.

        Args:
            model_config: Model architecture configuration.
            training_config: Optimizer, scheduler, and training settings.
            scaler: Optional parameter scaler used to convert predictions
                back to physical units.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])

        self.model_config = model_config
        self.training_config = training_config
        self.scaler = scaler

        model_cls = ModelRegistry.get(model_config.architecture.value)
        self.model = model_cls(model_config)
        logger.info(
            f"TelluricTrainingModule: {model_cls.__name__} with "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} "
            "trainable parameters"
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        observed: torch.Tensor,
        stellar: torch.Tensor,
        metadata: torch.Tensor,
        time: torch.Tensor,
    ) -> ModelOutput:
        """Run the telluric predictor.

        Args:
            observed: Observed spectra with shape (B, T, N).
            stellar: Stellar spectrum with shape (B, N) or (N,).
            metadata: Pressure, temperature, and humidity with shape (B, T, 3).
            time: Exposure times with shape (B, T).

        Returns:
            ModelOutput containing predicted parameters with shape (B, T, P).
        """
        return self.model(
            observed, stellar=stellar, metadata=metadata, time=time
        )

    # ------------------------------------------------------------------ #
    def _run_estimator(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Run the estimator using a DataModule batch."""
        return self(
            batch["observed"],
            stellar=batch["stellar"],
            metadata=batch["theta"]["metadata"],
            time=batch["theta"]["time"],
        )

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Run one training, validation, or test step.

        The loss is MSE between predicted and target parameters.
        """
        pred = self._run_estimator(batch).params                 # (B, T, P)
        target = batch["theta"]["output_params"]                 # (B, T, P)
        loss = F.mse_loss(pred, target)
        rmse = loss.sqrt()
        is_train = stage == "train"
        batch_size = batch["observed"].size(0)

        loss_name = {"train": "training_loss", "val": "validation_loss"}.get(
            stage, f"{stage}_loss"
        )
        nan_name = {"train": "nans", "val": "nans_val"}.get(
            stage, f"nans_{stage}"
        )

        self.log(
            loss_name, loss,
            on_step=False, on_epoch=True, prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_rmse", rmse,
            on_step=False, on_epoch=True,
            batch_size=batch_size,
        )
        # Percentage of the epoch's batches whose loss was NaN (0% = healthy).
        self.log(
            nan_name, 100.0 * torch.isnan(loss).float(),
            on_step=False, on_epoch=True,
        )
        if is_train:
            self.log(
                "learning_rate", self._current_lr(),
                on_step=False, on_epoch=True, prog_bar=True,
            )
        return loss

    def _current_lr(self) -> float:
        """Learning rate of the first optimizer's first parameter group."""
        optimizers = self.trainer.optimizers
        if not optimizers:
            return 0.0
        return float(optimizers[0].param_groups[0]["lr"])

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One Training step."""
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One Validation step."""
        return self._shared_step(batch, "val")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One Test step."""
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> ModelOutput:
        """Run the model on a batch without calculating loss.

        Returns predictions in the same normalized space used during training.
        Use ``predict_physical()`` to convert parameters to physical units.
        """
        return self._run_estimator(batch)

    # ------------------------------------------------------------------ #
    def predict_physical(
        self, batch: dict[str, torch.Tensor]
    ) -> ModelOutput:
        """Predict parameters and convert them to physical units.

        Returns:
            ModelOutput with parameters converted from normalized values
            to physical units.
        """
        return self.to_physical(self._run_estimator(batch))

    def to_physical(self, output: ModelOutput) -> ModelOutput:
        """Convert an existing ModelOutput from normalized to physical units.

        Returns the output unchanged when no parameter scaling is enabled.
        """
        if self.scaler is None or output.params is None:
            return output
        if not self.scaler.scale_params:
            return output
        return replace(
            output, params=self.scaler.inverse_params_torch(output.params)
        )

    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> dict:
        """Configure the optimizer and LR scheduler."""
        opt_config = self.training_config.optimizer
        sched_config = self.training_config.scheduler

        match opt_config.optimizer_type:
            case OptimizerType.ADAM:
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=opt_config.learning_rate,
                    betas=opt_config.betas,
                    weight_decay=opt_config.weight_decay,
                )
            case OptimizerType.SGD:
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=opt_config.learning_rate,
                    momentum=opt_config.momentum,
                    weight_decay=opt_config.weight_decay,
                )

        match sched_config.scheduler_type:
            case SchedulerType.COSINE:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.training_config.max_epochs,
                    eta_min=sched_config.min_lr,
                )
            case SchedulerType.REDUCE_ON_PLATEAU:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=sched_config.patience,
                    factor=sched_config.gamma,
                    min_lr=sched_config.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.training_config.monitor_metric,
                        "interval": "epoch",
                    },
                }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
