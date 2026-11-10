"""Weights & Biases configuration: the ``wandb:`` section of a manifest."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WandbConfig(BaseModel):
    """Where a run's metrics are logged.

    Maps onto :class:`pytorch_lightning.loggers.WandbLogger`.
    """

    project: str = Field(
        default="neural_telluric_predictor",
        description="W&B project the run is logged to.",
    )
    title: str | None = Field(
        default=None,
        description="Human-readable run name/title (WandbLogger's ``name``). "
                    "None -> W&B generates one.",
    )
    mode: str = Field(
        default="online",
        description="W&B mode: 'online', 'offline', or 'disabled'.",
    )