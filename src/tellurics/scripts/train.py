
"""
Train one telluric estimator: the whole-night model or a single-frame model.

Four experiments share this script; each has its own manifest and the
``model.architecture`` value selects both the model and the sample type:

    experiments/multiple_frames/experiment.yaml               neural_telluric_predictor  (B, T, N)
    experiments/single_frame/experiment_cnn_ae.yaml           cnn_autoencoder            (B, N) -> (B, N)
    experiments/single_frame/experiment_transformer_ae.yaml   transformer_autoencoder    (B, N) -> (B, N)
    experiments/single_frame/experiment_specpt.yaml           specpt                     (B, N) -> (B, N)

The manifest folders mirror ``tellurics.models``: ``multiple_frames/`` for the
whole-night model, ``single_frame/`` for the single-exposure autoencoders.

Pipeline
--------
1. Load the night-major telluric HDF5 dataset (nights x 73 frames x 51,557
   wavelength samples), prepared by tests/testing_dataset_reshape.ipynb.

2. TelluricDataModule pairs each night with a fixed, seeded PHOENIX stellar
   spectrum and serves either whole nights or, for single-frame models,
   single exposures (night-level split either way):
   - observed spectra: (B, T, N)  or single observations (B, N)
   - stellar spectra: (B, N)
   - telluric transmission (target): (B, T, N)  or (B, N)
   - theta: time, atmospheric metadata (pressure, temperature, humidity)
     as inputs, plus the 16 telluric parameters (not used in the loss).

3. TelluricTrainingModule trains and validates the estimator by minimizing
   mean squared error (MSE) between predicted and true telluric spectra:

       loss = MSE(estimator(...).telluric, batch["telluric"])

Configuration
-------------
A YAML manifest defines the run directory, WandB settings, model
architecture, data paths and splits, and training settings such as
the optimizer, number of epochs, batch size, and logging.

Each run saves its configuration, stellar assignments, logs, and
checkpoints in its output directory:

    runs/<experiment>/
    ├── config.yaml
    ├── stellar_assignment.csv
    ├── logs/
    └── checkpoints/

Examples
--------
Train using an experiment configuration:

    python -m tellurics.scripts.train --config experiments/multiple_frames/experiment.yaml
    python -m tellurics.scripts.train --config experiments/single_frame/experiment_cnn_ae.yaml
    python -m tellurics.scripts.train --config experiments/single_frame/experiment_transformer_ae.yaml
    python -m tellurics.scripts.train --config experiments/single_frame/experiment_specpt.yaml

Train on CPU or resume from a checkpoint (checkpoints/<architecture>/):

    python -m tellurics.scripts.train --config experiments/single_frame/experiment_specpt.yaml \\
        --device cpu --resume runs/specpt/checkpoints/specpt/x.ckpt
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from tellurics.config import (
    DataConfig,
    AnyModelConfig,
    ExperimentConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
)
from tellurics.data.datamodule import TelluricDataModule
from tellurics.training.module import TelluricTrainingModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the deliberately tiny CLI: the manifest plus two run overrides."""
    parser = argparse.ArgumentParser(
        description="Train the whole-night telluric time-series estimator."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML experiment manifest with 'model', 'data' and 'training' "
             "sections (see experiments/).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint (.ckpt) to resume training from.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="'auto' (default), 'cpu', 'gpu'/'cuda', 'mps', or a GPU index "
             "such as '0'.",
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value using a dotted path. "
            "Repeat for multiple overrides."
        ),
    )

    return parser.parse_args()


def device_kwargs(device: str) -> dict:
    """Translate ``--device`` into ``pl.Trainer`` accelerator/devices kwargs."""
    value = device.strip().lower()
    if value == "auto":
        return {"accelerator": "auto", "devices": "auto"}
    if value == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    if value in {"gpu", "cuda"}:
        return {"accelerator": "gpu", "devices": "auto"}
    if value == "mps":
        return {"accelerator": "mps", "devices": 1}
    if value.isdigit():
        return {"accelerator": "gpu", "devices": int(value)}
    raise ValueError(
        "--device must be 'auto', 'cpu', 'gpu'/'cuda', 'mps' or an integer "
        f"GPU index, got {device!r}"
    )


def prepare_run_dir(config: ExperimentConfig) -> Path:
    """Create ``run_dir`` and drop a resolved manifest copy inside."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    logger.info(f"run directory: {run_dir}")
    return run_dir


def build_datamodule(
    data: DataConfig,
    training: TrainingConfig,
    run_dir: Path,
    single_frame: bool = False,
) -> TelluricDataModule:
    """Build the DataModule, failing early on missing inputs."""
    if not data.night_h5.exists():
        raise FileNotFoundError(
            f"night-major HDF5 not found at {data.night_h5}. It is resolved "
            "relative to $TELLURICS_DATA -- check that the variable points at "
            "the data directory and that the file exists. Create it once with "
            "tests/testing_dataset_reshape.ipynb."
        )
    if not data.phoenix_dir.is_dir():
        raise FileNotFoundError(
            f"Phoenix stellar pool directory not found at {data.phoenix_dir}. "
            "It is resolved relative to $TELLURICS_DATA."
        )
    return TelluricDataModule(
        data,
        run_dir=run_dir,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        single_frame=single_frame,
    )


def build_model(
    model_config: AnyModelConfig,
    training_config: TrainingConfig,
) -> TelluricTrainingModule:
    """Build the LightningModule (registry-resolved estimator) to train.
    """
    frames = (
        "single observations (B, N) -> (B, N)" if model_config.single_frame
        else f"whole nights, T={model_config.n_frames_per_series}"
    )
    logger.info(
        f"Training {model_config.architecture.value} | "
        f"N={model_config.num_wavelength_bins}, {frames}, "
        f"telluric_activation={model_config.telluric_activation.value}"
    )
    return TelluricTrainingModule(model_config, training_config)


def build_trainer(
    config: TrainingConfig,
    wandb_config: WandbConfig,
    run_dir: Path,
    device: str,
    architecture: str = "neural_telluric_predictor",
) -> pl.Trainer:
    """Build the Trainer: callbacks, W&B logger and device placement."""
    checkpoint_dir = run_dir / config.checkpoint_dir
    log_dir = run_dir / config.log_dir

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir / architecture,
            monitor=config.monitor_metric,
            mode="min",
            every_n_epochs=config.save_every_n_epochs,
            save_top_k=-1,
            filename="estimator-{epoch:03d}-{validation_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=config.monitor_metric,
                patience=config.early_stopping_patience,
                mode="min",
            )
        )

    if wandb_config.mode == "disabled":
        logger = False
    else:
        logger = WandbLogger(
            save_dir=str(log_dir),
            project=wandb_config.project,
            name=wandb_config.title,
            offline=(wandb_config.mode == "offline"),
        ),

    return pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        # W&B auth: run `wandb login` once, or set WANDB_API_KEY; use
        # WANDB_MODE=offline for a purely local run.
        logger=logger, 
        callbacks=callbacks,
        enable_progress_bar=True,
        **device_kwargs(device),
    )


def train(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    config: TrainingConfig,
    wandb_config: WandbConfig,
    run_dir: Path,
    device: str,
    resume: Path | None,
    architecture: str = "neural_telluric_predictor",
) -> None:
    """Build the Trainer and run ``fit`` (optionally resuming a checkpoint)."""
    if resume is not None and not resume.exists():
        raise FileNotFoundError(f"checkpoint to resume from not found: {resume}")

    trainer = build_trainer(config, wandb_config, run_dir, device, architecture)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=None if resume is None else str(resume),
    )



def apply_overrides(
    config: ExperimentConfig,
    overrides: list[str],
) -> ExperimentConfig:
    """Apply validated dotted-path overrides to a loaded config."""
    if not overrides:
        return config

    # Keep resolved paths and values from the normal config loader.
    data = config.model_dump(mode="python")

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}; expected KEY=VALUE"
            )

        key, raw_value = override.split("=", 1)
        keys = key.split(".")

        if not all(keys):
            raise ValueError(f"Invalid config path: {key!r}")

        # Parse YAML scalar types: floats, integers, booleans, null, lists.
        value = yaml.safe_load(raw_value)

        target = data
        for part in keys[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(
                    f"Unknown config path: {key!r} "
                    f"(missing section {part!r})"
                )
            target = target[part]

        leaf = keys[-1]
        if leaf not in target:
            raise ValueError(f"Unknown config field: {key!r}")

        target[leaf] = value

    # Revalidate types and nested config constraints.
    return ExperimentConfig.model_validate(data)



def main() -> None:
    """parse CLI -> load YAML -> build configs -> build objects -> train."""
    args = parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)

    run_dir = prepare_run_dir(config)
    # The whole-night model trains on whole nights (B, T, N); the
    # per-observation autoencoders on single observations (B, N), the
    # exposures of all nights pooled. Decided by the model config alone.
    datamodule = build_datamodule(
        config.data, config.training, run_dir,
        single_frame=config.model.single_frame,
    )
    model = build_model(config.model, config.training)

    train(
        model=model,
        datamodule=datamodule,
        config=config.training,
        wandb_config=config.wandb,
        run_dir=run_dir,
        device=args.device,
        resume=args.resume,
        architecture=config.model.architecture.value,
    )


if __name__ == "__main__":
    main()
