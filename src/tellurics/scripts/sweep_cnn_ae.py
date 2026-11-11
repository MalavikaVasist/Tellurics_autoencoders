"""
dawgz sweep of the CNN autoencoder (one observation in, one telluric out).

/home/mvasist/data/conda/envs/tellurics/bin/python -m src.tellurics.scripts.sweep_cnn_ae

    python -m src.tellurics.scripts.sweep_cnn_ae --dry-run      # print the commands
    python -m src.tellurics.scripts.sweep_cnn_ae --trials 16    # more trials
    python -m src.tellurics.scripts.sweep_cnn_ae --summarize    # rank finished trials

Runs land in runs/sweeps/<SWEEP_NAME>/cnn_###/. Edit the settings below freely;
the shared machinery lives in sweep_utils.py.
"""

from .sweep_utils import run_sweep

CONFIG = "experiments/single_frame/experiment_cnn_ae.yaml"
SWEEP_NAME = "cnn_ae"
N_TRIALS = 8
RANDOM_SEED = 42          # reproducible trial selection
THROTTLE = 4              # max concurrently running jobs

# Slurm resources of one trial.
JOB_RESOURCES = dict(gpus=1, ram="16GB", time="12:00:00", partition="gpu")

# Search space: dotted --set paths -> candidate values.
SEARCH_SPACE = {
    "training.optimizer.learning_rate": [1e-3, 3e-4, 1e-4],
    "training.optimizer.weight_decay": [1e-4, 1e-3],
    "training.batch_size": [32, 64, 128],          # observations per batch
    "model.dropout": [0.0, 0.1, 0.2],
    "model.encoder_channels": [
        [16, 32, 64],
        [16, 32, 64, 96, 128],
        [32, 64, 128, 128, 256],
    ],
    "model.kernel_size": [5, 7, 11],
    "model.bottleneck_channels": [4, 8, 16],
    "model.latent_dim": [64, 128, 256],
    "model.metadata_enc_dim": [8, 16, 32],
    "data.seed": [42],
}


if __name__ == "__main__":
    run_sweep(
        description="dawgz sweep of the CNN autoencoder",
        config=CONFIG,
        space=SEARCH_SPACE,
        prefix="cnn",
        sweep_name=SWEEP_NAME,
        n_trials=N_TRIALS,
        seed=RANDOM_SEED,
        throttle=THROTTLE,
        resources=JOB_RESOURCES,
    )
