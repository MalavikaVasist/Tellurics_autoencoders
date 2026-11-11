"""
dawgz sweep of the transformer autoencoder (one observation in, one telluric out).

/home/mvasist/data/conda/envs/tellurics/bin/python -m src.tellurics.scripts.sweep_transformer_ae

    python -m src.tellurics.scripts.sweep_transformer_ae --dry-run      # print the commands
    python -m src.tellurics.scripts.sweep_transformer_ae --trials 16    # more trials
    python -m src.tellurics.scripts.sweep_transformer_ae --summarize    # rank finished trials

Runs land in runs/sweeps/<SWEEP_NAME>/tae_###/. Edit the settings below freely;
the shared machinery lives in sweep_utils.py.
"""

from .sweep_utils import run_sweep

CONFIG = "experiments/single_frame/experiment_transformer_ae.yaml"
SWEEP_NAME = "transformer_ae"
N_TRIALS = 8
RANDOM_SEED = 42          # reproducible trial selection
THROTTLE = 4              # max concurrently running jobs

# Slurm resources of one trial.
JOB_RESOURCES = dict(gpus=1, ram="16GB", time="12:00:00", partition="gpu")

# Search space: dotted --set paths -> candidate values.
# Memory: ~0.05 GB per observation in fp32 at the manifest's settings.
SEARCH_SPACE = {
    "training.optimizer.learning_rate": [1e-3, 3e-4, 1e-4],
    "training.optimizer.weight_decay": [1e-4, 1e-3],
    "training.batch_size": [16, 32, 64],           # observations per batch
    "model.dropout": [0.0, 0.1, 0.2],
    "model.patch_size": [8, 16, 32],
    "model.d_model": [128, 256],
    "model.num_heads": [4, 8],
    "model.num_encoder_layers": [2, 4, 6],
    "model.num_decoder_layers": [2, 4],
    "model.dim_feedforward": [512, 1024],
    "model.num_latents": [8, 16, 32],
    "model.latent_dim": [32, 64],
    "model.wavelength_frequencies": [6, 8, 10],
    "model.metadata_enc_dim": [8, 16, 32],
    "data.seed": [42],
}


if __name__ == "__main__":
    run_sweep(
        description="dawgz sweep of the transformer autoencoder",
        config=CONFIG,
        space=SEARCH_SPACE,
        prefix="tae",
        sweep_name=SWEEP_NAME,
        n_trials=N_TRIALS,
        seed=RANDOM_SEED,
        throttle=THROTTLE,
        resources=JOB_RESOURCES,
    )
