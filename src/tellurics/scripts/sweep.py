
'''
/home/mvasist/data/conda/envs/tellurics/bin/python -m src.tellurics.scripts.sweep
'''

import itertools
import random
import subprocess

import dawgz

N_TRIALS = 1
RANDOM_SEED = 42  # Reproducible selection

# Hyperparameter search space
LEARNING_RATES = [1e-3, 1e-4, 1e-5]
WEIGHT_DECAYS = [1e-2, 1e-3, 1e-4]
BATCH_SIZES = [256, 512, 1024]
MAX_EPOCHS = [100, 150, 200]
DROPOUTS = [0.1, 0.2, 0.3]
NUM_HEADS = [4, 8, 16, 32]
NUM_QUERIES = [16, 32]
SEEDS = [42]


@dawgz.job(
    gpus=1,
    ram="8GB",
    time="01:00:00",
    partition="gpu",
)
def train_trial(
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    dropout: float,
    num_heads: int,
    num_queries: int,
    seed: int,
):
    trial_name = (
        f"lr{learning_rate:g}"
        f"_wd{weight_decay:g}"
        f"_bs{batch_size}"
        f"_ep{max_epochs}"
        f"_do{dropout:g}"
        f"_heads{num_heads}"
        f"_q{num_queries}"
        f"_seed{seed}"
    )

    subprocess.run(
        [
            "python",
            "-m",
            "src.tellurics.scripts.train",
            "--config",
            "experiments/experiment.yaml",
            "--device",
            "gpu",
            "--set",
            f"training.optimizer.learning_rate={learning_rate}",
            "--set",
            f"training.optimizer.weight_decay={weight_decay}",
            "--set",
            f"training.batch_size={batch_size}",
            "--set",
            f"training.max_epochs={max_epochs}",
            "--set",
            f"model.dropout={dropout}",
            "--set",
            f"model.num_heads={num_heads}",
            "--set",
            f"model.num_queries={num_queries}",
            "--set",
            f"data.seed={seed}",
            "--set",
            f"run_dir=runs/{trial_name}",
            "--set",
            f"wandb.title={trial_name}",
        ],
        check=True,
    )


if __name__ == "__main__":
    parameter_grid = list(itertools.product(
        LEARNING_RATES,
        WEIGHT_DECAYS,
        BATCH_SIZES,
        MAX_EPOCHS,
        DROPOUTS,
        NUM_HEADS,
        NUM_QUERIES,
        SEEDS,
    ))

    rng = random.Random(RANDOM_SEED)
    selected_trials = rng.sample(parameter_grid, k=min(N_TRIALS, len(parameter_grid)))

    jobs = [
        train_trial(lr, wd, bs, epochs, dropout, heads, queries, seed)
        for lr, wd, bs, epochs, dropout, heads, queries, seed in selected_trials
    ]

    print(f"Total combinations: {len(parameter_grid)}")
    print(f"Selected trials: {len(jobs)}")

    dawgz.schedule(
        dawgz.array(*jobs, throttle=2),
        backend="slurm",
    )
