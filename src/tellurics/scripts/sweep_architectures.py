"""
Architecture sweep with dawgz: train several architectures and find the best.

Same pattern as ``tellurics.scripts.sweep`` (dawgz job -> ``train`` with
``--set`` overrides, a seeded random sample of a grid, a Slurm job array), but
over *architectures*: every architecture has its own manifest and its own
search space, and trials of all of them are scheduled in one array.

The three per-frame autoencoders train on the same 2048-sample data (whole
nights per batch, flattened to frames by the models) with
the same loss (MSE on the telluric transmission), so their
``validation_loss`` values are directly comparable. The whole-night model
(51,557 samples, whole nights) can be added with ``--architectures`` but is
not comparable on that metric.

Run from the repository root::

    # schedule N_TRIALS random trials per architecture on Slurm
    /home/mvasist/data/conda/envs/tellurics/bin/python -m src.tellurics.scripts.sweep_architectures

    # only some architectures / more trials / print the commands only
    python -m src.tellurics.scripts.sweep_architectures --architectures cnn_autoencoder specpt --trials 16
    python -m src.tellurics.scripts.sweep_architectures --dry-run

    # rank finished trials by their best validation_loss
    python -m src.tellurics.scripts.sweep_architectures --summarize

Layout of the runs::

    runs/sweeps/<sweep>/<architecture>/<trial>/   (config.yaml, logs/, checkpoints/)
    runs/sweeps/<sweep>/trials.json               every scheduled trial + its overrides
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import subprocess
import sys
from pathlib import Path

N_TRIALS = 8          # random trials per architecture
RANDOM_SEED = 42      # reproducible trial selection
SWEEP_NAME = "architectures"
THROTTLE = 4          # max concurrently running jobs of the array

# Slurm resources of one trial (one architecture, one hyper-parameter set).
JOB_RESOURCES = dict(gpus=1, ram="16GB", time="12:00:00", partition="gpu")

# Shared by every architecture.
COMMON_SPACE: dict[str, list] = {
    "training.optimizer.learning_rate": [1e-3, 3e-4, 1e-4],
    "training.optimizer.weight_decay": [1e-4, 1e-3],
    "model.dropout": [0.0, 0.1, 0.2],
    "data.seed": [42],
}

# One manifest + search space per architecture (keys are dotted --set paths).
ARCHITECTURES: dict[str, dict] = {
    "cnn_autoencoder": {
        "config": "experiments/single_frame/experiment_cnn_ae.yaml",
        "space": {
            "model.encoder_channels": [
                [16, 32, 64],
                [16, 32, 64, 96, 128],
                [32, 64, 128, 128, 256],
            ],
            "model.kernel_size": [5, 7, 11],
            "model.bottleneck_channels": [4, 8, 16],
            "model.latent_dim": [64, 128, 256],
            "model.metadata_enc_dim": [8, 16, 32],
            "training.batch_size": [2, 4, 8],             # nights (x T frames)
        },
    },
    "transformer_autoencoder": {
        "config": "experiments/single_frame/experiment_transformer_ae.yaml",
        "space": {
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
            "training.batch_size": [1, 2],                # nights (x T frames); ~0.05-0.07 GB/frame fp32
        },
    },
    "specpt": {
        "config": "experiments/single_frame/experiment_specpt.yaml",
        "space": {
            "model.token_stride": [1, 2],
            "model.d_model": [256, 512],
            "model.num_heads": [4, 8],
            "model.num_encoder_layers": [2, 3, 4],
            "model.num_decoder_layers": [2, 3],
            "model.dim_feedforward": [1024, 2048],
            "model.head_hidden": [256, 512],
            "training.batch_size": [1, 2],                # nights (x T frames); ~0.05-0.07 GB/frame fp32
        },
    },
    # Whole-night model: different data and sample type, so its
    # validation_loss is not comparable with the three above. Opt in with
    # --architectures neural_telluric_predictor.
    "neural_telluric_predictor": {
        "config": "experiments/multiple_frames/experiment.yaml",
        "space": {
            "model.num_heads": [4, 8, 16],
            "model.num_queries": [16, 32],
            "training.batch_size": [4, 8],
        },
    },
}
DEFAULT_ARCHITECTURES = ["cnn_autoencoder", "transformer_autoencoder", "specpt"]

SHORT = {
    "cnn_autoencoder": "cnn",
    "transformer_autoencoder": "tae",
    "specpt": "specpt",
    "neural_telluric_predictor": "ntp",
}


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
def sample_trials(architectures: list[str], n_trials: int, seed: int) -> list[dict]:
    """``n_trials`` random grid points per architecture (seeded, no repeats)."""
    rng = random.Random(seed)
    trials = []
    for arch in architectures:
        space = {**COMMON_SPACE, **ARCHITECTURES[arch]["space"]}
        keys = list(space)
        grid = list(itertools.product(*(space[k] for k in keys)))
        picked = rng.sample(grid, k=min(n_trials, len(grid)))
        for i, values in enumerate(picked):
            name = f"{SHORT[arch]}_{i:03d}"
            trials.append({
                "architecture": arch,
                "name": name,
                "config": ARCHITECTURES[arch]["config"],
                "overrides": dict(zip(keys, values)),
                "grid_size": len(grid),
            })
    return trials


def train_command(trial: dict, sweep: str, python: str = "python") -> list[str]:
    """The ``train`` command line of one trial."""
    run_dir = f"runs/sweeps/{sweep}/{trial['architecture']}/{trial['name']}"
    overrides = {
        **trial["overrides"],
        "run_dir": run_dir,
        "wandb.title": f"{sweep}/{trial['name']}",
    }
    cmd = [python, "-m", "src.tellurics.scripts.train",
           "--config", trial["config"], "--device", "gpu"]
    for key, value in overrides.items():
        # YAML flow syntax so lists / floats round-trip through --set
        cmd += ["--set", f"{key}={json.dumps(value)}"]
    return cmd


# --------------------------------------------------------------------------- #
# Summary: rank finished trials by their best validation_loss
# --------------------------------------------------------------------------- #
_CKPT_LOSS = re.compile(r"validation_loss=([0-9.eE+-]+)\.ckpt$")


def summarize(sweep: str) -> list[dict]:
    """Best ``validation_loss`` per trial, from the checkpoint file names."""
    root = Path("runs/sweeps") / sweep
    trials_file = root / "trials.json"
    trials = json.loads(trials_file.read_text()) if trials_file.exists() else []
    rows = []
    for trial in trials:
        run = root / trial["architecture"] / trial["name"]
        losses = [
            float(m.group(1))
            for ckpt in run.glob("checkpoints/**/*.ckpt")
            if (m := _CKPT_LOSS.search(ckpt.name))
        ]
        rows.append({**trial, "best_validation_loss": min(losses) if losses else None})
    done = sorted((r for r in rows if r["best_validation_loss"] is not None),
                  key=lambda r: r["best_validation_loss"])
    pending = [r for r in rows if r["best_validation_loss"] is None]

    print(f"Sweep '{sweep}': {len(done)} finished, {len(pending)} without checkpoints\n")
    print(f"{'rank':>4}  {'trial':<12} {'architecture':<26} {'best val loss':>14}")
    for i, r in enumerate(done, 1):
        print(f"{i:>4}  {r['name']:<12} {r['architecture']:<26} "
              f"{r['best_validation_loss']:>14.6g}")
    if done:
        print("\nBest per architecture:")
        seen = set()
        for r in done:
            if r["architecture"] in seen:
                continue
            seen.add(r["architecture"])
            print(f"  {r['architecture']:<26} {r['name']}  "
                  f"val loss {r['best_validation_loss']:.6g}")
            print(f"    overrides: {json.dumps(r['overrides'])}")
    return done


# --------------------------------------------------------------------------- #
# dawgz
# --------------------------------------------------------------------------- #
def make_job():
    import dawgz

    @dawgz.job(**JOB_RESOURCES)
    def train_trial(command: list[str]) -> None:
        # Use the interpreter running this job (the cluster env) for train.
        subprocess.run([sys.executable, *command[1:]], check=True)

    return train_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHITECTURES,
                        choices=list(ARCHITECTURES))
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="random trials per architecture")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--name", default=SWEEP_NAME, help="sweep name (runs/sweeps/<name>)")
    parser.add_argument("--throttle", type=int, default=THROTTLE)
    parser.add_argument("--backend", default="slurm", choices=["slurm", "async", "dummy"])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the trials and commands, schedule nothing")
    parser.add_argument("--summarize", action="store_true",
                        help="rank the finished trials of --name and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args.name)
        return

    trials = sample_trials(args.architectures, args.trials, args.seed)
    commands = [train_command(t, args.name) for t in trials]

    for arch in args.architectures:
        n = sum(t["architecture"] == arch for t in trials)
        grid = next(t["grid_size"] for t in trials if t["architecture"] == arch)
        print(f"{arch:<26} {n} trials (grid of {grid})")
    print(f"Total trials: {len(trials)}")

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    root = Path("runs/sweeps") / args.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "trials.json").write_text(json.dumps(trials, indent=2))

    import dawgz

    train_trial = make_job()
    jobs = [train_trial(cmd) for cmd in commands]
    dawgz.schedule(
        dawgz.array(*jobs, throttle=args.throttle),
        backend=args.backend,
        name=f"sweep_{args.name}",
    )


if __name__ == "__main__":
    main()
