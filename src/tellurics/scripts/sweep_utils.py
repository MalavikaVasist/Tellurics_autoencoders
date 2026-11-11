"""Shared helpers of the dawgz sweep scripts (one script per architecture).

Each sweep script (``sweep_cnn_ae.py``, ``sweep_transformer_ae.py``) only
declares its manifest, search space and Slurm resources and calls
:func:`run_sweep`. The pattern is the one of ``sweep.py``: a dawgz job runs
``tellurics.scripts.train`` with ``--set`` overrides for every trial of a
seeded random sample of the grid, scheduled as one Slurm job array.

Every script supports::

    --trials N      random trials (default: the script's N_TRIALS)
    --name NAME     sweep name -> runs/sweeps/<NAME>/
    --dry-run       print the commands, schedule nothing
    --summarize     rank finished trials of --name by their best validation_loss
    --backend       slurm (default) | async | dummy
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

RUNS = Path("runs/sweeps")


def sample_trials(space: dict[str, list], n_trials: int, seed: int, prefix: str) -> list[dict]:
    """``n_trials`` random points of the grid ``space`` (seeded, no repeats)."""
    keys = list(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    picked = random.Random(seed).sample(grid, k=min(n_trials, len(grid)))
    return [
        {"name": f"{prefix}_{i:03d}", "overrides": dict(zip(keys, values)),
         "grid_size": len(grid)}
        for i, values in enumerate(picked)
    ]


def train_command(config: str, trial: dict, sweep: str, python: str = "python") -> list[str]:
    """The ``train`` command line of one trial."""
    overrides = {
        **trial["overrides"],
        "run_dir": str(RUNS / sweep / trial["name"]),
        "wandb.title": f"{sweep}/{trial['name']}",
    }
    cmd = [python, "-m", "src.tellurics.scripts.train",
           "--config", config, "--device", "gpu"]
    for key, value in overrides.items():
        # JSON is valid YAML flow syntax, so lists / floats round-trip through --set
        cmd += ["--set", f"{key}={json.dumps(value)}"]
    return cmd


_CKPT_LOSS = re.compile(r"validation_loss=([0-9.eE+-]+)\.ckpt$")


def summarize(sweep: str) -> list[dict]:
    """Rank the trials of a sweep by the best ``validation_loss`` in their checkpoints."""
    root = RUNS / sweep
    trials_file = root / "trials.json"
    if not trials_file.exists():
        print(f"No {trials_file}: nothing scheduled under the name '{sweep}'.")
        return []
    rows = []
    for trial in json.loads(trials_file.read_text()):
        losses = [
            float(m.group(1))
            for ckpt in (root / trial["name"]).glob("checkpoints/**/*.ckpt")
            if (m := _CKPT_LOSS.search(ckpt.name))
        ]
        rows.append({**trial, "best_validation_loss": min(losses) if losses else None})
    done = sorted((r for r in rows if r["best_validation_loss"] is not None),
                  key=lambda r: r["best_validation_loss"])
    print(f"Sweep '{sweep}': {len(done)} finished, {len(rows) - len(done)} without checkpoints\n")
    print(f"{'rank':>4}  {'trial':<12} {'best val loss':>14}")
    for i, r in enumerate(done, 1):
        print(f"{i:>4}  {r['name']:<12} {r['best_validation_loss']:>14.6g}")
    if done:
        print(f"\nBest: {done[0]['name']}  overrides: {json.dumps(done[0]['overrides'])}")
    return done


def run_sweep(
    *,
    description: str,
    config: str,
    space: dict[str, list],
    prefix: str,
    sweep_name: str,
    n_trials: int,
    seed: int,
    throttle: int,
    resources: dict,
) -> None:
    """CLI entry point shared by the per-architecture sweep scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trials", type=int, default=n_trials, help="random trials")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--name", default=sweep_name, help="sweep name (runs/sweeps/<name>)")
    parser.add_argument("--throttle", type=int, default=throttle,
                        help="max concurrently running jobs")
    parser.add_argument("--backend", default="slurm", choices=["slurm", "async", "dummy"])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands, schedule nothing")
    parser.add_argument("--summarize", action="store_true",
                        help="rank the finished trials of --name and exit")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.name)
        return

    trials = sample_trials(space, args.trials, args.seed, prefix)
    commands = [train_command(config, t, args.name) for t in trials]
    print(f"{args.name}: {len(trials)} trials (grid of {trials[0]['grid_size']}) from {config}")

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    root = RUNS / args.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "trials.json").write_text(json.dumps(trials, indent=2))

    import dawgz

    @dawgz.job(**resources)
    def train_trial(command: list[str]) -> None:
        # Use the interpreter running this job (the cluster env) for train.
        subprocess.run([sys.executable, *command[1:]], check=True)

    dawgz.schedule(
        dawgz.array(*(train_trial(cmd) for cmd in commands), throttle=args.throttle),
        backend=args.backend,
        name=args.name,
    )
