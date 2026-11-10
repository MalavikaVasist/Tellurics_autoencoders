#!/usr/bin/env python3
"""
Generate telluric transmissions over a full night (time series) in parallel
using dawgz + SLURM.

Method
------
Native TelFit/LBLRTM is generated first (no wavegrid, no resolution), then
convolved to the target resolving power in log-lambda space and resampled onto
the shared constant-R output grid. This matches ``generate_grid.py``.

Usage
-----
    # Submit all condition frames in parallel on SLURM + an aggregation job:
    python simulation/telluric/generate_grid_timeseries.py --parallel \
        --conditions <path>/telfit_condition_series.h5 \
        --batch-size 200

    # Generate a single batch locally (useful for testing):
    python simulation/telluric/generate_grid_timeseries.py \
        --conditions <path>/telfit_condition_series.h5 \
        --array-index 5 --batch-size 200

    # Generate all batches sequentially locally:
    python simulation/telluric/generate_grid_timeseries.py \
        --conditions <path>/telfit_condition_series.h5 \
        --batch-size 200

    # Aggregate existing chunks:
    python simulation/telluric/generate_grid_timeseries.py \
        --conditions <path>/telfit_condition_series.h5 --aggregate
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src` importable so `tellurics` is a package when run as a script
_SRC_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _SRC_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))
from tellurics.simulation.telluric.conditions import ConditionSeries
from tellurics.simulation.telluric.dataset import aggregate, generate_batch
from tellurics.simulation.telluric.model import TelluricSimulation
from tellurics.simulation.utils.running_hpc import create_slurm_workspace
from tellurics.config.paths import resolve_data_path


DEFAULT_CONDITIONS_H5 = (
    _REPO_ROOT / "site_parameter_evolution" / "telfit_condition_series.h5"
)
DEFAULT_OUTDIR = resolve_data_path("telluric_timeseries")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate telluric time-series transmissions")
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS_H5,
                        help="telfit_condition_series.h5 (per-night atmospheric conditions)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR,
                        help="Output directory for chunks and final HDF5")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max flattened condition frames to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="TelFit calls per chunk/job")
    parser.add_argument("--seed", type=int, default=42,
                        help="Retained for CLI compatibility; conditions are deterministic")
    parser.add_argument("--array-index", type=int, default=None,
                        help="Generate a single chunk (for manual runs) locally")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate chunk files into final HDF5")
    parser.add_argument("--parallel", action="store_true",
                        help="Submit SLURM jobs via dawgz to SLURM")
    parser.add_argument("--conda-env", type=str, default="tellurics",
                        help="Conda environment for SLURM jobs")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per SLURM job")
    parser.add_argument("--ram", type=str, default="1GB", help="RAM per SLURM job")
    parser.add_argument("--aggregate-ram", type=str, default="4GB",
                        help="RAM for aggregation SLURM job")
    parser.add_argument("--time", type=str, default="10:00:00",
                        help="Wall time per SLURM job")
    return parser.parse_args()


def main():
    args = parse_args()

    conditions_h5 = args.conditions.expanduser().resolve()
    outdir = args.output_dir.expanduser().resolve()

    conditions = ConditionSeries(conditions_h5)
    total_available = conditions.n_samples
    n_samples = total_available if args.n_samples is None else int(args.n_samples)

    if n_samples < 1 or n_samples > total_available:
        raise ValueError(f"--n-samples must be in [1, {total_available}], got {n_samples}")

    n_jobs = int(np.ceil(n_samples / args.batch_size))
    simulator = TelluricSimulation()

    print("=" * 88)
    print("TELFIT TIME-SERIES GENERATOR")
    print("=" * 88)
    print(f"Conditions:                  {conditions_h5}")
    print(f"Series / frames:             {conditions.n_series} / {conditions.n_frames}")
    print(f"Available TelFit calls:      {total_available}")
    print(f"Calls requested:             {n_samples}")
    print(f"Batch size / jobs:           {args.batch_size} / {n_jobs}")
    print(f"Wavelength range:            {simulator.wave_start:.1f}-{simulator.wave_end:.1f} nm")
    print(f"Target resolving power:      {simulator.target_R:.0f}")
    print(f"Output wavelength points:    {len(simulator.wavelength)}")
    print(f"Output directory:            {outdir}")

    # === Parallel mode: submit generation + aggregation with dependency ===
    if args.parallel:
        from dawgz import job, after, schedule

        @job(array=n_jobs, cpus=args.cpus, ram=args.ram, time=args.time)
        def telluric_job(array_index: int):
            generate_batch(
                array_index=array_index,
                n_samples=n_samples,
                batch_size=args.batch_size,
                conditions_h5=conditions_h5,
                outdir=outdir,
                workdir=create_slurm_workspace(),
                seed=args.seed,
            )

        @after(telluric_job)
        @job(cpus=1, ram=args.aggregate_ram, time="1:00:00")
        def aggregate_job():
            aggregate(outdir)

        schedule(
            aggregate_job,
            name="TelluricTimeseries",
            backend="slurm",
            env=["source ~/.bashrc", f"conda activate {args.conda_env}"],
        )
        print(f"Submitted {n_jobs} generation jobs + 1 aggregation job (with dependency)")
        return

    # === Non-parallel modes ===
    if args.aggregate:
        aggregate(outdir)
        return

    if args.array_index is not None:
        generate_batch(
            array_index=args.array_index,
            n_samples=n_samples,
            batch_size=args.batch_size,
            conditions_h5=conditions_h5,
            outdir=outdir,
            workdir= None,
            seed=args.seed,
        )
        return

    # Sequential: generate all then aggregate (still needs SLURM_TMPDIR).
    print(f"Running {n_jobs} batches sequentially...")
    for i in range(n_jobs):
        generate_batch(
            array_index=i,
            n_samples=n_samples,
            batch_size=args.batch_size,
            conditions_h5=conditions_h5,
            outdir=outdir,
            workdir= None,
            seed=args.seed,
        )

    print("\nAll batches done. Aggregating...")
    aggregate(outdir)


if __name__ == "__main__":
    main()
