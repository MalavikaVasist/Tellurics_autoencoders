#!/usr/bin/env python3
"""
Generate telluric template grid in parallel using dawgz + SLURM.

Usage:
    # Submit all generation jobs in parallel on SLURM, followed by an
    # aggregation job that starts automatically after generation
    # jobs have completed:
    python generate_telluric_grid.py --parallel --n-samples 262000 --batch-size 200

    # Generate a single batch manually.
    # Here, --array-index specifies the batch/chunk index:
    python generate_telluric_grid.py --array-index 5

    # Aggregate existing chunk files locally:
    python generate_telluric_grid.py --aggregate --n-samples 262000 --batch-size 200
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

# Make `src` importable so `tellurics` is a package when run as a script
_SRC_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _SRC_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))
from tellurics.simulation.utils.running_hpc import create_slurm_workspace
from tellurics.simulation.utils.spectral import build_constant_R_grid, convolve_and_resample
from tellurics.simulation.utils.telfit_home import add_telfit_to_path
from tellurics.config.paths import resolve_data_path


# === Configuration ===
PARAM_RANGES = {
    "pressure": (750, 850),  # hPa
    "temperature": (270, 300),  # Kelvin
    "humidity": (10, 100),  # percent
}

FIXED_PARAMS = {
    "lat": 30.6,
    "alt": 2.1,
    "co2": 368.5,  # ppmv
    "o3": 3.9e-2,  # ppmv
    "n2o": 1e-4,  # ppmv
    "co": 0.14,  # ppmv
    "ch4": 1.8,  # ppmv
    "o2": 2.1e5,  # ppmv
    "no": 1.1e-19,  # ppmv
    "so2": 1e-4,  # ppmv
    "no2": 1e-4,  # ppmv
    "nh3": 1e-4,  # ppmv
    "hno3": 5.6e-4,  # ppmv
    "angle": 45.0, #degrees
}

WAVESTART_NM = 800.0
WAVEEND_NM = 950.0

# Target resolving power and output grid sampling (must match Phoenix pool)
TARGET_R = 100_000
SAMPLES_PER_RESEL = 3.0
WAVE_PADDING_NM = 0.5  # extend native TelFit range for clean convolution edges

# Repo layout: this file is at simulation/telluric/generate_grid.py
REPO_ROOT = _REPO_ROOT

# Output telluric templates under <TELLURICS_DATA>/telluric
OUTDIR = resolve_data_path("telluric")
CHUNKS_DIR = OUTDIR / "chunks"

# TelFit lives outside the repo; its ``src`` is resolved from $TELFIT_HOME (see
# simulation.utils.telfit_home) and added to sys.path in ``generate_batch``.


def get_output_wavegrid():
    """Build the constant-R output wavelength grid (in nm).

    This matches the Phoenix pool grid so tellurics and stellar spectra
    share the same wavelength axis.
    """
    return build_constant_R_grid(
        WAVESTART_NM, WAVEEND_NM, TARGET_R, SAMPLES_PER_RESEL,
    ).astype(np.float64)


def generate_all_samples(n_samples, seed=42):
    """Generate Latin Hypercube samples for the full grid."""
    param_names = list(PARAM_RANGES.keys())
    bounds_low = [v[0] for v in PARAM_RANGES.values()]
    bounds_high = [v[1] for v in PARAM_RANGES.values()]

    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    samples_unit = sampler.random(n=n_samples)
    samples = qmc.scale(samples_unit, bounds_low, bounds_high)

    return param_names, samples


def generate_batch(array_index: int, n_samples: int, batch_size: int, seed: int = 42):
    """Generate a single batch of telluric models and save to HDF5 chunk."""
    add_telfit_to_path()
    from telfit import Modeler

    OUTDIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)

    chunk_path = CHUNKS_DIR / f"chunk_{array_index:06d}.h5"
    if chunk_path.exists():
        print(f"Chunk {array_index} already exists, skipping.")
        return

    workdir = create_slurm_workspace()

    # Generate samples (same seed = same LHS for all workers)
    param_names, all_samples = generate_all_samples(n_samples, seed=seed)

    # Determine this batch's slice
    start = array_index * batch_size
    end = min(start + batch_size, n_samples)
    if start >= n_samples:
        print(f"Chunk {array_index}: no samples (start={start} >= n_samples={n_samples})")
        return

    batch_samples = all_samples[start:end]
    print(f"Chunk {array_index}: samples [{start}:{end}] ({len(batch_samples)} models)")

    # Build the constant-R output wavegrid (shared with Phoenix pool)
    wavegrid_nm = get_output_wavegrid()

    # Generate native TelFit models over a padded range so the convolution
    # kernel has clean edges, then convolve to TARGET_R and resample.
    lowfreq = 1e7 / (WAVEEND_NM + WAVE_PADDING_NM)
    highfreq = 1e7 / (WAVESTART_NM - WAVE_PADDING_NM)

    modeler = Modeler(print_lblrtm_output=False, debug=False)
    transmission_list = []
    labels_list = []
    wavelength = wavegrid_nm.astype(np.float32)
    failed_indices = []

    for i, sample in enumerate(tqdm(batch_samples)):
        params = dict(zip(param_names, sample.astype(float)))

        if i % 10 == 0:
            print(f"  Chunk {array_index}: {i+1}/{len(batch_samples)}")

        try:
            # Native LBLRTM model (no wavegrid, no resolution).
            model = modeler.MakeModel(
                lowfreq=lowfreq,
                highfreq=highfreq,
                workdir=str(workdir),
                vac2air=False,
                libfile="",
                save=False,
                **params,
                **FIXED_PARAMS,
            )

            # Convolve to TARGET_R and resample onto the shared output grid.
            _, transmission = convolve_and_resample(
                np.asarray(model.x, dtype=np.float64),
                np.asarray(model.y, dtype=np.float64),
                target_R=TARGET_R,
                target_wave=wavegrid_nm,
            )

            transmission_list.append(transmission.astype(np.float32))
            labels_list.append(sample.astype(np.float32))

        except Exception as e:
            print(f"  FAILED at sample {start + i}: {type(e).__name__}: {e}")
            print(f"    params: {params}")
            failed_indices.append(start + i)
            continue

    if not transmission_list:
        print(f"Chunk {array_index}: no models generated!")
        return

    transmission = np.stack(transmission_list)
    labels = np.stack(labels_list)

    with h5py.File(chunk_path, "w") as hf:
        hf.create_dataset("wavelength", data=wavelength)
        hf.create_dataset("transmission", data=transmission)
        hf.create_dataset("labels", data=labels)
        hf.create_dataset("failed_indices", data=np.asarray(failed_indices, dtype=np.int64))
        

        hf["labels"].attrs["columns"] = param_names

    print(f"Chunk {array_index} done: {len(transmission_list)} models succeeded-> {chunk_path}, {len(failed_indices)} failed")


def aggregate():
    """Combine all chunk HDF5 files into a single telluric_templates.h5."""
    final_path = OUTDIR / "telluric_templates.h5"
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.h5"))

    if not chunk_files:
        print(f"No chunk files found in {CHUNKS_DIR}")
        return

    print(f"Aggregating {len(chunk_files)} chunk files...")

    all_transmission = []
    all_labels = []
    wavelength = None
    param_names = None


    for cf in chunk_files:
        with h5py.File(cf, "r") as hf:
            if wavelength is None:
                wavelength = hf["wavelength"][:]
                param_names = list(hf["labels"].attrs["columns"])

            all_transmission.append(hf["transmission"][:])
            all_labels.append(hf["labels"][:])

    transmission = np.concatenate(all_transmission, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Total models: {transmission.shape[0]}")
    print(f"Wavelength points: {wavelength.shape[0]}")
    print(f"Parameters: {param_names}")

    with h5py.File(final_path, "w") as hf:
        hf.create_dataset("wavelength", data=wavelength)
        hf.create_dataset("transmission", data=transmission)
        hf.create_dataset("labels", data=labels)
        hf["labels"].attrs["columns"] = param_names
        for k, v in FIXED_PARAMS.items():
            hf.attrs[f"fixed_{k}"] = v
        hf.attrs["wavestart_nm"] = WAVESTART_NM
        hf.attrs["waveend_nm"] = WAVEEND_NM
        hf.attrs["target_resolution"] = TARGET_R
        hf.attrs["samples_per_resolution_element"] = SAMPLES_PER_RESEL

    print(f"\nSaved: {final_path}")
    print(f"  wavelength:   {wavelength.shape}")
    print(f"  transmission: {transmission.shape}")
    print(f"  labels:       {labels.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate telluric template grid")
    parser.add_argument("--n-samples", type=int, default=262000,
                        help="Total number of LHS samples")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Samples per chunk/job")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS")
    parser.add_argument("--array-index", type=int, default=None,
                        help="Generate a single chunk (for manual runs)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate chunk files into final HDF5")
    parser.add_argument("--parallel", action="store_true",
                        help="Submit SLURM jobs via dawgz")
    parser.add_argument("--conda-env", type=str, default="tellurics",
                        help="Conda environment for SLURM jobs")
    parser.add_argument("--cpus", type=int, default=1,
                        help="CPUs per SLURM job")
    parser.add_argument("--ram", type=str, default="1GB",
                        help="RAM per SLURM job")
    parser.add_argument("--aggregate-ram", type=str, default="32GB",
                        help="RAM for aggregation SLURM job")
    parser.add_argument("--time", type=str, default="10:00:00",
                        help="Wall time per SLURM job")
    return parser.parse_args()


def main():
    args = parse_args()

    n_jobs = int(np.ceil(args.n_samples / args.batch_size))

    # === Parallel mode: submit generation + aggregation with dependency ===
    if args.parallel:
        from dawgz import job, after, schedule

        @job(array=n_jobs, cpus=args.cpus, ram=args.ram, time=args.time)
        def telluric_job(array_index: int):
            generate_batch(
                array_index=array_index,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                seed=args.seed,
            )

        @after(telluric_job)
        @job(cpus=1, ram=args.aggregate_ram, time="1:00:00")
        def aggregate_job():
            aggregate()

        schedule(
            aggregate_job,
            name="TelluricGrid",
            backend="slurm",
            env=[
                "source ~/.bashrc",
                f"conda activate {args.conda_env}",
            ],
        )
        print(f"Submitted {n_jobs} generation jobs + 1 aggregation job (with dependency)")
        return

    # === Non-parallel modes ===
       
    # Aggregate only
    if args.aggregate:
        aggregate()
        return

    # Single batch
    if args.array_index is not None:
        generate_batch(
            array_index=args.array_index,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return

    # Sequential: generate all then aggregate
    print(f"Running {n_jobs} batches sequentially...")
    for i in range(n_jobs):
        generate_batch(
            array_index=i,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    print("\nAll batches done. Aggregating...")
    aggregate()


if __name__ == "__main__":
    main()
