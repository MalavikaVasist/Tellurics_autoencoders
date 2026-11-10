#!/usr/bin/env python3
"""
make_observations.py

Generate synthetic observed spectra by multiplying convolved PHOENIX
stellar fluxes by TelFit telluric transmissions:

    observed_flux = stellar_flux * telluric_transmission

Both inputs must share the same constant-resolution (constant-R)
wavelength grid, generated using matching resolution settings by
simulation/phoenix/download_pool.py and simulation/telluric/generate_grid.py.

The training task is to predict telluric transmission from observed flux.
The output HDF5 file stores the synthetic observed spectra, their
ground-truth telluric transmissions (training targets), and the stellar
fluxes used to generate them for reference.

Usage:
    python simulation/combine/make_observations.py \\
        --telluric "$TELLURICS_DATA/telluric_timeseries/telluric_templates.h5" \\
        --phoenix-dir "$TELLURICS_DATA/phoenix/convolved" \\
        --output "$TELLURICS_DATA/observations/observations.h5"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[4]

from tellurics.config.paths import resolve_data_path

DEFAULT_TELLURIC   = resolve_data_path("telluric") / "telluric_templates.h5"
DEFAULT_PHOENIX_DIR = resolve_data_path("phoenix") / "convolved"
DEFAULT_OUTPUT     = resolve_data_path("observations") / "observations.h5"


def load_phoenix_pool(phoenix_dir: Path):
    """Load all convolved PHOENIX spectra. Returns wave_nm and a dict of spectra."""
    fits_files = sorted(phoenix_dir.glob("phoenix_*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"No PHOENIX FITS files in {phoenix_dir}")

    wave_nm = None
    spectra = {}
    labels = {}
    for f in fits_files:
        with fits.open(f) as hdul:
            data = hdul["SPECTRUM"].data
            header = hdul["SPECTRUM"].header
            w = np.asarray(data["WAVE_MICRON"], dtype=np.float64) * 1e3
            flux = np.asarray(data["FLUX"], dtype=np.float64)

        if wave_nm is None:
            wave_nm = w
        elif w.shape != wave_nm.shape or not np.allclose(w, wave_nm, atol=1e-6):
            raise ValueError(f"Wavelength grid mismatch in {f.name}")

        spectra[f.stem] = flux
        labels[f.stem] = {
            "teff": header.get("TEFF"),
            "logg": header.get("LOGG"),
            "feh": header.get("FEH"),
        }

    return wave_nm, spectra, labels


def make_observations(telluric_h5, phoenix_dir, output_h5, normalize, seed):
    rng = np.random.default_rng(seed)

    # Load telluric grid
    with h5py.File(telluric_h5, "r") as hf:
        tell_wave = hf["wavelength"][:].astype(np.float64)
        transmission = hf["transmission"][:]  # (N, D)
        tell_labels = hf["labels"][:]
        tell_columns = list(hf["labels"].attrs["columns"])

    # Load PHOENIX pool
    phoenix_wave, phoenix_spectra, phoenix_labels = load_phoenix_pool(phoenix_dir)

    # Verify grids match
    if transmission.shape[1] != len(phoenix_wave):
        raise ValueError(
            f"Grid size mismatch: telluric {transmission.shape[1]} "
            f"vs phoenix {len(phoenix_wave)}"
        )
    if not np.allclose(tell_wave, phoenix_wave, atol=1e-6):
        raise ValueError("Telluric and PHOENIX wavelength grids differ")

    n_obs = transmission.shape[0]
    n_wave = transmission.shape[1]
    star_keys = list(phoenix_spectra.keys())

    # For each telluric, assign a random stellar spectrum
    star_choices = rng.integers(0, len(star_keys), size=n_obs)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as out:
        out.create_dataset("wavelength", data=phoenix_wave.astype(np.float32))
        out["wavelength"].attrs["unit"] = "nm"

        ds_obs = out.create_dataset(
            "observed", shape=(n_obs, n_wave), dtype=np.float32,
            chunks=(1, n_wave), compression="gzip", compression_opts=4,
        )
        ds_tell = out.create_dataset(
            "telluric", shape=(n_obs, n_wave), dtype=np.float32,
            chunks=(1, n_wave), compression="gzip", compression_opts=4,
        )
        ds_star_idx = out.create_dataset("star_index", shape=(n_obs,), dtype=np.int32)
        ds_tell_labels = out.create_dataset("telluric_labels", data=tell_labels)
        ds_tell_labels.attrs["columns"] = tell_columns

        # Store the star key strings for reference
        out.create_dataset(
            "star_keys",
            data=np.array(star_keys, dtype=h5py.string_dtype()),
        )

        for i in range(n_obs):
            star_key = star_keys[star_choices[i]]
            stellar = phoenix_spectra[star_key]

            if normalize:
                stellar = stellar / np.max(stellar)

            observed = stellar * transmission[i]

            ds_obs[i] = observed.astype(np.float32)
            ds_tell[i] = transmission[i].astype(np.float32)
            ds_star_idx[i] = star_choices[i]

            if i % 500 == 0:
                print(f"  {i}/{n_obs} observations")

        out.attrs["n_observations"] = n_obs
        out.attrs["n_stars"] = len(star_keys)
        out.attrs["normalized_stellar"] = bool(normalize)

    print(f"\nSaved {n_obs} observations -> {output_h5}")
    print(f"  observed:  ({n_obs}, {n_wave})")
    print(f"  telluric:  ({n_obs}, {n_wave})")
    print(f"  stars used: {len(star_keys)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine stellar x telluric -> observations")
    parser.add_argument("--telluric", type=Path, default=DEFAULT_TELLURIC)
    parser.add_argument("--phoenix-dir", type=Path, default=DEFAULT_PHOENIX_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize each stellar spectrum by its max before multiplying")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    make_observations(
        telluric_h5=args.telluric.expanduser().resolve(),
        phoenix_dir=args.phoenix_dir.expanduser().resolve(),
        output_h5=args.output.expanduser().resolve(),
        normalize=args.normalize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
