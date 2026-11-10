#!/usr/bin/env python3
"""
download_pool.py

Download a pool of PHOENIX HiRes stellar spectra, crop to a wavelength
range, and properly convolve them to a target resolving power using a
Gaussian instrumental profile in ln(λ) space.

Unlike the naive resample-only approach, this convolves the native
R~500k spectrum down to the target R before resampling, so the stored
spectra genuinely have the requested resolving power.

Reads a CSV pool definition (see configs/phoenix_pool.csv):
    label,teff,logg,feh,wmin,wmax,resolution,samples_per_resolution_element

Usage:
    python simulation/phoenix/download_pool.py
    python simulation/phoenix/download_pool.py --config configs/phoenix_pool.csv
    python simulation/phoenix/download_pool.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from astropy.io import fits

# Make `src` importable so `tellurics` is a package when run as a script
_SRC_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = _SRC_ROOT.parent
sys.path.insert(0, str(_SRC_ROOT))
from tellurics.simulation.utils.spectral import convolve_and_resample, build_constant_R_grid
from tellurics.config.paths import resolve_data_path


PHOENIX_BASE_URL   = "https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS"
PHOENIX_MODEL_GRID = "PHOENIX-ACES-AGSS-COND-2011"
PHOENIX_WAVE_FILE  = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

# Data lives under <TELLURICS_DATA>/phoenix
DATA_ROOT    = resolve_data_path("phoenix")
DOWNLOAD_DIR = DATA_ROOT / "downloads"
OUTPUT_DIR   = DATA_ROOT / "convolved"


def metallicity_strings(feh):
    """Convert numeric [Fe/H] into PHOENIX folder/file metallicity strings."""
    feh = round(float(feh), 1)
    if feh == 0.0:
        return "Z-0.0", "-0.0"
    sign = "+" if feh > 0 else "-"
    return f"Z{sign}{abs(feh):.1f}", f"{sign}{abs(feh):.1f}"


def download_if_needed(url, output_path):
    if output_path.exists():
        print(f"  already downloaded: {output_path.name}")
        return
    print(f"  downloading: {url}")
    urlretrieve(url, output_path)


def process_model(row, wave_file, overwrite):
    """Download, crop, convolve, and save one PHOENIX model."""
    label = row["label"]
    teff  = int(row["teff"])
    logg  = float(row["logg"])
    feh   = float(row["feh"])
    wmin  = float(row["wmin"])
    wmax  = float(row["wmax"])
    R     = float(row["resolution"])
    spr   = float(row["samples_per_resolution_element"])

    grid_tag = f"R{int(R)}_s{spr:g}"
    out_name = f"phoenix_{label}_T{teff}_logg{logg:.1f}_feh{feh:+.1f}_{grid_tag}.fits"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not overwrite:
        print(f"  output exists, skipping: {out_name}")
        return

    # Download the spectrum
    z_folder, z_file = metallicity_strings(feh)
    spectrum_name = f"lte{teff:05d}-{logg:.2f}{z_file}.{PHOENIX_MODEL_GRID}-HiRes.fits"
    spectrum_url  = f"{PHOENIX_BASE_URL}/{PHOENIX_MODEL_GRID}/{z_folder}/{spectrum_name}"
    spectrum_file = DOWNLOAD_DIR / spectrum_name
    download_if_needed(spectrum_url, spectrum_file)

    # Load and crop
    wave_angstrom = fits.getdata(wave_file)
    flux_raw      = fits.getdata(spectrum_file)
    wave_um       = wave_angstrom * 1e-4
    
    # wmin = 0.800000801
    cut = (wave_um >= wmin) & (wave_um <= wmax)
    wave_cut = np.asarray(wave_um[cut], dtype=np.float64)
    flux_cut = np.asarray(flux_raw[cut], dtype=np.float64)

    good = np.isfinite(wave_cut) & np.isfinite(flux_cut)
    wave_cut, flux_cut = wave_cut[good], flux_cut[good]

    if len(wave_cut) == 0:
        print(f"  WARNING: no valid points in range for {label}, skipping")
        return

    # Build the output grid and convolve+resample
    target_grid = build_constant_R_grid(wmin, wmax, R, spr)
    _, flux_out = convolve_and_resample(
        wave_cut, flux_cut, target_R=R, target_wave=target_grid,
    )

    # Save
    columns = [
        fits.Column(name="WAVE_MICRON", array=target_grid.astype("float64"), format="D"),
        fits.Column(name="FLUX", array=flux_out.astype("float32"), format="E"),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.name = "SPECTRUM"
    hdu.header["LABEL"]   = label
    hdu.header["TEFF"]    = teff
    hdu.header["LOGG"]    = logg
    hdu.header["FEH"]     = feh
    hdu.header["WMIN"]    = wmin
    hdu.header["WMAX"]    = wmax
    hdu.header["R"]       = R
    hdu.header["SPRES"]   = spr
    hdu.header["NPOINTS"] = len(target_grid)
    hdu.header["SRC"]     = "PHOENIX-HiRes"
    hdu.header["COMMENT"] = f"Convolved to R={int(R)} with Gaussian in ln(lambda)"

    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = f"{label}_T{teff}"

    fits.HDUList([primary, hdu]).writeto(out_path, overwrite=True)
    print(f"  saved: {out_name}  ({len(target_grid)} points)")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and convolve a PHOENIX pool")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "configs" / "phoenix_pool.csv",
        help="CSV file defining the pool of stellar models",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing convolved FITS files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download the shared wavelength grid once
    wave_file = DOWNLOAD_DIR / PHOENIX_WAVE_FILE
    print("Wavelength grid:")
    download_if_needed(f"{PHOENIX_BASE_URL}/{PHOENIX_WAVE_FILE}", wave_file)

    with args.config.open() as f:
        rows = list(csv.DictReader(f))

    print(f"\nProcessing {len(rows)} PHOENIX models from {args.config.name}\n")
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['label']} "
              f"(Teff={row['teff']}, logg={row['logg']}, [Fe/H]={row['feh']})")
        try:
            process_model(row, wave_file, args.overwrite)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. Convolved spectra in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
