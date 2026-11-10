#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tellurics.simulation.utils.site_parameters import MOLECULES, read_condition_layout


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild telfit_condition_series.h5 from telluric_templates_nights.h5"
    )
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/mvasist/Tellurics_data/telluric_timeseries/telfit_condition_series.h5"),
    )
    parser.add_argument("--lat", type=float, default=-24.625278, help="observatory latitude, deg")
    parser.add_argument("--alt", type=float, default=2.635, help="observatory altitude, km")
    args = parser.parse_args()

    with h5py.File(args.templates, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.float64)
        columns = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in f["labels"].attrs["columns"]
        ]

    def column(name):
        return labels[:, :, columns.index(name)]

    pressure = column("pressure")
    temperature = column("temperature")
    humidity = column("humidity")
    angle = column("angle")
    airmass = column("airmass")
    time_hours = column("time_hours")[0]
    molecules = {mol: column(mol)[:, 0] for mol in MOLECULES}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("pressure_hpa", data=pressure, compression="gzip")
        f.create_dataset("temperature_k", data=temperature, compression="gzip")
        f.create_dataset("humidity_pct", data=humidity, compression="gzip")
        f.create_dataset("zenith_angle_deg", data=angle, compression="gzip")
        f.create_dataset("airmass", data=airmass, compression="gzip")
        f.create_dataset("time_hours", data=time_hours)
        for mol in MOLECULES:
            f.create_dataset(f"{mol}_ppmv", data=molecules[mol])
        f.attrs["latitude_deg"] = float(args.lat)
        f.attrs["altitude_km"] = float(args.alt)
        f.attrs["source_templates"] = str(args.templates)

    n_series, n_frames, observatory = read_condition_layout(args.output)
    print(f"wrote {args.output}")
    print(f"  n_series={n_series} n_frames={n_frames} observatory={observatory}")
    print(f"  time_hours: {time_hours[0]:.3f} .. {time_hours[-1]:.3f}")


if __name__ == "__main__":
    main()