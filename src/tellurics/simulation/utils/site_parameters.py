"""
Site parameter (atmospheric condition series) loading utilities.

The condition series HDF5 describes how the atmosphere evolves through an
observing night: frame-wise quantities (pressure, temperature, humidity,
zenith angle, airmass) plus series-level molecular abundances.
"""

from pathlib import Path

import h5py

# Molecules stored as series-level (constant per night) abundances.
MOLECULES = (
    "co2", "o3", "n2o", "co", "ch4", "o2",
    "no", "so2", "no2", "nh3", "hno3",
)


def read_condition_layout(conditions_h5: Path):
    """Read dimensions and global configuration from a condition series file.

    Parameters
    ----------
    conditions_h5 : Path
        Path to ``telfit_condition_series.h5`` (produced by the site-parameter
        builder). Expected datasets:
        - frame-wise (N_series, N_frames): pressure_hpa, temperature_k,
          humidity_pct, zenith_angle_deg, airmass
        - series-level (N_series,): {mol}_ppmv for each molecule
        - optional (N_frames,): time_hours
        Expected attributes: latitude_deg, altitude_km.

    Returns
    -------
    n_series : int
    n_frames : int
    observatory : dict
        {"lat": latitude_deg, "alt": altitude_km}
    """
    with h5py.File(conditions_h5, "r") as hf:
        required = (
            "pressure_hpa",
            "temperature_k",
            "humidity_pct",
            "zenith_angle_deg",
            "airmass",
            *[f"{mol}_ppmv" for mol in MOLECULES],
        )
        missing = [key for key in required if key not in hf]
        if missing:
            raise KeyError(f"Conditions HDF5 is missing datasets: {missing}")

        shape = hf["pressure_hpa"].shape
        if len(shape) != 2:
            raise ValueError(f"pressure_hpa must be (N_series, N_frames), got {shape}")
        n_series, n_frames = map(int, shape)

        for key in ("temperature_k", "humidity_pct", "zenith_angle_deg", "airmass"):
            if hf[key].shape != shape:
                raise ValueError(f"{key} shape {hf[key].shape} != {shape}")

        for mol in MOLECULES:
            if hf[f"{mol}_ppmv"].shape != (n_series,):
                raise ValueError(
                    f"{mol}_ppmv shape {hf[f'{mol}_ppmv'].shape} != ({n_series},)"
                )

        if "time_hours" in hf and hf["time_hours"].shape != (n_frames,):
            raise ValueError(
                f"time_hours shape {hf['time_hours'].shape} != ({n_frames},)"
            )

        observatory = {
            "lat": float(hf.attrs["latitude_deg"]),
            "alt": float(hf.attrs["altitude_km"]),
        }

    return n_series, n_frames, observatory


def load_condition_sample(hf, flat_index: int, n_frames: int):
    """Return MakeModel parameters and a numeric label row for one frame.

    A flattened index maps to a (series, frame) pair:
        series_index = flat_index // n_frames
        frame_index  = flat_index %  n_frames

    Frame-wise quantities vary per frame; molecular abundances are constant
    per series.

    Parameters
    ----------
    hf : h5py.File
        An open condition series file.
    flat_index : int
        Flattened (series, frame) index.
    n_frames : int
        Number of frames per series.

    Returns
    -------
    params : dict
        MakeModel keyword arguments (pressure, temperature, humidity, angle,
        and molecular abundances).
    label : numpy.ndarray
        Numeric provenance row (see LABEL_COLUMNS in the caller).
    """
    import numpy as np

    series_index = int(flat_index // n_frames)
    frame_index = int(flat_index % n_frames)

    params = {
        "pressure": float(hf["pressure_hpa"][series_index, frame_index]),
        "temperature": float(hf["temperature_k"][series_index, frame_index]),
        "humidity": float(hf["humidity_pct"][series_index, frame_index]),
        "angle": float(hf["zenith_angle_deg"][series_index, frame_index]),
    }
    for mol in MOLECULES:
        params[mol] = float(hf[f"{mol}_ppmv"][series_index])

    time_hours = (
        float(hf["time_hours"][frame_index])
        if "time_hours" in hf
        else float(frame_index)
    )
    airmass = float(hf["airmass"][series_index, frame_index])

    label = np.asarray(
        [
            float(flat_index),
            float(series_index),
            float(frame_index),
            time_hours,
            params["pressure"],
            params["temperature"],
            params["humidity"],
            params["angle"],
            airmass,
            *[params[mol] for mol in MOLECULES],
        ],
        dtype=np.float64,
    )

    return params, label
