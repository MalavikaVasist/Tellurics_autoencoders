from pathlib import Path

import h5py
import numpy as np

from tellurics.simulation.telluric.conditions import ConditionSeries
from tellurics.simulation.telluric.model import TelluricSimulation
from tellurics.simulation.utils.site_parameters import MOLECULES

LABEL_COLUMNS = (
    "flat_index",
    "series_index",
    "frame_index",
    "time_hours",
    "pressure",
    "temperature",
    "humidity",
    "angle",
    "airmass",
    *MOLECULES,
)


def chunk_path(outdir, array_index):
    return Path(outdir) / "chunks" / f"chunk_{array_index:06d}.h5"


def generate_batch(
    array_index,
    n_samples,
    batch_size,
    conditions_h5,
    outdir,
    workdir=None,
    seed=42,
):
    outdir = Path(outdir)
    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    path = chunk_path(outdir, array_index)
    if path.exists():
        print(f"Chunk {array_index} already exists, skipping.")
        return

    conditions = ConditionSeries(conditions_h5)

    simulator = TelluricSimulation(
        observatory=conditions.observatory,
        workdir=workdir,
    )

    start = array_index * batch_size
    stop = min(start + batch_size, n_samples)
    if start >= n_samples:
        print(f"Chunk {array_index}: no samples (start={start} >= n_samples={n_samples})")
        return

    transmissions = []
    labels = []
    failed_indices = []

    for idx, params, label in conditions.load_range(start, stop):
        try:
            transmission = simulator.simulate(params)
            transmissions.append(transmission.astype(np.float32))
            labels.append(label.astype(np.float32))
        except Exception as exc:
            print(f"  FAILED at sample {idx}: {type(exc).__name__}: {exc}")
            print(f"    params: {params}")
            failed_indices.append(int(idx))

    if not transmissions:
        print(f"Chunk {array_index}: no models generated!")
        return

    transmission = np.stack(transmissions)
    label_array = np.stack(labels)

    with h5py.File(path, "w") as hf:
        hf.create_dataset(
            "wavelength",
            data=simulator.wavelength.astype(np.float32),
        )
        hf.create_dataset(
            "transmission",
            data=transmission,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        hf.create_dataset("labels", data=label_array)
        hf.create_dataset(
            "failed_indices",
            data=np.asarray(failed_indices, dtype=np.int64),
        )

        hf["wavelength"].attrs["unit"] = "nm"
        hf["labels"].attrs["columns"] = LABEL_COLUMNS
        hf.attrs["conditions_h5"] = str(conditions_h5)
        hf.attrs["target_resolution"] = simulator.target_R
        hf.attrs["samples_per_resolution_element"] = simulator.samples_per_resel
        hf.attrs["n_frames_per_series"] = conditions.n_frames

    print(
        f"Chunk {array_index} done: {len(transmissions)} models succeeded -> "
        f"{path}, {len(failed_indices)} failed"
    )


def aggregate(outdir: Path):
    """Combine all chunk HDF5 files into telluric_timeseries.h5 (streamed)."""
    chunks_dir = outdir / "chunks"
    final_path = outdir / "telluric_timeseries.h5"
    chunk_files = sorted(chunks_dir.glob("chunk_*.h5"))

    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return

    print(f"Aggregating {len(chunk_files)} chunk files...")

    wavelength = None
    label_columns = None
    total_models = 0
    total_failed = 0
    metadata = {}

    for cf in chunk_files:
        with h5py.File(cf, "r") as hf:
            wave = hf["wavelength"][:]
            if wavelength is None:
                wavelength = wave
                label_columns = tuple(hf["labels"].attrs["columns"])
                metadata = dict(hf.attrs)
            elif wave.shape != wavelength.shape or not np.allclose(
                wave, wavelength, rtol=0, atol=1e-6
            ):
                raise ValueError(f"Wavelength grid mismatch in {cf}")

            total_models += int(hf["transmission"].shape[0])
            total_failed += int(hf["failed_indices"].shape[0])

    n_wave = int(wavelength.shape[0])
    n_labels = len(label_columns)

    with h5py.File(final_path, "w") as out:
        out.create_dataset("wavelength", data=wavelength)
        out["wavelength"].attrs["unit"] = "nm"

        ds_transmission = out.create_dataset(
            "transmission",
            shape=(total_models, n_wave),
            dtype=np.float32,
            chunks=(1, n_wave),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        ds_labels = out.create_dataset(
            "labels", shape=(total_models, n_labels), dtype=np.float32,
        )
        ds_labels.attrs["columns"] = label_columns
        ds_failed = out.create_dataset(
            "failed_indices", shape=(total_failed,), dtype=np.int64,
        )

        for key, value in metadata.items():
            out.attrs[key] = value

        model_pos = 0
        fail_pos = 0
        for cf in chunk_files:
            with h5py.File(cf, "r") as hf:
                n = hf["transmission"].shape[0]
                nf = hf["failed_indices"].shape[0]

                ds_transmission[model_pos:model_pos + n] = hf["transmission"][:]
                ds_labels[model_pos:model_pos + n] = hf["labels"][:]
                if nf:
                    ds_failed[fail_pos:fail_pos + nf] = hf["failed_indices"][:]

                model_pos += n
                fail_pos += nf

    print(f"Total models: {total_models}")
    print(f"Failed models: {total_failed}")
    print(f"Wavelength points: {n_wave}")
    print(f"Parameters: {label_columns}")
    print(f"\nSaved: {final_path}")
    print(f"  wavelength:   {wavelength.shape}")
    print(f"  transmission: ({total_models}, {n_wave})")
    print(f"  labels:       ({total_models}, {n_labels})")
