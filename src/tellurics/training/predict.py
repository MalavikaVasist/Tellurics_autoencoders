"""Predict the tellurics of real observations (no target) and save them.

Used after training (``tellurics.scripts.train``) and by
``train --predict CKPT`` for a trained checkpoint. For every ``data.real``
file it writes ``<run_dir>/predictions/<name>.h5`` with::

    telluric_pred  (n, N) one telluric spectrum per observation
    observed       (n, N) the input the model saw
    night_id       (n,)   night (series) position within the file
    frame_id       (n,)   frame within the night
    wavelength     (N,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tellurics.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["load_trained_module", "predict_loader", "predict_real_observations"]


def load_trained_module(path, map_location="cpu"):
    """Load a checkpoint written by ``tellurics.scripts.train``.

    The checkpoints store the run's pydantic configs as hyperparameters, which
    PyTorch's default ``weights_only`` unpickler refuses; they are our own
    run outputs, so they are loaded with ``weights_only=False``.
    """
    from tellurics.training.module import TelluricTrainingModule

    return TelluricTrainingModule.load_from_checkpoint(
        str(path), map_location=map_location, weights_only=False
    )


@torch.no_grad()
def predict_loader(module: torch.nn.Module, loader: DataLoader) -> dict[str, np.ndarray]:
    """Run ``module`` over ``loader``; return the stacked arrays."""
    module.eval()
    device = next(module.parameters()).device
    out: dict[str, list[np.ndarray]] = {"telluric_pred": [], "observed": [], "night_id": []}
    frames: list[np.ndarray] = []
    wavelength = None
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in batch.items()
        }
        pred = module._run_estimator(batch).telluric
        out["telluric_pred"].append(pred.float().cpu().numpy())
        out["observed"].append(batch["observed"].float().cpu().numpy())
        out["night_id"].append(np.asarray(batch["night_id"].cpu()))
        if "frame_id" in batch:
            frames.append(np.asarray(batch["frame_id"].cpu()))
        if wavelength is None:
            wavelength = batch["wavelength"][0].float().cpu().numpy()
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    if frames:
        arrays["frame_id"] = np.concatenate(frames)
    arrays["wavelength"] = wavelength
    return arrays


def predict_real_observations(module, datamodule, run_dir: Path) -> list[Path]:
    """Predict every ``data.real`` file; return the written HDF5 paths."""
    import h5py

    loaders = datamodule.real_dataloaders()
    if not loaders:
        return []
    out_dir = Path(run_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, loader in loaders.items():
        arrays = predict_loader(module, loader)
        path = out_dir / f"{name}.h5"
        with h5py.File(path, "w") as f:
            for key, value in arrays.items():
                f.create_dataset(key, data=value)
        logger.info(f"real observations '{name}': {len(arrays['night_id'])} predictions -> {path}")
        written.append(path)
    return written
