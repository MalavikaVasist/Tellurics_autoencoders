"""
HPC / SLURM runtime helpers for simulation jobs.

Provides a per-job TelFit workspace under SLURM_TMPDIR so concurrent array
jobs don't collide on shared TelFit run directories.
"""

import os
import shutil
from pathlib import Path


def create_slurm_workspace(template: Path | None = None) -> Path:
    """Create a per-job TelFit workspace under ``SLURM_TMPDIR``.

    Copies a TelFit run-directory template into a job-unique folder so that
    concurrent SLURM array tasks each have an isolated working directory.

    Parameters
    ----------
    template : Path, optional
        TelFit run directory to copy. Defaults to ``~/.TelFit/rundir1``.

    Returns
    -------
    workdir : Path
        The per-job workspace directory.

    Raises
    ------
    RuntimeError
        If ``SLURM_TMPDIR`` is not set (i.e. not running under SLURM).
    """
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir is None:
        raise RuntimeError(
            "SLURM_TMPDIR is not available. "
            "This is intended to run under SLURM."
        )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    workdir = Path(slurm_tmpdir) / f"telfit-{job_id}-{array_id}"
    workdir.mkdir(parents=True, exist_ok=True)

    if template is None:
        template = Path.home() / ".TelFit" / "rundir1"

    shutil.copytree(template, workdir, dirs_exist_ok=True, symlinks=True)

    return workdir
