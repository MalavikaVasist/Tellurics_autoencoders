
# simulation/telluric/model.py

from pathlib import Path

import numpy as np

from tellurics.simulation.telluric.backend import TelFitBackend
from tellurics.simulation.telluric.conditions import ConditionSeries
from tellurics.simulation.utils.spectral import (
    build_constant_R_grid,
    convolve_and_resample,
)

WAVE_START_NM = 800.0
WAVE_END_NM = 950.0
TARGET_R = 100_000
SAMPLES_PER_RESEL = 3.0
PADDING_NM = 0.1


class TelluricSimulation:
    """Forward model for telluric transmission spectra.

    Parameters
    ----------
    wave_start, wave_end : float
        Output wavelength range in nm.
    target_R : float
        Target resolving power.
    samples_per_resel : float
        Output samples per resolution element.
    padding_nm : float
        Padding applied to the native TelFit wavelength range.
    observatory : dict
        Observatory metadata, including ``lat`` and ``alt``.
    workdir : Path or str or None
        Optional TelFit working directory.
    modeler : object or None
        Optional injected TelFit-compatible modeler, useful for testing.
    """

    def __init__(
        self,
        wave_start=WAVE_START_NM,
        wave_end=WAVE_END_NM,
        target_R=TARGET_R,
        samples_per_resel=SAMPLES_PER_RESEL,
        padding_nm=PADDING_NM,
        observatory=None,
        workdir=None,
        modeler=None,
    ):
        self.wave_start = float(wave_start)
        self.wave_end = float(wave_end)
        self.target_R = float(target_R)
        self.samples_per_resel = float(samples_per_resel)
        self.padding_nm = float(padding_nm)
        self.observatory = observatory
        self.workdir = Path(workdir) if workdir is not None else None

        self.wavelength = build_constant_R_grid(
            self.wave_start,
            self.wave_end,
            self.target_R,
            self.samples_per_resel,
        ).astype(np.float64)

        self.backend = TelFitBackend(modeler=modeler)

    def simulate(self, params, *, workdir=None):
        """Simulate one atmospheric state.

        Parameters
        ----------
        params : dict
            Keyword arguments accepted by TelFit's MakeModel.
            Must include the required atmospheric parameters.
        workdir : Path or str or None
            Optional per-call working directory override.

        Returns
        -------
        np.ndarray
            Transmission on ``self.wavelength``.
        """
        # Convert the output wavelength limits to native wavenumbers.
        lowfreq = 1e7 / (self.wave_end + self.padding_nm)
        highfreq = 1e7 / (self.wave_start - self.padding_nm)

        effective_workdir = (
            Path(workdir) if workdir is not None else self.workdir
        )

        model = self.backend.make_model(
            params,
            lowfreq=lowfreq,
            highfreq=highfreq,
            observatory=self.observatory,
            workdir=effective_workdir,
        )

        _, transmission = convolve_and_resample(
            np.asarray(model.x, dtype=np.float64),
            np.asarray(model.y, dtype=np.float64),
            target_R=self.target_R,
            target_wave=self.wavelength,
        )

        transmission = np.asarray(transmission, dtype=np.float64)

        if transmission.shape != self.wavelength.shape:
            raise ValueError(
                f"Expected transmission shape {self.wavelength.shape}, "
                f"got {transmission.shape}"
            )

        if not np.all(np.isfinite(transmission)):
            raise ValueError("Transmission contains NaN or infinite values.")

        return transmission

    def simulate_batch(self, parameter_samples, *, workdirs=None):
        """Simulate multiple states; returns (n_samples, n_wavelengths)."""
        parameter_samples = list(parameter_samples)

        if workdirs is None:
            workdirs = [None] * len(parameter_samples)
        else:
            workdirs = list(workdirs)

        if len(workdirs) != len(parameter_samples):
            raise ValueError(
                "workdirs and parameter_samples must have equal lengths."
            )

        if not parameter_samples:
            return np.empty(
                (0, len(self.wavelength)),
                dtype=np.float64,
            )

        spectra = [
            self.simulate(params, workdir=wd)
            for params, wd in zip(parameter_samples, workdirs)
        ]

        return np.stack(spectra, axis=0)

    def simulate_series(self, conditions, start=0, stop=None):
        """Simulate a flattened range of a ConditionSeries."""
        if not isinstance(conditions, ConditionSeries):
            conditions = ConditionSeries(conditions)

        stop = conditions.n_samples if stop is None else stop

        spectra = []
        labels = []

        for _, params, label in conditions.load_range(start, stop):
            spectra.append(self.simulate(params))
            labels.append(label)

        if not spectra:
            return (
                np.empty((0, len(self.wavelength)), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64),
            )

        return np.stack(spectra), np.stack(labels)
