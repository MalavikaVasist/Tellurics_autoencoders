"""
Spectral utilities shared across Phoenix and TelFit pipelines.

Functions for:
- Building constant-R wavelength grids
- Convolving spectra to a target resolving power
- Resampling onto output grids
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


def build_constant_R_grid(wmin, wmax, R, samples_per_resel=3.0):
    """
    Build a wavelength grid with constant resolving-power sampling.

    One resolution element spans d(ln λ) = 1/R. The grid has
    ``samples_per_resel`` samples per element (default 3, Nyquist-like).

    Parameters
    ----------
    wmin, wmax : float
        Wavelength range (any consistent units).
    R : float
        Target resolving power.
    samples_per_resel : float
        Samples per resolution element.

    Returns
    -------
    grid : ndarray
        Wavelength grid, geometrically spaced (uniform in ln λ).
    """
    n_intervals = int(np.ceil(np.log(wmax / wmin) * R * samples_per_resel))
    return np.geomspace(wmin, wmax, n_intervals + 1)


def convolve_to_resolution(wave, flux, target_R):
    """
    Convolve a spectrum (already on a uniform log-λ grid) to a lower
    resolving power. Returns convolved flux on the same grid.

    Use this when the input is already on a geomspace / uniform-log grid
    (e.g., Phoenix after resampling, or TelFit on a uniform log grid).

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (must be uniform in log-space).
    flux : ndarray
        Flux or transmission on that grid.
    target_R : float
        Target resolving power.

    Returns
    -------
    flux_convolved : ndarray
        Convolved flux on the same wavelength grid.
    """
    log_wave = np.log(wave)
    dlog = log_wave[1] - log_wave[0]

    sigma_log = 1.0 / (target_R * FWHM_TO_SIGMA)
    sigma_pixels = sigma_log / dlog

    if sigma_pixels < 0.5:
        raise ValueError(
            f"Grid sampling too coarse for R={target_R}: "
            f"sigma = {sigma_pixels:.2f} pixels (need >= 0.5)."
        )

    return gaussian_filter1d(flux, sigma=sigma_pixels, mode="nearest", truncate=4.0)


def convolve_and_resample(
    wave_native,
    flux_native,
    target_R,
    target_wave=None,
    samples_per_resel=3.0,
    wave_padding=0.0,
):
    """
    Convolve a high-resolution spectrum to ``target_R`` and resample
    onto an output grid.

    Handles non-uniform input grids (e.g., raw PHOENIX, native LBLRTM)
    by first interpolating onto a uniform ln(λ) grid.

    Parameters
    ----------
    wave_native : ndarray
        Native wavelength grid (any units, must be sorted ascending).
    flux_native : ndarray
        Native flux/transmission on ``wave_native``.
    target_R : float
        Target resolving power.
    target_wave : ndarray, optional
        Output wavelength grid. If None, one is built with
        ``samples_per_resel`` over the range of ``wave_native``
        (minus any padding).
    samples_per_resel : float
        Samples per resolution element for the auto-built grid.
    wave_padding : float
        If > 0, the native spectrum should extend this far beyond the
        target grid edges (for clean convolution boundaries). The
        auto-built target grid will be shrunk by this amount on each side.

    Returns
    -------
    wave_out : ndarray
        Output wavelength grid.
    flux_out : ndarray
        Convolved and resampled flux.
    """
    wave_native = np.asarray(wave_native, dtype=np.float64)
    flux_native = np.asarray(flux_native, dtype=np.float64)

    # Clean: finite, positive, sorted, unique
    valid = np.isfinite(wave_native) & np.isfinite(flux_native) & (wave_native > 0)
    wave_native = wave_native[valid]
    flux_native = flux_native[valid]

    # order = np.argsort(wave_native)
    # wave_native = wave_native[order]
    # flux_native = flux_native[order]

    # wave_native, idx = np.unique(wave_native, return_index=True)
    # flux_native = flux_native[idx]

    # Uniform ln(λ) grid at native sampling density
    log_native = np.log(wave_native)
    dlog = float(np.median(np.diff(log_native)))
    log_uniform = np.arange(log_native[0], log_native[-1] + 0.5 * dlog, dlog)
    flux_uniform = np.interp(log_uniform, log_native, flux_native)

    # Gaussian convolution
    sigma_log = 1.0 / (target_R * FWHM_TO_SIGMA)
    sigma_pixels = sigma_log / dlog

    if sigma_pixels < 0.5:
        raise ValueError(
            f"Native sampling too coarse for R={target_R}: "
            f"sigma = {sigma_pixels:.2f} pixels (need >= 0.5)."
        )

    flux_convolved = gaussian_filter1d(
        flux_uniform, sigma=sigma_pixels, mode="nearest", truncate=4.0,
    )

    # Build target grid if not provided
    if target_wave is None:
        wmin = wave_native[0] + wave_padding
        wmax = wave_native[-1] - wave_padding
        target_wave = build_constant_R_grid(wmin, wmax, target_R, samples_per_resel)

    # Resample
    flux_out = np.interp(np.log(target_wave), log_uniform, flux_convolved)

    return target_wave, flux_out
