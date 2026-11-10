"""Stellar spectrum pool and the seeded, fixed night -> star assignment.

One whole night is multiplied by a *single* stellar spectrum ``S``:
``observed = transmission * S``. The choice of ``S`` is fixed per night and
seeded, so it is reproducible and auditable (see
:func:`save_stellar_assignment`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StellarPool:
    """All stellar spectra from ``data/phoenix/convolved/*.fits``.

    Stellar spectra are assumed to already be on the same wavelength grid as
    the transmission spectra. A single stellar spectrum is used per whole
    night.

    Args:
        directory: Directory containing the stellar FITS files.
        star_files: Optional list of FITS filenames to load. If None, all
            ``*.fits`` files in ``directory`` are loaded.
        scale: If True, normalize each stellar spectrum so its mean flux is 1
            (raises if that mean is zero or non-finite). If False, return the
            original flux values.
    """

    _cache: tuple[tuple[str, ...], np.ndarray] | None = None

    def __init__(
        self,
        directory: str | Path,
        star_files: list[str] | None = None,
        scale: bool = True,
    ) -> None:
        directory = Path(directory)

        if StellarPool._cache is not None:
            names, spectra = StellarPool._cache
        else:
            files = (
                [directory / s for s in star_files]
                if star_files is not None
                else sorted(directory.glob("*.fits"))
            )

            if not files:
                raise FileNotFoundError(
                    f"No *.fits stellar spectra in {directory}"
                )

            names, spectra = self._load(files, scale)

            StellarPool._cache = (names, spectra)

        self.directory = directory
        self.names = names
        self.spectra = spectra

    @staticmethod
    def _load(
        files: list[Path],
        scale: bool,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Load each FITS spectrum.

        The stellar spectra are assumed to already be on the same wavelength
        grid as the transmission spectra.
        """
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "astropy is required to read Phoenix FITS"
            ) from exc

        names: list[str] = []
        spectra: list[np.ndarray] = []

        for path in files:
            with fits.open(path) as hdul:
                data = hdul["SPECTRUM"].data
                flux = np.asarray(data["FLUX"], np.float32)

            if scale:
                mean = float(flux.mean())
                if not np.isfinite(mean) or mean == 0.0:
                    raise ValueError(
                        f"cannot normalize {path.name}: mean flux is {mean}. "
                        "Set data.stellar_scale=false to use the raw flux."
                    )
                flux = flux / mean

            spectra.append(flux)
            names.append(path.name)

        return tuple(names), np.stack(spectra)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        """Return ``(spectrum (N,) float32, name)`` for stellar index ``idx``."""
        return self.spectra[idx], self.names[idx]


def assign_stellar_per_night(
    n_nights: int,
    pool: StellarPool,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic, fixed-per-night star assignment.

    Returns an ``(n_nights,)`` int array of stellar-pool indices. The same
    star is used for every exposure of a night and stays the same across
    epochs (no augmentation), so the mapping is reproducible and auditable.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(pool), size=n_nights)


def save_stellar_assignment(
    assignment: np.ndarray,
    pool: StellarPool,
    splits: dict[str, list[int]] | None,
    out_csv: str | Path,
) -> None:
    """Write ``night_id, split, star_index, star_file`` so the used stars are
    auditable per night and per split."""
    import csv

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_of = {}
    if splits is not None:
        for split, ids in splits.items():
            for nid in ids:
                split_of[int(nid)] = split
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["night_id", "split", "star_index", "star_file"])
        for night in range(assignment.size):
            sidx = int(assignment[night])
            writer.writerow(
                [night, split_of.get(int(night), ""), sidx, pool.names[sidx]]
            )
