# Simulation

Data-generation pipeline for training the tellurics NN. This is self-contained
and separate from the `tellurics` ML package (`src/tellurics/`). Shared spectral
utilities (convolution, constant-R grids) live in `simulation/utils/spectral.py`.

## Pipeline

```
1. PHOENIX stellar spectra      simulation/phoenix/download_pool.py
   download R~500k -> convolve to R=100k -> data/phoenix/convolved/

2. TelFit telluric transmissions
   a) independent snapshots (LHS)   simulation/telluric/generate_grid.py
      native LBLRTM -> convolve to R=100k -> data/telluric/telluric_templates.h5
   b) full-night time series         simulation/telluric/generate_grid_timeseries.py
      reads a condition series -> data/telluric_timeseries/telluric_timeseries.h5

3. Combine into observations     simulation/combine/make_observations.py
   stellar x telluric -> data/observations/observations.h5
```

The NN is trained on `observed` (input) to recover `telluric` (target).

## Resolution consistency

All three stages use the same constant-R grid (R=100,000, 3 samples per
resolution element, 0.8-0.95 um), built by `build_constant_R_grid` in
`simulation.utils.spectral`. Both stellar and telluric spectra are properly
convolved with a Gaussian instrumental profile (FWHM = 1/R in ln-lambda)
before being resampled onto this grid.

## Data (external)

Datasets live in a per-machine root. The experiment manifests declare it as
`root: "${TELLURICS_DATA}"` (expanded by `tellurics.config.load_config`); the
generators resolve the same variable via `tellurics.config.paths`, so they write
into the data bundle instead of into the repo:

```bash
export TELLURICS_DATA="$HOME/Documents/Tellurics_data"
```

## TelFit (external)

The TelFit distribution is not vendored in this repo. The telluric generators
(`simulation/telluric/*.py`) resolve it via `simulation/utils/telfit_home.py`:
`$TELFIT_HOME` if set, else `<repo>/../Tellurics_data/telfit`.

```bash
export TELFIT_HOME=/scratch/<user>/Tellurics_data/telfit   # contains src/ and data/
```

## Usage

```bash
# 1. Download and convolve the PHOENIX pool (20 stars)
python simulation/phoenix/download_pool.py

# 2. Generate telluric transmissions (needs TelFit + SLURM for full grid)
python simulation/telluric/generate_grid.py --array-index 0 --batch-size 200

# 3. Combine into observations
python simulation/combine/make_observations.py --normalize
```

## Python API

```python
from simulation.telluric import TelluricSimulation

sim = TelluricSimulation(observatory={"lat": 30.6, "alt": 2.1})

transmission = sim.simulate(params)
transmissions = sim.simulate_batch(parameter_samples)
transmissions, labels = sim.simulate_series(conditions_h5, start=0, stop=100)
```

## Directory layout

```
simulation/
├── utils/
│   ├── spectral.py            # convolution + constant-R grid helpers
│   ├── site_parameters.py     # load atmospheric condition series
│   └── running_hpc.py         # SLURM workspace management
├── site_parameters/
│   └── telfit_condition_series.h5   # per-night atmospheric conditions
├── phoenix/
│   ├── configs/
│   │   └── phoenix_pool.csv   # stellar pool definition
│   └── download_pool.py       # PHOENIX download + convolution
├── telluric/
│   ├── __init__.py
│   ├── model.py                    # TelluricSimulation public API
│   ├── backend.py                  # TelFit adapter and lifecycle
│   ├── conditions.py               # condition-series HDF5 reader
│   ├── dataset.py                  # batch generation and HDF5 storage
│   ├── generate_grid.py            # independent snapshots (LHS sampling)
│   └── generate_grid_timeseries.py # CLI + SLURM for full-night series
└── combine/
    └── make_observations.py   # stellar x telluric

data/                          # generated outputs (gitignored)
├── .Telfit
├── telfit                     # telfit package
├── phoenix/
│   ├── downloads/             # raw PHOENIX HiRes FITS
│   └── convolved/             # R=100k convolved spectra
├── telluric_timeseries/       # full-night series (generate_grid_timeseries.py)
│   ├── chunks/
│   └── telluric_templates.h5  # independent snapshots (generate_grid.py)
│   └── telluric_templates_nights.h5  
```
