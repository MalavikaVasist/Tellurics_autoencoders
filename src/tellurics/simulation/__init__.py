"""
Simulation infrastructure for generating training data.

This package produces the data used to train the tellurics NN:
- phoenix/  : download and convolve PHOENIX stellar spectra
- telluric/ : generate TelFit telluric transmissions
- combine/  : multiply stellar x telluric to build observations

It is separate from the `tellurics` ML package (src/tellurics) but imports
shared spectral utilities from it.
"""
