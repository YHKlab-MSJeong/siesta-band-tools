# SIESTA band tools

Tools for reading SIESTA band outputs, plotting bands and fatbands, and
calculating bands from a saved SIESTA TSHS Hamiltonian. The TSHS path computes
**supercell** bands; it does not perform band unfolding.

## Installation

Use Python 3.9 or later in an environment with compatible NumPy, SciPy,
Matplotlib and [sisl](https://github.com/zerothi/sisl). The existing workflow
has been exercised with sisl 0.14.3, NumPy 1.26.4, SciPy 1.14.0 and Matplotlib
3.9.2. Compatibility with other versions is not yet established.

```sh
python -m pip install .
```

## Plot SIESTA band output

The plotter retains the previous `band.py` selectors and style. Point it at a
case directory containing `output/`, or directly at the output directory:

```sh
siesta-band-plot ./case/output --no-fat -eref fermi -er -2 2
siesta-band-plot ./case/output y 23.6 W -c orangered -fig 1.8 4.5 -eref left
siesta-band-plot ./case/output atom 35 4d -eref fermi
siesta-band-plot --help
```

This interface reads `*.bands.WFSX`, `stdout.txt` and `RUN.fdf`. When present,
`*.bands` supplies the k-axis and labels. An EIG file is required for `-eref
vbm`; STRUCT.fdf or PDOS position data are needed for orbital selections.
`left` and `right` require MS-DFT electrode levels in stdout. WFSX weights are
sums of squared coefficients from the file, not unfolded weights; they should
not be interpreted as overlap-corrected Löwdin weights. This reader retains
the legacy WFSX format assumptions and has not been validated for spinors.
The plotter writes PNGs beside the specified output and can overwrite an
existing PNG with the same name, as in the original script.

## Calculate bands from TSHS

`examples/kpath.json` defines vertices in **fractional supercell reciprocal
coordinates**, labels and the number of intervals in each segment. Supply a
path appropriate to the actual supercell; the example is only Gamma to X.

```sh
siesta-band-calculate path/to/model.TSHS examples/kpath.json bands.npz
siesta-band-plot-calculated bands.npz bands.png -er -2 2
```

These commands do not overwrite existing outputs. Calculation currently
supports unpolarized Hamiltonians only. The NPZ contains k points, reciprocal
path distance (1/Angstrom), absolute eigenvalues (eV), the TSHS Fermi offset,
cell, labels and conversion metadata. Calculated plots subtract the TSHS Fermi
offset by default; use `--reference ENERGY_IN_EV` for a different absolute
reference, such as an MS-DFT electrode potential.

By default the installed sisl conversion constants are used. For a SIESTA
binary known to use Bohr = 0.529177 Angstrom and Ry = 13.60580 eV, pass
`--legacy-siesta-units`. This option temporarily changes sisl's reader
constants inside the process while reading TSHS, then restores them; use it in
a dedicated process, not concurrently from multiple threads. Unit choice must
be verified against that calculation's SIESTA source and EIG output.

## Python interface

`siesta_band_tools.io` contains the WFSX, `.bands` and EIG readers;
`siesta_band_tools.calculation.calculate` evaluates a supercell k-path from
TSHS, and `diagonalize` operates on an existing sisl Hamiltonian. These
interfaces are also intended for local analysis code. No
unfolding or localization implementation is included here.

Run the small synthetic tests with `python -m unittest discover -s tests -v`.
The readers and plotting code are adapted from the existing `band.py` and
associated utility routines. No license has yet been assigned to this code.
