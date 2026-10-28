# SIESTA band tools

Calculate and plot **supercell** bands from a saved SIESTA TSHS Hamiltonian
using sisl. As an additional feature, plot SIESTA WFSX bands and fatbands.
This package does not perform band unfolding.

## Installation

Use Python 3.9 or later in an environment with compatible NumPy, SciPy,
Matplotlib and [sisl](https://github.com/zerothi/sisl). The existing workflow
has been exercised with sisl 0.14.3, NumPy 1.26.4, SciPy 1.14.0 and Matplotlib
3.9.2. Compatibility with other versions is not yet established.

```sh
python -m pip install .
```

To install directly from GitHub:

```sh
python -m pip install git+https://github.com/YHKlab-MSJeong/siesta-band-tools.git
```

## Calculate and plot bands from TSHS

`examples/kpath.json` defines vertices in **fractional supercell reciprocal
coordinates**, labels and the number of intervals in each segment. Supply a
path appropriate to the actual supercell; the example is only Gamma to X.

```sh
siesta-band calculate path/to/model.TSHS examples/kpath.json bands.npz
siesta-band plot bands.npz bands.png -er -2 2
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

For command-line help, use `siesta-band --help`, `siesta-band calculate --help`,
or `siesta-band plot --help`.

## Plot bands from WFSX

The plotter retains the previous `band.py` selectors and style. Point it at a
case directory containing `output/`, or directly at the output directory:

```sh
siesta-band wfsx plot ./case/output --no-fat -eref fermi -er -2 2
siesta-band wfsx plot ./case/output y 23.6 W -c orangered -fig 1.8 4.5 -eref left
siesta-band wfsx plot ./case/output atom 35 4d -eref fermi
siesta-band wfsx plot --help
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

## Python interface

`siesta_band_tools.calculation.calculate` evaluates a supercell k-path from
TSHS, and `diagonalize` operates on an existing sisl Hamiltonian. The separate
`siesta_band_tools.io` module contains WFSX, `.bands` and EIG readers. These
interfaces are also intended for local analysis code. No unfolding or
localization implementation is included here.

To calculate a band path from a TSHS file in your own script:

```python
from siesta_band_tools.calculation import calculate

result = calculate(
    "path/to/model.TSHS",
    vertices=[[0, 0, 0], [0.5, 0, 0]],
    intervals=[40],
)
energy = result["energy_eV"]  # absolute energy, shape (k-points, bands)
k_distance = result["k_distance_inv_ang"]
fermi_energy = float(result["fermi_eV"])

print("energy relative to TSHS Fermi level:", energy - fermi_energy)
```

The example vertices are fractional coordinates in the TSHS supercell's
reciprocal basis; choose a path for your system. Set
`legacy_siesta_units=True` only when the SIESTA source uses the legacy
constants described above. To make a PNG directly from SIESTA outputs, use
the `siesta-band wfsx plot` commands in the section above.

To draw the calculated energies on your own Matplotlib axes, continue from the
previous example:

```python
import matplotlib.pyplot as plt
from siesta_band_tools.plotting import draw_bands

fig, ax = plt.subplots()
draw_bands(ax, k_distance, energy, reference_energy=fermi_energy,
           energy_range=(-2, 2), color="black")
ax.set_ylim(-2, 2)
ax.set_ylabel("Energy - Fermi energy (eV)")
plt.show()
```

The `siesta_band_tools.data` module also exposes the WFSX band preparation
functions separately from rendering. `siesta_band_tools.plotting.PlotStyle`,
`plot_band`, and `plot_fatband` provide the standard PNG style without using
the command-line plotter's global settings.

As an additional feature, read WFSX metadata in your own script:

```python
from siesta_band_tools.io import read_wfsx_metadata

metadata = read_wfsx_metadata("path/to/model.bands.WFSX")
print("k-points:", metadata["nk"])
print("orbitals:", metadata["norb"])
print("species:", sorted(set(metadata["symbols"])))
```

## Command changes from 0.1

The previous entry points were replaced by the `siesta-band` command:

| Previous command | New command |
| --- | --- |
| `siesta-band-calculate` | `siesta-band calculate` |
| `siesta-band-plot-calculated` | `siesta-band plot` |
| `siesta-band-plot` | `siesta-band wfsx plot` |

Update existing scripts that use the previous command names. The Python
functions and NPZ format are unchanged.

Run the small synthetic tests with `python -m unittest discover -s tests -v`.
The readers and plotting code are adapted from the existing `band.py` and
associated utility routines. No license has yet been assigned to this code.
