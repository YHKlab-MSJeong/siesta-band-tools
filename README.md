# SIESTA band tools

Calculate and plot **supercell** bands and projected DOS from a saved SIESTA
TSHS Hamiltonian using sisl. As an additional feature, plot SIESTA WFSX bands
and fatbands. This package does not perform band unfolding.

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

## Calculate and plot PDOS from TSHS

Specify the full supercell Brillouin-zone mesh and an energy range relative to
the TSHS Fermi energy. The numbers below are examples, not convergence
recommendations:

```sh
siesta-band pdos calculate model.TSHS pdos.npz --kgrid 7 1 5 -er -3 3 --points 1201 --sigma 0.05
siesta-band pdos plot pdos.npz total_pdos.png
siesta-band pdos plot pdos.npz atom_pdos.png --atom 1 2
```

The full Monkhorst-Pack grid is centered with no time-reversal reduction;
`--kshift` optionally supplies a displacement in fractional supercell
reciprocal coordinates. `--sigma` is the Gaussian standard deviation in eV.
Choose and converge the grid, energy limits and broadening for your system;
the high-symmetry band path is not a Brillouin-zone integral.

TSHS alone gives the Hamiltonian, overlap, atom positions and orbital ordering;
it does **not** reliably identify element and orbital names. For name-based
selection, supply the matching SIESTA ORB_INDX during calculation:

```sh
siesta-band pdos calculate model.TSHS pdos_labeled.npz --kgrid 7 1 5 -er -3 3 --orb-indx model.ORB_INDX
siesta-band pdos plot pdos_labeled.npz W_layer.png --element W --orbital 5d --axis y --coordinate 23.6 --tolerance 1.5
```

Atom indices are 1-based. All selection filters are intersected; axis and
coordinate must be supplied together and are in Angstrom. ORB_INDX atom/orbital
counts and order are checked against TSHS, but matching counts alone do not
prove that the files came from the same calculation. Without ORB_INDX, use
atom indices or coordinates instead of element/orbital names.

PDOS is the overlap-aware Mulliken orbital projection from sisl, in states/eV
**per spin and per supercell** for an unpolarized Hamiltonian; no factor of two is applied.
Individual nonorthogonal orbital contributions need not all be positive.
Spin-polarized and spin-orbit Hamiltonians are not supported by this command.
The NPZ stores orbital-resolved PDOS, absolute energies, TSHS Fermi energy,
atom mapping/positions, k-points/weights and calculation metadata. Plots use
TSHS Fermi energy by default; `--reference` sets a different absolute energy
in eV. To compare with SIESTA PDOS, match k-grid, smearing, energy reference,
spin normalization and projection conventions. The NPZ and PNG destinations
must be new files. See `siesta-band pdos calculate --help` and
`siesta-band pdos plot --help` for options.

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

Calculate and select orbital PDOS in Python:

```python
import numpy as np
from siesta_band_tools.pdos import calculate_pdos, select_orbitals

result = calculate_pdos("model.TSHS", kgrid=[7, 1, 5], energy_range=[-3, 3],
                        points=1201, sigma=0.05, orbital_index="model.ORB_INDX")
indices = select_orbitals(result, element="W", orbital="5d")
W_d_pdos = result["pdos_per_spin"][indices].sum(axis=0)
with open("pdos_python.npz", "xb") as output:
    np.savez_compressed(output, **result)
```

Use `siesta_band_tools.plotting.draw_pdos` to draw the selected PDOS on a
Matplotlib axes.

As an additional feature, read WFSX metadata in your own script:

```python
from siesta_band_tools.io import read_wfsx_metadata

metadata = read_wfsx_metadata("path/to/model.bands.WFSX")
print("k-points:", metadata["nk"])
print("orbitals:", metadata["norb"])
print("species:", sorted(set(metadata["symbols"])))
```

Run the small synthetic tests with `python -m unittest discover -s tests -v`.
The readers and plotting code are adapted from the existing `band.py` and
associated utility routines. No license has yet been assigned to this code.
