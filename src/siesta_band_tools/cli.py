"""Command-line interface for TSHS bands and WFSX plots."""
import argparse
from io import BytesIO
from pathlib import Path
import sys
import numpy as np

from .calculation import calculate, load_path


def compute(arguments=None):
    parser = argparse.ArgumentParser(prog='siesta-band calculate', description='Calculate unpolarized bands from a SIESTA TSHS')
    parser.add_argument('tshs', type=Path)
    parser.add_argument('path', type=Path, help='JSON file: vertices, intervals, labels')
    parser.add_argument('output', type=Path, help='new NPZ output (never overwritten)')
    parser.add_argument('--legacy-siesta-units', action='store_true', help='use legacy SIESTA unit constants')
    args = parser.parse_args(arguments)
    vertices, intervals, labels = load_path(args.path)
    if args.output.suffix != '.npz':
        parser.error('output must have the .npz extension')
    if args.output.exists():
        parser.error('output exists; choose another output')
    result = calculate(args.tshs, vertices, intervals, legacy_siesta_units=args.legacy_siesta_units)
    result['tick_labels'] = np.asarray(labels)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **result)
    print(f'saved {args.output}')


def plot_computed(arguments=None):
    parser = argparse.ArgumentParser(prog='siesta-band plot', description='Plot a computed TSHS band NPZ')
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('-er', '--energy-range', nargs=2, type=float, default=(-3., 3.))
    parser.add_argument('-fig', '--figure-size', nargs=2, type=float, default=(3.5, 5.))
    parser.add_argument('-dpi', type=int, default=280)
    parser.add_argument('--reference', type=float, help='absolute energy reference in eV (default: TSHS Fermi)')
    args = parser.parse_args(arguments)
    if args.output.exists():
        parser.error('output exists; choose another output')
    import matplotlib
    matplotlib.use('Agg')
    from .io import normalize_k_label
    from .plotting import PlotStyle, plot_band
    with np.load(args.data, allow_pickle=False) as data:
        energy = data['energy_eV']
        distance = data['k_distance_inv_ang']
        indices = data['tick_indices']
        labels = [normalize_k_label(str(x)) for x in data['tick_labels']]
        reference = float(data['fermi_eV']) if args.reference is None else args.reference
        if energy.ndim != 2 or len(energy) != len(distance) or len(indices) != len(labels):
            raise ValueError('invalid computed band dimensions')
        style = PlotStyle(width=args.figure_size[0], height=args.figure_size[1],
                          energy_range=tuple(args.energy_range), dpi=args.dpi)
        count = plot_band(args.output, [(1, energy)], True, reference, 'reference',
                          distance, distance[indices].tolist(), labels, style=style)
    if not count:
        print('warning: no band points in the selected energy range')
    print(f'saved {args.output}')


def compute_pdos(arguments=None):
    parser = argparse.ArgumentParser(prog='siesta-band pdos calculate',
                                     description='Calculate orbital PDOS from a TSHS on a full k-grid')
    parser.add_argument('tshs', type=Path)
    parser.add_argument('output', type=Path, help='new NPZ output (never overwritten)')
    parser.add_argument('--kgrid', nargs=3, type=int, required=True, metavar=('N1', 'N2', 'N3'))
    parser.add_argument('--kshift', nargs=3, type=float, default=(0., 0., 0.),
                        metavar=('K1', 'K2', 'K3'), help='fractional reciprocal-coordinate displacement')
    parser.add_argument('-er', '--energy-range', nargs=2, type=float, required=True, metavar=('MIN', 'MAX'),
                        help='energy limits in eV relative to the TSHS Fermi energy')
    parser.add_argument('--points', type=int, default=801, help='number of energy grid points')
    parser.add_argument('--sigma', type=float, default=.1, help='Gaussian standard deviation in eV')
    parser.add_argument('--orb-indx', type=Path, help='matching SIESTA ORB_INDX for element/orbital labels')
    parser.add_argument('--legacy-siesta-units', action='store_true', help='use legacy SIESTA unit constants')
    args = parser.parse_args(arguments)
    if args.output.suffix.lower() != '.npz':
        parser.error('output must have the .npz extension')
    if args.output.exists():
        parser.error('output exists; choose another output')
    from .pdos import calculate_pdos
    result = calculate_pdos(args.tshs, kgrid=args.kgrid, kshift=args.kshift,
                            energy_range=args.energy_range, points=args.points, sigma=args.sigma,
                            orbital_index=args.orb_indx, legacy_siesta_units=args.legacy_siesta_units)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **result)
    print(f'saved {args.output}')


def plot_pdos(arguments=None):
    parser = argparse.ArgumentParser(prog='siesta-band pdos plot', description='Plot selected orbital PDOS from NPZ')
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path, help='new PNG output (never overwritten)')
    parser.add_argument('--atom', nargs='+', type=int, help='1-based SIESTA atom indices')
    parser.add_argument('--element', help='species label from ORB_INDX, e.g. W')
    parser.add_argument('--orbital', help='orbital selector, e.g. d, 5d, dz2')
    parser.add_argument('--axis', choices=['x', 'y', 'z'])
    parser.add_argument('--coordinate', type=float, help='position in Angstrom along --axis')
    parser.add_argument('--tolerance', type=float, default=1.5, help='position tolerance in Angstrom')
    parser.add_argument('--reference', type=float, help='absolute energy reference in eV (default: TSHS Fermi)')
    parser.add_argument('-er', '--energy-range', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('-fig', '--figure-size', nargs=2, type=float, default=(5., 3.5))
    parser.add_argument('-dpi', '--dpi', type=int, default=280)
    parser.add_argument('-c', '--color', default='k')
    args = parser.parse_args(arguments)
    if args.output.suffix.lower() != '.png':
        parser.error('output must have the .png extension')
    if args.output.exists():
        parser.error('output exists; choose another output')
    import matplotlib
    matplotlib.use('Agg')
    from .pdos import select_orbitals
    from .plotting import plot_pdos as render
    with np.load(args.data, allow_pickle=False) as data:
        energy = np.asarray(data['energy_eV'], dtype=float)
        pdos = np.asarray(data['pdos_per_spin'], dtype=float)
        indices = select_orbitals(data, atoms=args.atom, element=args.element, orbital=args.orbital,
                                  axis=args.axis, coordinate=args.coordinate, tolerance=args.tolerance)
        if pdos.ndim != 2 or pdos.shape != (len(data['orbital_atom_indices']), len(energy)):
            raise ValueError('invalid orbital PDOS dimensions')
        reference = float(data['fermi_eV']) if args.reference is None else args.reference
        image = BytesIO()
        render(image, energy, pdos[indices].sum(axis=0), reference_energy=reference,
               reference_label='Fermi' if args.reference is None else 'reference',
               energy_range=args.energy_range, figsize=args.figure_size, dpi=args.dpi, color=args.color)
    with args.output.open('xb') as handle:
        handle.write(image.getvalue())
    print(f'saved {args.output}; selected orbitals={len(indices)}')


def main(arguments=None):
    """Dispatch to TSHS calculation/plotting or the WFSX plotting tool."""
    if arguments is None:
        arguments = sys.argv[1:]
    arguments = list(arguments)
    parser = argparse.ArgumentParser(
        prog='siesta-band',
        description='Calculate and plot TSHS bands and PDOS; optionally plot SIESTA WFSX bands.',
        epilog='commands: calculate (TSHS bands), plot (band PNG), pdos calculate/plot, wfsx plot')
    parser.add_argument('command', choices=['calculate', 'plot', 'pdos', 'wfsx'])
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.parse_args(arguments)
        return 0
    command = arguments.pop(0)
    if command == 'calculate':
        return compute(arguments)
    if command == 'plot':
        return plot_computed(arguments)
    if command == 'pdos':
        pdos = argparse.ArgumentParser(prog='siesta-band pdos', description='Calculate and plot TSHS PDOS.')
        pdos.add_argument('command', choices=['calculate', 'plot'])
        if not arguments or arguments[0] in ('-h', '--help'):
            pdos.parse_args(arguments)
            return 0
        subcommand = arguments.pop(0)
        if subcommand == 'calculate':
            return compute_pdos(arguments)
        if subcommand == 'plot':
            return plot_pdos(arguments)
        pdos.error('expected command: calculate or plot')
    if command == 'wfsx':
        wfsx = argparse.ArgumentParser(prog='siesta-band wfsx', description='Plot SIESTA WFSX bands and fatbands.')
        wfsx.add_argument('command', choices=['plot'])
        if not arguments or arguments[0] in ('-h', '--help'):
            wfsx.parse_args(arguments)
            return 0
        if arguments.pop(0) != 'plot':
            wfsx.error('expected command: plot')
        from . import legacy_plot
        return legacy_plot.main(arguments, prog='siesta-band wfsx plot')
    parser.error(f'invalid command: {command}')


if __name__ == '__main__':
    sys.exit(main())
