"""Command-line interface for TSHS bands and WFSX plots."""
import argparse
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


def main(arguments=None):
    """Dispatch to TSHS calculation/plotting or the WFSX plotting tool."""
    if arguments is None:
        arguments = sys.argv[1:]
    arguments = list(arguments)
    parser = argparse.ArgumentParser(
        prog='siesta-band',
        description='Calculate and plot TSHS bands; optionally plot SIESTA WFSX bands.',
        epilog='commands: calculate (TSHS to NPZ), plot (NPZ to PNG), wfsx plot (WFSX to PNG)')
    parser.add_argument('command', choices=['calculate', 'plot', 'wfsx'])
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.parse_args(arguments)
        return 0
    command = arguments.pop(0)
    if command == 'calculate':
        return compute(arguments)
    if command == 'plot':
        return plot_computed(arguments)
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
