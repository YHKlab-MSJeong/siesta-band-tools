"""Command-line entry points for computed bands."""
import argparse
from pathlib import Path
import numpy as np

from .calculation import calculate, load_path


def compute():
    parser = argparse.ArgumentParser(description='Calculate unpolarized bands from a SIESTA TSHS')
    parser.add_argument('tshs', type=Path)
    parser.add_argument('path', type=Path, help='JSON file: vertices, intervals, labels')
    parser.add_argument('output', type=Path, help='new NPZ output (never overwritten)')
    parser.add_argument('--legacy-siesta-units', action='store_true', help='use legacy SIESTA unit constants')
    args = parser.parse_args()
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


def plot_computed():
    parser = argparse.ArgumentParser(description='Plot computed band NPZ')
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('-er', '--energy-range', nargs=2, type=float, default=(-3., 3.))
    parser.add_argument('-fig', '--figure-size', nargs=2, type=float, default=(3.5, 5.))
    parser.add_argument('-dpi', type=int, default=280)
    parser.add_argument('--reference', type=float, help='absolute energy reference in eV (default: TSHS Fermi)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose another output')
    from . import legacy_plot as plot
    with np.load(args.data, allow_pickle=False) as data:
        energy = data['energy_eV']
        distance = data['k_distance_inv_ang']
        indices = data['tick_indices']
        labels = [plot.normalize_k_label(str(x)) for x in data['tick_labels']]
        reference = float(data['fermi_eV']) if args.reference is None else args.reference
        if energy.ndim != 2 or len(energy) != len(distance) or len(indices) != len(labels):
            raise ValueError('invalid computed band dimensions')
        plot.WIDTH, plot.HEIGHT = args.figure_size
        plot.ENERGY_RANGE = tuple(args.energy_range)
        plot.DPI = args.dpi
        count = plot.plot_band(args.output, [(1, energy)], True, reference, 'reference',
                               distance, distance[indices].tolist(), labels)
    if not count:
        print('warning: no band points in the selected energy range')
    print(f'saved {args.output}')
