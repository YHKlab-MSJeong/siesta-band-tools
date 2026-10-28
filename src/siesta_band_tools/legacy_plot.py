#!/usr/bin/env python3
import argparse
import os
import re
import struct
import sys

import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

if 'MPLCONFIGDIR' not in os.environ and not os.access(os.path.expanduser('~/.config/matplotlib'), os.W_OK):
    os.environ['MPLCONFIGDIR'] = '/tmp'

import matplotlib
import numpy as np
from scipy.io import FortranFile

from .io import (normalize_k_label, read_bands_path_data, normalize_orbital_name,
                 read_wfsx_metadata, read_wfsx_series, read_eig_vbm,
                 WFSX_ORBITAL_RECORD, WFSX_BLOCK_HEADER)
from .util import (
    LMAP,
    find_run_fdf,
    find_single_file,
    get_color_map,
    get_output_file_path,
    get_selector_groups,
    parse_plot_arguments,
    read_fermi_level,
    read_stdout_electrode_levels,
    read_structure,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''
 Adapted from the existing band.py (2026-04-08).

 description : make element/orbital-resolved fatband plots from
               SIESTA WFSX outputs in each case directory.
               Set ENERGY_REFERENCE_MODE to choose fermi or vbm alignment.

 useage : python band.py [root dir] [element] [orbital] [element] [orbital] ...
          python band.py [root dir] [x/y/z coord [element] [orbital]] ...

 e.g.    python band.py distance_top_fit
         python band.py distance_top_fit Mo 4d
         python band.py distance_top_fit z 39.7 Mo 4d
         python band.py distance_top_fit Ag Mo 4d S p
'''


DPI = 280  # Saved PNG resolution.
WIDTH = 3.5  # Figure width in inches.
HEIGHT = 5  # Figure height in inches.
ENERGY_RANGE = (-3.0, 3.0)  # Energy window after reference-energy shift.
ENERGY_REFERENCE_MODE = 'left'  # Energy zero: 'fermi' from stdout.txt or 'vbm' from *.EIG HOMO.
ENERGY_TICK_COUNT = 5#None  # Number of y-axis energy ticks; None keeps matplotlib auto ticks.
DEFAULT_MARKER_SCALE = 26.0
MARKER_SCALE = DEFAULT_MARKER_SCALE  # Fatband marker-size / plain-band linewidth scale.
BASE_BAND_LINEWIDTH = 0.7  # Plain-band linewidth when MARKER_SCALE is at its default.
ALPHA = 0.70  # Fatband marker opacity.
LINEWIDTH = 0.1  # Fermi-level line width.
LABEL_SIZE = 16  # Axis-label font size.
TICK_SIZE = 16  # Tick-label font size.
HIDE_TICK_AND_AXIS_LABELS = True#False  # Hide tick labels and axis labels together when True.
SHOW_LEGEND = False  # Show legend when True.
AXIS_LINEWIDTH = 2.2  # Plot border/spine width.
TICK_WIDTH = AXIS_LINEWIDTH  # Tick mark width.
TICK_LENGTH = 12  # Tick mark length.
GROUP_COLORS = ['k', 'g', 'b', 'k']  # Fallback per-group colors; repeats when shorter than group count. Use None for default palette.
SPIN_GROUP_COLORS = {
    1: ['b', 'g', 'b', 'k'],
    2: ['r', 'orange', 'purple', 'r'],
}  # Optional per-spin group colors. Missing spin indices fall back to GROUP_COLORS.
BOHR_TO_ANG = 0.529177210903
COMBINE_SELECTED_GROUPS = True#False  # Merge explicit input selectors into one group when True.
COMBINED_GROUP_LABEL = 'selected'  # Legend label for the merged selector group.
WEIGHT_TOL = 1.0e-4  # Minimum visible orbital weight.
WEIGHT_NORMALIZATION = 'none'  # Normalize fatband weights per selected atom or cell atom.
WFSX_ORBITAL_RECORD = struct.Struct('<i20sii20s')
WFSX_BLOCK_HEADER = struct.Struct('<idddd')


def format_default(value):
    if value is None:
        return 'auto'
    if isinstance(value, (list, tuple)):
        return ' '.join(str(item) for item in value)
    return str(value)


def parse_color_list(value):
    token = value.strip()
    if token.lower() in ['none', 'auto']:
        return None
    colors = [item.strip() for item in token.split(',') if item.strip()]
    if not colors:
        raise argparse.ArgumentTypeError('color list should not be empty')
    return colors


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def build_argument_parser():
    description = (
        'Make element/orbital-resolved fatband or plain band plots from SIESTA *.bands.WFSX outputs. '
        'Without -s, all spin channels are plotted together in one PNG.'
    )
    parser = argparse.ArgumentParser(
        prog='band.py',
        description=description,
        formatter_class=HelpFormatter,
        epilog=(
            'examples:\n'
            '  python band.py distance_top_fit\n'
            '  python band.py --no-fat -ms 40 distance_top_fit\n'
            '  python band.py -s distance_top_fit Mo 4d\n'
            '  python band.py -s1 distance_top_fit Mo 4d\n'
            '  python band.py distance_top_fit atom 35 4d\n'
            '  python band.py distance_top_fit z 39.7 Mo 4d\n'
            '  python band.py distance_top_fit Ag Mo 4d S p\n\n'
            'plain band:\n'
            '  --no-fat draws all eigenvalue bands without orbital weights.\n'
            '  -ms/--scale controls fatband marker size and plain-band line width.\n'
            '  Element/orbital/atom/position selectors are only valid for fatband plots.\n\n'
            'atom selector:\n'
            '  Use atom index [orbital]. Atom indices are 1-based SIESTA atom numbers.\n'
            '  python band.py distance_top_fit atom 35 4d\n\n'
            'position selector:\n'
            '  Coordinates are in Angstrom. Use axis coord [element] [orbital].\n'
            '  python band.py distance_top_fit y 18.6 W    # W atoms near y=18.6 Ang\n'
            '  W y 18.6 means two selectors: all W, then atoms near y=18.6.\n'
            '  Selector tokens are appended to the PNG name, e.g. fatband_y_18.6_W.png.'
        ),
    )
    parser.add_argument('-s', '--spin', nargs='?', const='split', default='all', metavar='N',
                        help='split spin channels when no value is given; N selects one spin channel')
    parser.add_argument('-er', '--energy-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                        default=ENERGY_RANGE, help='energy window after reference-energy shift')
    parser.add_argument('-eref', '--energy-reference', choices=['fermi', 'vbm', 'left', 'right'],
                        default=ENERGY_REFERENCE_MODE,
                        help='energy-zero reference; left/right use MSDFT electrode levels from stdout.txt')
    parser.add_argument('-et', '--energy-ticks', type=int, default=ENERGY_TICK_COUNT,
                        help='number of y-axis energy ticks')
    parser.add_argument('-eta', '--energy-ticks-auto', action='store_true',
                        help='use matplotlib automatic y-axis ticks')
    parser.add_argument('-fig', '--figure-size', nargs=2, type=float, metavar=('W', 'H'),
                        default=(WIDTH, HEIGHT), help='figure size in inches')
    parser.add_argument('-dpi', '--dpi', type=int, default=DPI, help='saved PNG resolution')
    parser.add_argument('--no-fat', action='store_true',
                        help='draw plain band lines without orbital-weight markers')
    parser.add_argument('-ms', '--scale', '--marker-scale', dest='marker_scale',
                        type=float, default=MARKER_SCALE,
                        help='fatband marker-size and plain-band line-width scale')
    parser.add_argument('-alpha', '--alpha', type=float, default=ALPHA,
                        help='fatband marker opacity')
    parser.add_argument('-lw', '--line-width', type=float, default=LINEWIDTH,
                        help='Fermi/reference horizontal line width')
    parser.add_argument('-fs', '--font-size', type=float, default=LABEL_SIZE,
                        help='axis-label font size')
    parser.add_argument('-ts', '--tick-size', type=float, default=TICK_SIZE,
                        help='tick-label font size')
    parser.add_argument('-tl', '--tick-length', type=float, default=TICK_LENGTH,
                        help='tick mark length')
    parser.add_argument('-aw', '--axis-width', type=float, default=AXIS_LINEWIDTH,
                        help='plot border/spine width')
    parser.add_argument('-tshow', '--show-tick-labels', action='store_true',
                        help='show tick labels and axis labels')
    parser.add_argument('-leg', '--legend', action='store_true', help='show legend')
    parser.add_argument('-c', '--colors', type=parse_color_list, default=GROUP_COLORS,
                        help='comma-separated group colors; use none for default palette')
    parser.add_argument('-c1', '--spin1-colors', type=parse_color_list,
                        default=SPIN_GROUP_COLORS.get(1),
                        help='comma-separated colors for spin 1 split plot')
    parser.add_argument('-c2', '--spin2-colors', type=parse_color_list,
                        default=SPIN_GROUP_COLORS.get(2),
                        help='comma-separated colors for spin 2 split plot')
    parser.add_argument('-sep', '--separate-groups', action='store_true',
                        help='do not merge explicit input selectors into one group')
    parser.add_argument('-label', '--combined-label', default=COMBINED_GROUP_LABEL,
                        help='legend label for merged selector group')
    parser.add_argument('-wtol', '--weight-tol', type=float, default=WEIGHT_TOL,
                        help='minimum visible orbital weight')
    parser.add_argument('--weight-normalization', choices=['none', 'atom', 'cell'],
                        default=WEIGHT_NORMALIZATION,
                        help='normalize weights per selected atom (atom) or all cell atoms (cell); none keeps raw weights')
    return parser


def parse_cli_arguments(arguments):
    parser = build_argument_parser()
    spin_cleaned, spin_option = extract_spin_option(arguments, arguments[0])
    options, plot_args = parser.parse_known_args(spin_cleaned[1:])
    for token in plot_args:
        if token.startswith('-') and not re.fullmatch(r'-?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?', token):
            parser.error(f'unrecognized option: {token}')
    cleaned = [arguments[0]] + plot_args
    return cleaned, spin_option, options


def apply_cli_options(options):
    global DPI, WIDTH, HEIGHT, ENERGY_RANGE, ENERGY_REFERENCE_MODE, ENERGY_TICK_COUNT
    global MARKER_SCALE, ALPHA, LINEWIDTH, LABEL_SIZE, TICK_SIZE
    global HIDE_TICK_AND_AXIS_LABELS, SHOW_LEGEND, AXIS_LINEWIDTH, TICK_WIDTH, TICK_LENGTH
    global GROUP_COLORS, SPIN_GROUP_COLORS, COMBINE_SELECTED_GROUPS, COMBINED_GROUP_LABEL
    global WEIGHT_TOL, WEIGHT_NORMALIZATION

    DPI = options.dpi
    WIDTH, HEIGHT = options.figure_size
    ENERGY_RANGE = tuple(options.energy_range)
    ENERGY_REFERENCE_MODE = options.energy_reference
    ENERGY_TICK_COUNT = None if options.energy_ticks_auto else options.energy_ticks
    MARKER_SCALE = options.marker_scale
    ALPHA = options.alpha
    LINEWIDTH = options.line_width
    LABEL_SIZE = options.font_size
    TICK_SIZE = options.tick_size
    AXIS_LINEWIDTH = options.axis_width
    TICK_WIDTH = AXIS_LINEWIDTH
    TICK_LENGTH = options.tick_length
    HIDE_TICK_AND_AXIS_LABELS = not options.show_tick_labels
    SHOW_LEGEND = options.legend
    GROUP_COLORS = options.colors
    SPIN_GROUP_COLORS = {
        1: options.spin1_colors,
        2: options.spin2_colors,
    }
    COMBINE_SELECTED_GROUPS = not options.separate_groups
    COMBINED_GROUP_LABEL = options.combined_label
    WEIGHT_TOL = options.weight_tol
    WEIGHT_NORMALIZATION = options.weight_normalization


def parse_spin_value(value, script_name):
    value = value.lower()
    if value == 'all':
        return 'all'
    if value == 'split':
        return 'split'
    try:
        spin_index = int(value)
    except ValueError:
        print(f'error : invalid spin option [{value}]')
        raise SystemExit(1)
    if spin_index < 1:
        print(f'error : spin index should be >= 1')
        raise SystemExit(1)
    return spin_index


def extract_spin_option(arguments, script_name):
    cleaned = [arguments[0]]
    spin_option = 'all'
    seen = False
    index = 1

    while index < len(arguments):
        token = arguments[index]
        value = None

        if token in ['-s', '--spin']:
            if index + 1 < len(arguments) and arguments[index + 1].lower() in ['all', 'split']:
                value = arguments[index + 1]
                index += 2
            elif index + 1 < len(arguments) and re.fullmatch(r'[0-9]+', arguments[index + 1]):
                value = arguments[index + 1]
                index += 2
            else:
                value = 'split'
                index += 1
        elif token.startswith('--spin='):
            value = token.split('=', 1)[1]
            index += 1
        elif token.startswith('-s') and len(token) > 2:
            value = token[2:]
            index += 1
        else:
            cleaned.append(token)
            index += 1
            continue

        if seen:
            print(f'error : duplicate spin option')
            raise SystemExit(1)
        spin_option = parse_spin_value(value, script_name)
        seen = True

    return cleaned, spin_option


def get_spin_indices(nspin, spin_option):
    if spin_option in ['all', 'split']:
        return list(range(1, nspin + 1))
    if spin_option > nspin:
        raise ValueError(f'requested spin {spin_option}, but nspin is {nspin}')
    return [spin_option]



def read_bandlines(run_fdf_path):
    lines = run_fdf_path.read_text(encoding='utf-8', errors='replace').splitlines()
    in_block = False
    points = []
    current_index = None

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith('%block bandlines'):
            in_block = True
            continue
        if in_block and lower.startswith('%endblock bandlines'):
            break
        if not in_block or not stripped or stripped.startswith('#'):
            continue

        fields = stripped.split()
        if len(fields) < 5:
            continue

        try:
            count = int(fields[0])
        except ValueError:
            continue

        if current_index is None:
            current_index = count - 1
        else:
            current_index += count

        points.append((current_index, normalize_k_label(fields[-1])))

    return points




def read_pdos_atom_positions(pdos_path):
    atom_positions = {}
    root = ET.parse(pdos_path).getroot()

    for orbital in root.findall('orbital'):
        atom_index = int(orbital.attrib['atom_index'])
        if atom_index in atom_positions:
            continue
        atom_positions[atom_index] = np.array(
            [float(value) for value in orbital.attrib['position'].split()],
            dtype=float,
        ) * BOHR_TO_ANG

    return atom_positions


def read_struct_atom_positions(struct_path):
    atom = read_structure(str(struct_path))
    positions = np.array(atom.xyz, dtype=float)
    return {index: positions[index - 1] for index in range(1, len(positions) + 1)}


def find_struct_fdf(case_dir, output_dir=None):
    case_dir = Path(case_dir)
    candidates = []
    if output_dir is not None:
        candidates.append(Path(output_dir) / 'STRUCT.fdf')
    candidates.extend([case_dir / 'output' / 'STRUCT.fdf', case_dir / 'input' / 'STRUCT.fdf'])
    for path in candidates:
        if path.is_file():
            return path
    return None






def build_orbital_info(metadata, atom_positions):
    info = []

    for index in range(metadata['norb']):
        atom_index = int(metadata['atom_indices'][index])
        if atom_index not in atom_positions:
            raise ValueError(f'missing atom position for atom [{atom_index}]')

        orbital_name = metadata['orbital_names'][index]
        if not orbital_name:
            raise ValueError(f'missing orbital name for orbital [{index + 1}]')

        info.append(
            {
                'symbol': metadata['symbols'][index],
                'n': int(metadata['nvalues'][index]),
                'l': orbital_name[0],
                'name': orbital_name,
                'atom_index': atom_index,
                'position': np.array(atom_positions[atom_index], dtype=float),
            }
        )

    return info


def get_groups(metadata, selectors, atom_positions):
    orbital_info = build_orbital_info(metadata, atom_positions)
    groups = get_selector_groups(orbital_info, selectors, 'symbol')
    if COMBINE_SELECTED_GROUPS and selectors:
        indices = []
        for group_indices in groups.values():
            indices.extend(group_indices)
        groups = OrderedDict(
            [(COMBINED_GROUP_LABEL, np.array(sorted(set(indices)), dtype=int))]
        )
    return groups


def get_weight_normalization_factors(metadata, groups, atom_positions):
    if WEIGHT_NORMALIZATION == 'none':
        return {label: 1 for label in groups}

    if WEIGHT_NORMALIZATION == 'cell':
        cell_atom_count = len(atom_positions)
        if cell_atom_count < 1:
            raise ValueError('cannot normalize weights per cell atom without STRUCT.fdf atom positions')
        return {label: cell_atom_count for label in groups}

    atom_indices = np.asarray(metadata['atom_indices'], dtype=int)
    factors = {}
    for label, orbital_indices in groups.items():
        selected_atoms = np.unique(atom_indices[orbital_indices])
        if len(selected_atoms) < 1:
            raise ValueError(f'cannot normalize empty orbital group [{label}] per atom')
        factors[label] = len(selected_atoms)
    return factors


def build_kdist(kpoints):
    if len(kpoints) == 0:
        return np.array([], dtype=float)
    if len(kpoints) == 1:
        return np.array([0.0], dtype=float)
    steps = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))






def resolve_band_energy_reference(output_dir, run_fdf_path, fermi):
    mode = ENERGY_REFERENCE_MODE.lower()
    if mode == 'fermi':
        return float(fermi), 'Ef'
    if mode == 'vbm':
        eig_path = get_output_file_path(str(output_dir), str(run_fdf_path), '.EIG')
        if eig_path is None:
            raise ValueError('ENERGY_REFERENCE_MODE is vbm but no *.EIG file was found')
        return read_eig_vbm(eig_path), 'VBM'
    if mode in ['left', 'right']:
        left_level, right_level = read_stdout_electrode_levels(Path(output_dir) / 'stdout.txt')
        if mode == 'left':
            return left_level, 'EL'
        return right_level, 'ER'
    raise ValueError("ENERGY_REFERENCE_MODE should be 'fermi', 'vbm', 'left', or 'right'")


def get_group_color_map(labels, spin_index=None):
    labels = list(labels)
    colors = GROUP_COLORS
    if spin_index is not None:
        colors = SPIN_GROUP_COLORS.get(spin_index, GROUP_COLORS)

    if colors is None:
        return get_color_map(labels)
    if len(colors) == 0:
        raise ValueError('group color lists should not be empty; use None for the default palette')
    return {label: colors[index % len(colors)] for index, label in enumerate(labels)}


def get_spin_output_path(output_png, spin_option, nspin):
    output_png = Path(output_png)
    if nspin == 1 or spin_option == 'all':
        return output_png
    return output_png.with_name(f'{output_png.stem}_{spin_option}{output_png.suffix}')


def read_fatband_data(case_dir, output_dir, run_fdf_path, wfsx_path, selectors):
    metadata = read_wfsx_metadata(wfsx_path)
    atom_positions = {}

    struct_path = find_struct_fdf(case_dir, output_dir)
    if WEIGHT_NORMALIZATION == 'cell' and struct_path is None:
        raise ValueError('cell weight normalization requires STRUCT.fdf')
    if struct_path is not None:
        atom_positions.update(read_struct_atom_positions(struct_path))

    pdos_path = get_output_file_path(str(output_dir), str(run_fdf_path), '.PDOS.xml')
    if pdos_path is not None and not atom_positions:
        atom_positions.update(read_pdos_atom_positions(Path(pdos_path)))

    groups = get_groups(metadata, selectors, atom_positions)
    normalization_factors = get_weight_normalization_factors(metadata, groups, atom_positions)
    spin_series = read_wfsx_series(wfsx_path, groups, metadata['nspin'])
    spin_plots = []

    for spin_index, series in spin_series.items():
        eig_blocks = series['eig_blocks']
        if len(eig_blocks) == 0:
            continue

        kpoints = np.array(series['kpoints'], dtype=float)
        kdist = build_kdist(kpoints)
        nbands = len(eig_blocks[0])
        data = OrderedDict()
        for label in groups:
            weight = np.concatenate(series['weight_blocks'][label])
            weight = weight / normalization_factors[label]
            data[label] = {
                'x': np.repeat(kdist, nbands),
                'y': np.concatenate(eig_blocks),
                'w': weight,
            }
        spin_plots.append((spin_index, data, kdist))

    if len(spin_plots) == 0:
        raise ValueError(f'no WFSX eigenstate blocks in [{wfsx_path}]')

    return metadata['nspin'], spin_plots


def read_plain_band_data(wfsx_path):
    metadata = read_wfsx_metadata(wfsx_path)
    groups = OrderedDict([('all', np.arange(metadata['norb'], dtype=int))])
    spin_series = read_wfsx_series(wfsx_path, groups, metadata['nspin'])
    spin_plots = []

    for spin_index, series in spin_series.items():
        eig_blocks = series['eig_blocks']
        if len(eig_blocks) == 0:
            continue

        kpoints = np.array(series['kpoints'], dtype=float)
        kdist = build_kdist(kpoints)
        spin_plots.append((spin_index, np.vstack(eig_blocks), kdist))

    if len(spin_plots) == 0:
        raise ValueError(f'no WFSX eigenstate blocks in [{wfsx_path}]')

    return metadata['nspin'], spin_plots


def style_axes(ax, kdist, tick_positions, tick_labels, energy_reference_label):
    ax.tick_params(
        which='both',
        direction='in',
        top=False,
        right=False,
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
    )
    if HIDE_TICK_AND_AXIS_LABELS:
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        ax.set_xlabel('k-path', fontsize=LABEL_SIZE)
        ax.set_ylabel(f'Energy - {energy_reference_label} (eV)', fontsize=LABEL_SIZE)
    ax.set_ylim(*ENERGY_RANGE)
    if ENERGY_TICK_COUNT is not None:
        if ENERGY_TICK_COUNT < 2:
            raise ValueError('ENERGY_TICK_COUNT should be at least 2 or None')
        ax.set_yticks(np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], ENERGY_TICK_COUNT))

    if len(kdist) > 0:
        ax.set_xlim(float(kdist[0]), float(kdist[-1]))

    if tick_positions:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        for xpos in tick_positions:
            ax.axvline(x=xpos, color='0.85', linewidth=0.7, zorder=0)
    else:
        ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    ax.axhline(y=0.0, color='0.55', linewidth=LINEWIDTH, zorder=0)


def plot_fatband(
    output_png,
    spin_items,
    merge_spins,
    reference_energy,
    energy_reference_label,
    kdist,
    tick_positions,
    tick_labels,
):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    ymin, ymax = ENERGY_RANGE
    plotted = 0
    plotted_labels = set()

    for spin_index, data in spin_items:
        color_spin_index = None if merge_spins else spin_index
        color_map = get_group_color_map(data.keys(), color_spin_index)
        for label, series in data.items():
            x = series['x']
            y = series['y'] - reference_energy
            w = series['w']
            mask = (y >= ymin) & (y <= ymax) & (w > WEIGHT_TOL)
            if not np.any(mask):
                continue

            # Keep energy on the vertical axis and k-path on the horizontal axis.
            size = np.maximum(w[mask], WEIGHT_TOL) * MARKER_SCALE
            plot_label = label
            if merge_spins and label in plotted_labels:
                plot_label = '_nolegend_'
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                color=color_map[label],
                alpha=ALPHA,
                linewidths=0.0,
                label=plot_label,
            )
            plotted_labels.add(label)
            plotted += 1

    style_axes(ax, kdist, tick_positions, tick_labels, energy_reference_label)
    if SHOW_LEGEND and plotted:
        ax.legend(frameon=False, fontsize=TICK_SIZE, loc='upper right')
    fig.tight_layout()
    fig.savefig(output_png, dpi=DPI, transparent=True)
    plt.close(fig)

    return plotted


def get_band_color(spin_index, merge_spins):
    if not merge_spins:
        colors = SPIN_GROUP_COLORS.get(spin_index, GROUP_COLORS)
        if colors:
            return colors[0]
    if GROUP_COLORS:
        return GROUP_COLORS[0]
    return 'k'


def get_band_linewidth():
    return BASE_BAND_LINEWIDTH * MARKER_SCALE / DEFAULT_MARKER_SCALE


def plot_band(
    output_png,
    spin_items,
    merge_spins,
    reference_energy,
    energy_reference_label,
    kdist,
    tick_positions,
    tick_labels,
):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    ymin, ymax = ENERGY_RANGE
    plotted = 0

    for spin_index, eig_matrix in spin_items:
        color = get_band_color(spin_index, merge_spins)
        for band_index in range(eig_matrix.shape[1]):
            y = eig_matrix[:, band_index] - reference_energy
            if not np.any((y >= ymin) & (y <= ymax)):
                continue
            ax.plot(kdist, y, color=color, linewidth=get_band_linewidth())
            plotted += 1

    style_axes(ax, kdist, tick_positions, tick_labels, energy_reference_label)
    fig.tight_layout()
    fig.savefig(output_png, dpi=DPI, transparent=True)
    plt.close(fig)

    return plotted


def is_direct_output_dir(path):
    path = Path(path)
    if not path.is_dir():
        return False
    if not (path / 'stdout.txt').is_file():
        return False
    return (path / 'RUN.fdf').is_file() or find_run_fdf(path.parent) is not None


def find_run_fdf_for_output(case_dir, output_dir):
    output_run_path = Path(output_dir) / 'RUN.fdf'
    if output_run_path.is_file():
        return output_run_path
    return find_run_fdf(case_dir)


def sanitize_filename_token(token):
    token = str(token).strip()
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '_', token)
    token = token.strip('_')
    return token or 'none'


def get_selector_filename_tokens(selectors):
    tokens = []
    for selector in selectors:
        if selector['kind'] == 'plain':
            tokens.append(selector['symbol'])
            if selector['orbital'] is not None:
                tokens.append(selector['orbital'])
            continue

        if selector['kind'] == 'atom':
            tokens.append('atom')
            tokens.append(selector['atom_index'])
            if selector['orbital'] is not None:
                tokens.append(selector['orbital'])
            continue

        tokens.append(selector['axis'])
        tokens.append(format(selector['coord'], '.6g'))
        if selector['symbol'] is not None:
            tokens.append(selector['symbol'])
        if selector['orbital'] is not None:
            tokens.append(selector['orbital'])
    return [sanitize_filename_token(token) for token in tokens]


def append_selector_suffix(output_png, selectors):
    if not selectors:
        return output_png
    output_png = Path(output_png)
    suffix = '_'.join(get_selector_filename_tokens(selectors))
    return output_png.with_name(f'{output_png.stem}_{suffix}{output_png.suffix}')


def parse_band_arguments(arguments):
    arguments, spin_option, options = parse_cli_arguments(arguments)
    apply_cli_options(options)
    if options.no_fat and options.weight_normalization != 'none':
        print('error : --weight-normalization is only available for fatband plots')
        raise SystemExit(1)
    if len(arguments) < 2:
        return parse_plot_arguments(arguments, 'band.py')

    prefix = 'band' if options.no_fat else 'fatband'
    input_path = Path(arguments[1]).resolve()
    if is_direct_output_dir(input_path):
        parse_args = [arguments[0], str(input_path.parent)] + arguments[2:]
        _, _, selectors, _ = parse_plot_arguments(parse_args, 'band.py')
        if options.no_fat and selectors:
            print('error : --no-fat does not accept element/orbital/position selectors')
            raise SystemExit(1)
        output_png = input_path / f'{prefix}.png'
        if not options.no_fat:
            output_png = append_selector_suffix(output_png, selectors)
        return [(input_path.parent, input_path, output_png)], selectors, spin_option, options.no_fat

    target_dir, case_dirs, selectors, is_single_case = parse_plot_arguments(arguments, 'band.py')
    if options.no_fat and selectors:
        print('error : --no-fat does not accept element/orbital/position selectors')
        raise SystemExit(1)

    jobs = []
    for case_dir in case_dirs:
        output_dir = case_dir / 'output'
        if is_single_case:
            output_png = case_dir / f'{prefix}.png'
        else:
            output_png = target_dir / f'{prefix}_{case_dir.name}.png'
        if not options.no_fat:
            output_png = append_selector_suffix(output_png, selectors)
        jobs.append((case_dir, output_dir, output_png))
    return jobs, selectors, spin_option, options.no_fat


def process_case(case_dir, output_dir, output_png, selectors, spin_option, no_fat):
    if not output_dir.is_dir():
        return 'skip', 'no output'

    stdout_path = output_dir / 'stdout.txt'
    if not stdout_path.is_file():
        return 'skip', 'no stdout.txt'

    run_fdf_path = find_run_fdf_for_output(case_dir, output_dir)
    if run_fdf_path is None:
        return 'skip', 'no RUN.fdf'

    wfsx_path = find_single_file(output_dir, '*.bands.WFSX')
    if wfsx_path is None:
        return 'skip', 'no or multiple *.bands.WFSX'

    wfsx_metadata = read_wfsx_metadata(wfsx_path)
    bands_path = wfsx_path.with_suffix('')
    bands_path_data = None
    if bands_path.is_file():
        bands_path_data = read_bands_path_data(bands_path)
        if bands_path_data['nk'] != wfsx_metadata['nk']:
            raise ValueError(
                f'k-point count mismatch: bands={bands_path_data["nk"]}, '
                f'WFSX={wfsx_metadata["nk"]}'
            )
        if bands_path_data['nspin'] != wfsx_metadata['nspin']:
            raise ValueError(
                f'spin count mismatch: bands={bands_path_data["nspin"]}, '
                f'WFSX={wfsx_metadata["nspin"]}'
            )

    fermi = read_fermi_level(stdout_path)
    reference_energy, energy_reference_label = resolve_band_energy_reference(
        output_dir, run_fdf_path, fermi
    )
    if no_fat:
        nspin, spin_plots = read_plain_band_data(wfsx_path)
    else:
        nspin, spin_plots = read_fatband_data(case_dir, output_dir, run_fdf_path, wfsx_path, selectors)
    if not spin_plots:
        return 'skip', 'empty band data'

    bandlines = [] if bands_path_data is not None else read_bandlines(run_fdf_path)
    plotted_total = 0
    spin_plot_map = OrderedDict((spin_index, (data, kdist)) for spin_index, data, kdist in spin_plots)
    spin_indices = get_spin_indices(nspin, spin_option)
    plot_batches = []
    if spin_option == 'split':
        for spin_index in spin_indices:
            if spin_index in spin_plot_map:
                plot_batches.append((spin_index, [(spin_index, spin_plot_map[spin_index][0])]))
    else:
        spin_items = [
            (spin_index, spin_plot_map[spin_index][0])
            for spin_index in spin_indices
            if spin_index in spin_plot_map
        ]
        plot_batches.append((spin_option, spin_items))

    if not plot_batches or not plot_batches[0][1]:
        return 'skip', 'empty selected spin data'

    for output_spin, spin_items in plot_batches:
        wfsx_kdist = spin_plot_map[spin_items[0][0]][1]
        if bands_path_data is not None:
            kdist = bands_path_data['kdist']
            if len(kdist) != len(wfsx_kdist):
                raise ValueError(
                    f'k-point data length mismatch: bands={len(kdist)}, WFSX={len(wfsx_kdist)}'
                )
            tick_positions = [float(position) for position, _ in bands_path_data['labels']]
            tick_labels = [label for _, label in bands_path_data['labels']]
        else:
            kdist = wfsx_kdist
            tick_positions = []
            tick_labels = []
            for index, label in bandlines:
                index = min(max(index, 0), len(kdist) - 1)
                tick_positions.append(float(kdist[index]))
                tick_labels.append(label)

        plot_func = plot_band if no_fat else plot_fatband
        plotted_total += plot_func(
            get_spin_output_path(output_png, output_spin, nspin),
            spin_items,
            output_spin == 'all',
            reference_energy,
            energy_reference_label,
            kdist,
            tick_positions,
            tick_labels,
        )

    if plotted_total == 0:
        return 'skip', 'no visible band points'

    return 'done', None


def main():
    # Make band plots from WFSX outputs with optional fatband element/orbital selection.
    jobs, selectors, spin_option, no_fat = parse_band_arguments(sys.argv)
    failures = 0

    for case_dir, output_dir, output_png in jobs:
        try:
            status, message = process_case(case_dir, output_dir, output_png, selectors, spin_option, no_fat)
        except Exception as exc:
            print(f'failed : [{case_dir.name}] {exc}')
            failures += 1
            continue

        if status == 'done':
            print(f'[{case_dir.name}]')
        else:
            print(f'skip : [{case_dir.name}] {message}')
            failures += 1

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
