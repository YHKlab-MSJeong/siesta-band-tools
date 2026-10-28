"""Prepare SIESTA WFSX bands and weights independently of plotting."""
from collections import OrderedDict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .io import read_wfsx_metadata, read_wfsx_series
from .util import get_output_file_path, get_selector_groups, read_structure

BOHR_TO_ANG = 0.529177210903

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


def get_groups(metadata, selectors, atom_positions, *, combine=True, combined_label='selected'):
    orbital_info = build_orbital_info(metadata, atom_positions)
    groups = get_selector_groups(orbital_info, selectors, 'symbol')
    if combine and selectors:
        indices = []
        for group_indices in groups.values():
            indices.extend(group_indices)
        groups = OrderedDict(
            [(combined_label, np.array(sorted(set(indices)), dtype=int))]
        )
    return groups


def get_weight_normalization_factors(metadata, groups, atom_positions, *, normalization='none'):
    if normalization not in ('none', 'atom', 'cell'):
        raise ValueError('normalization must be none, atom or cell')
    if normalization == 'none':
        return {label: 1 for label in groups}

    if normalization == 'cell':
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


def read_fatband_data(case_dir, output_dir, run_fdf_path, wfsx_path, selectors, *, combine=True, combined_label='selected', normalization='none'):
    metadata = read_wfsx_metadata(wfsx_path)
    atom_positions = {}

    struct_path = find_struct_fdf(case_dir, output_dir)
    if normalization == 'cell' and struct_path is None:
        raise ValueError('cell weight normalization requires STRUCT.fdf')
    if struct_path is not None:
        atom_positions.update(read_struct_atom_positions(struct_path))

    pdos_path = get_output_file_path(str(output_dir), str(run_fdf_path), '.PDOS.xml')
    if pdos_path is not None and not atom_positions:
        atom_positions.update(read_pdos_atom_positions(Path(pdos_path)))

    groups = get_groups(metadata, selectors, atom_positions, combine=combine, combined_label=combined_label)
    normalization_factors = get_weight_normalization_factors(metadata, groups, atom_positions, normalization=normalization)
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
