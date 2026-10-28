"""Overlap-aware orbital PDOS from an unpolarized SIESTA TSHS Hamiltonian."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .calculation import read_hamiltonian
from .io import normalize_orbital_name
from .util import LMAP, parse_orbital


def read_orbital_index(path, expected_atoms):
    """Read unit-cell orbital metadata, checking order and per-atom counts."""
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    try:
        count = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError('invalid ORB_INDX header') from exc
    expected_atoms = np.asarray(expected_atoms, dtype=int)
    if count != len(expected_atoms):
        raise ValueError('ORB_INDX orbital count differs from TSHS')
    rows = [line.split() for line in lines[3:] if line.strip()][:count]
    if len(rows) != count:
        raise ValueError('incomplete ORB_INDX unit-cell records')
    species, names, n, l, m, zeta = [], [], [], [], [], []
    for index, row in enumerate(rows):
        try:
            number, atom = int(row[0]), int(row[1])
            nn, ll, mm, zz = (int(row[i]) for i in (5, 6, 7, 8))
            symbol, name = row[3], normalize_orbital_name(row[10])
        except (ValueError, IndexError) as exc:
            raise ValueError(f'invalid ORB_INDX record {index + 1}') from exc
        if number != index + 1 or atom != expected_atoms[index]:
            raise ValueError(f'ORB_INDX orbital or atom order differs from TSHS at {index + 1}')
        if not symbol or not name:
            raise ValueError(f'missing ORB_INDX species or orbital name at {index + 1}')
        species.append(symbol); names.append(name)
        n.append(nn); l.append(ll); m.append(mm); zeta.append(zz)
    return {'orbital_species': np.asarray(species), 'orbital_name': np.asarray(names),
            'orbital_n': np.asarray(n), 'orbital_l': np.asarray(l),
            'orbital_m': np.asarray(m), 'orbital_zeta': np.asarray(zeta),
            'orb_indx_sha256': hashlib.sha256(raw).hexdigest()}


def calculate_pdos(tshs, *, kgrid, energy_range, points=801, sigma=.1,
                   kshift=(0., 0., 0.), orbital_index=None,
                   legacy_siesta_units=False):
    """Return orbital PDOS per spin on a full Monkhorst-Pack mesh.

    Energy limits are relative to the TSHS Fermi energy; saved energies are
    absolute in eV, matching the band NPZ. The optional ORB_INDX supplies
    labels; its order and atom grouping must match the TSHS basis.
    """
    import sisl
    from sisl.physics.distribution import get_distribution

    grid = np.asarray(kgrid)
    shift = np.asarray(kshift, dtype=float)
    limits = np.asarray(energy_range, dtype=float)
    if grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer) or np.any(grid < 1):
        raise ValueError('kgrid must contain three positive integers')
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError('kshift must contain three finite fractional coordinates')
    if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError('energy_range must contain finite MIN < MAX')
    if isinstance(points, bool) or int(points) != points or points < 2:
        raise ValueError('points must be an integer >= 2')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Gaussian sigma must be positive and finite')

    tshs = Path(tshs)
    before = tshs.stat()
    H, fermi = read_hamiltonian(tshs, legacy_siesta_units=legacy_siesta_units)
    if H.spin.kind != 0:
        raise ValueError('PDOS currently supports only unpolarized Hamiltonians')
    if not np.isfinite(fermi):
        raise ValueError('non-finite TSHS Fermi energy')
    geometry = H.geometry
    positions = np.asarray(geometry.xyz, dtype=float)
    cell = np.asarray(geometry.cell, dtype=float)
    if (positions.shape != (geometry.na, 3) or cell.shape != (3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(cell).all()):
        raise ValueError('invalid TSHS geometry')
    first = np.asarray(geometry.firsto, dtype=int)
    if (first.shape != (geometry.na + 1,) or first[0] != 0 or first[-1] != geometry.no
            or np.any(np.diff(first) < 0)):
        raise ValueError('invalid TSHS atom-to-orbital mapping')
    atoms = np.repeat(np.arange(1, geometry.na + 1), np.diff(first))
    labels = read_orbital_index(orbital_index, atoms) if orbital_index is not None else None
    relative_energy = np.linspace(limits[0], limits[1], int(points))
    grid_object = sisl.MonkhorstPack(H, grid.tolist(), displacement=shift, centered=True, trs=False)
    kpoints = np.asarray(grid_object.k, dtype=float)
    weights = np.asarray(grid_object.weight, dtype=float)
    if (len(weights) != math.prod(int(n) for n in grid)
            or kpoints.shape != (len(weights), 3) or not np.all(np.isfinite(kpoints))
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=1.e-10, atol=1.e-12)):
        raise ValueError('invalid or unnormalized Brillouin-zone grid weights')
    distribution = get_distribution('gaussian', smearing=float(sigma))
    pdos = np.zeros((geometry.no, len(relative_energy)), dtype=float)
    for kpoint, weight in zip(kpoints, weights):
        state = H.eigenstate(k=kpoint, gauge='R')
        contribution = np.asarray(state.PDOS(relative_energy, distribution=distribution))
        if contribution.shape != (1, geometry.no, len(relative_energy)) or not np.isfinite(contribution).all():
            raise ValueError('unexpected or non-finite orbital PDOS')
        pdos += weight * contribution[0]
    if not np.isfinite(pdos).all():
        raise ValueError('non-finite averaged PDOS')
    after = tshs.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('TSHS changed during PDOS calculation')
    metadata = {'schema_version': 1, 'sisl_version': sisl.__version__,
                'source_tshs': tshs.name, 'source_size_bytes': before.st_size,
                'source_mtime_ns': before.st_mtime_ns, 'energy_convention': 'absolute eV',
                'pdos_convention': 'Mulliken orbital projection; unpolarized per spin',
                'kgrid': grid.tolist(), 'kshift_fractional': shift.tolist(),
                'kgrid_centered': True, 'time_reversal_reduction': False,
                'gaussian_sigma_eV': float(sigma), 'legacy_siesta_units': bool(legacy_siesta_units),
                'orbital_labels': labels is not None}
    arrays = {'energy_eV': relative_energy + fermi, 'fermi_eV': np.array(fermi),
              'pdos_per_spin': pdos, 'orbital_atom_indices': atoms,
              'atom_positions_ang': positions, 'cell_ang': cell,
              'k_fractional': kpoints, 'k_weights': weights}
    if labels is not None:
        metadata['orb_indx_sha256'] = labels.pop('orb_indx_sha256')
        metadata['orb_indx_name'] = Path(orbital_index).name
        arrays.update(labels)
    arrays['metadata_json'] = np.array(json.dumps(metadata))
    return arrays


def select_orbitals(data, *, atoms=None, element=None, orbital=None,
                    axis=None, coordinate=None, tolerance=1.5):
    """Return selected orbital indices; all supplied filters are intersected."""
    atom_ids = np.asarray(data['orbital_atom_indices'])
    positions = np.asarray(data['atom_positions_ang'], dtype=float)
    if (atom_ids.ndim != 1 or not np.issubdtype(atom_ids.dtype, np.integer)
            or positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all()):
        raise ValueError('invalid atomic mapping in PDOS archive')
    if len(atom_ids) == 0 or np.any(atom_ids < 1) or np.any(atom_ids > len(positions)):
        raise ValueError('invalid atom indices in PDOS archive')
    selected = np.ones(len(atom_ids), dtype=bool)
    if atoms is not None:
        ids = np.asarray(atoms)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0
                or np.any(ids < 1) or np.any(ids > len(positions))):
            raise ValueError('invalid atom selection')
        selected &= np.isin(atom_ids, ids)
    if element is not None or orbital is not None:
        required = ['orbital_species', 'orbital_name', 'orbital_n', 'orbital_l']
        if any(key not in data or len(data[key]) != len(atom_ids) for key in required):
            raise ValueError('element/orbital selection requires a matching ORB_INDX during calculation')
    if element is not None:
        selected &= np.asarray(data['orbital_species']) == element
    if orbital is not None:
        parsed = parse_orbital(orbital)
        letters = np.array([LMAP.get(int(value), '') for value in data['orbital_l']])
        selected &= letters == parsed['l']
        if parsed['n'] is not None:
            selected &= np.asarray(data['orbital_n']) == parsed['n']
        if len(parsed['name']) > 1:
            selected &= np.asarray(data['orbital_name']) == parsed['name']
    if (axis is None) != (coordinate is None):
        raise ValueError('axis and coordinate must be specified together')
    if axis is not None:
        if axis not in ('x', 'y', 'z') or not np.isfinite(coordinate) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('invalid axis, coordinate or tolerance')
        selected &= np.abs(positions[atom_ids - 1, 'xyz'.index(axis)] - coordinate) <= tolerance
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        raise ValueError('no orbitals match the requested selection')
    return indices
