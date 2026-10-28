"""Calculate unpolarized SIESTA TSHS eigenvalues on a supercell k path."""
import importlib
import json
from pathlib import Path

import numpy as np


def diagonalize(hamiltonian, k, *, eigenvectors=False):
    """Diagonalize an existing sisl Hamiltonian at a fractional supercell k point."""
    k = np.asarray(k, dtype=float)
    if k.shape != (3,) or not np.isfinite(k).all():
        raise ValueError('k must be a finite three-dimensional fractional coordinate')
    return hamiltonian.eigh(k=k, eigvals_only=not eigenvectors, gauge='R')


def make_path(vertices, intervals, cell):
    """Return fractional k points, Cartesian cumulative distance, and vertex indices.

    Vertices are in the reciprocal basis of the Hamiltonian's supercell.
    Cell vectors are rows in Angstrom. Distances are in inverse Angstrom.
    """
    vertices = np.asarray(vertices, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError('vertices must be an array of at least two 3D points')
    if not np.isfinite(vertices).all() or cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError('vertices and cell must be finite, cell must be 3x3')
    if len(intervals) != len(vertices) - 1 or any(int(n) != n or n < 1 for n in intervals):
        raise ValueError('one positive integer interval count is required per segment')
    points = np.vstack([np.linspace(vertices[i], vertices[i+1], int(n)+1)[:-1]
                        for i, n in enumerate(intervals)] + [vertices[-1:]])
    cartesian = points @ (2 * np.pi * np.linalg.inv(cell).T)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    ticks = np.r_[0, np.cumsum(intervals)].astype(int)
    return points, distance, ticks


def calculate(tshs, vertices, intervals, *, legacy_siesta_units=False):
    """Return absolute energies in eV. TSHS fractional coordinates use supercell reciprocal vectors.

    The legacy-unit switch applies only to known SIESTA binaries built with
    Bohr=0.529177 Ang and Ry=13.60580 eV. sisl conversion constants are
    temporarily changed during reading and always restored afterwards.
    """
    import sisl
    binary = importlib.import_module('sisl.io.siesta.binaries')
    old = {}
    if legacy_siesta_units:
        values = {'_Bohr2Ang': 0.529177, '_Ry2eV': 13.60580, '_eV2Ry': 1.0 / 13.60580}
        if not all(hasattr(binary, name) for name in values):
            raise RuntimeError('unsupported sisl unit conversion internals')
        old = {key: getattr(binary, key) for key in values}
        for key, value in values.items():
            setattr(binary, key, value)
    try:
        source = sisl.get_sile(str(tshs))
        H = source.read_hamiltonian()
        offset = float(source.read_fermi_level())
    finally:
        for key, value in old.items():
            setattr(binary, key, value)
    if H.spin.kind != 0:
        raise ValueError('TSHS calculation currently supports only unpolarized Hamiltonians')
    k, distance, ticks = make_path(vertices, intervals, H.geometry.cell)
    cache = {}
    energies = []
    for point in k:
        key = tuple(point)
        if key not in cache:
            cache[key] = np.asarray(diagonalize(H, point)) + offset
        energies.append(cache[key])
    energies = np.stack(energies)
    if not np.isfinite(energies).all():
        raise ValueError('non-finite eigenvalues')
    return {'energy_eV': energies, 'k_fractional': k, 'k_distance_inv_ang': distance,
            'tick_indices': ticks, 'fermi_eV': np.array(offset),
            'cell_ang': np.asarray(H.geometry.cell),
            'metadata_json': np.array(json.dumps({'sisl_version': sisl.__version__,
                'energy_convention': 'absolute; sisl eigenvalues plus TSHS Fermi',
                'legacy_siesta_units': legacy_siesta_units}))}


def load_path(path):
    data = json.loads(Path(path).read_text())
    labels = data['labels']
    if len(labels) != len(data['vertices']) or not all(isinstance(x, str) for x in labels):
        raise ValueError('provide one string label per vertex')
    return data['vertices'], data['intervals'], labels
