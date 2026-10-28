import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sisl
from scipy.linalg import eigh

from siesta_band_tools.pdos import calculate_pdos, read_orbital_index, select_orbitals


def write_fixture(directory):
    """Two nonorthogonal orbitals; H and S are k-independent."""
    geometry = sisl.Geometry([[0., 0., 0.]], [sisl.Atom(6, orbitals=[-1., -1.])],
                             lattice=sisl.Lattice([5., 5., 5.], nsc=[1, 1, 1]))
    h = sisl.Hamiltonian(geometry, orthogonal=False)
    for i, j, value in [(0,0,0.), (0,1,.3), (1,0,.3), (1,1,1.)]:
        h[i, j] = value
    for i, j, value in [(0,0,1.), (0,1,.2), (1,0,.2), (1,1,1.)]:
        h[i, j, h.S_idx] = value
    h.finalize()
    tshs = Path(directory) / 'tiny.TSHS'
    sisl.get_sile(str(tshs), 'w').write_hamiltonian(h)
    orbitals = Path(directory) / 'tiny.ORB_INDX'
    orbitals.write_text('''2 2 = orbitals in unit cell and supercell

 io ia is spec iao n l m z p sym rc isc iuo
 1 1 1 C 1 2 0 0 1 F s 4 0 0 0 1
 2 1 1 C 2 2 1 0 1 F pz 4 0 0 0 2
''')
    return tshs, orbitals


class PDOSTests(unittest.TestCase):
    def test_k_grid_and_shift_change_dispersive_pdos(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry = sisl.Geometry([[0.,0.,0.]], sisl.Atom(6),
                                     lattice=sisl.Lattice([5.,5.,5.], nsc=[3,1,1]))
            h = sisl.Hamiltonian(geometry)
            h[0,0] = 0.
            h[0,int(geometry.sc_index([1,0,0]))] = -1.
            h[0,int(geometry.sc_index([-1,0,0]))] = -1.
            h.finalize()
            path = Path(directory) / 'dispersive.TSHS'
            sisl.get_sile(str(path), 'w').write_hamiltonian(h)
            common = dict(energy_range=[-3,3], points=601, sigma=.1)
            full = calculate_pdos(path, kgrid=[2,1,1], **common)
            shifted = calculate_pdos(path, kgrid=[1,1,1], kshift=[.25,0,0], **common)
            e = full['energy_eV'] - float(full['fermi_eV'])
            gaussian = lambda center: np.exp(-.5*((e-center)/.1)**2)/(.1*np.sqrt(2*np.pi))
            np.testing.assert_allclose(full['pdos_per_spin'][0],
                                       .5*(gaussian(-2.)+gaussian(2.)), rtol=1.e-7, atol=1.e-9)
            np.testing.assert_allclose(shifted['pdos_per_spin'][0], gaussian(0.), rtol=1.e-7, atol=1.e-9)
            np.testing.assert_allclose(shifted['k_fractional'], [[.25,0,0]])

    def test_nonorthogonal_pdos_and_full_k_average(self):
        with tempfile.TemporaryDirectory() as directory:
            tshs, orbitals = write_fixture(directory)
            result = calculate_pdos(tshs, kgrid=[2,1,1], energy_range=[-4,4],
                                    points=1601, sigma=.1, orbital_index=orbitals)
            energy = result['energy_eV'] - float(result['fermi_eV'])
            eigenvalues, vectors = eigh([[0.,.3],[.3,1.]], [[1.,.2],[.2,1.]])
            overlap = np.array([[1.,.2],[.2,1.]])
            weights = (vectors.conj() * (overlap @ vectors)).real
            gaussian = np.exp(-.5 * ((energy[None,:] - eigenvalues[:,None]) / .1)**2) / (.1*np.sqrt(2*np.pi))
            expected = weights @ gaussian
            np.testing.assert_allclose(result['pdos_per_spin'], expected, rtol=1.e-9, atol=1.e-10)
            np.testing.assert_allclose(np.trapz(result['pdos_per_spin'], energy, axis=1), [1.,1.], atol=1.e-8)
            np.testing.assert_allclose(result['k_weights'], [.5,.5])
            self.assertEqual(result['pdos_per_spin'].shape, (2, 1601))
            np.testing.assert_array_equal(result['orbital_atom_indices'], [1,1])
            np.testing.assert_array_equal(result['orbital_species'], ['C','C'])
            meta = json.loads(str(result['metadata_json']))
            self.assertEqual(meta['pdos_convention'], 'Mulliken orbital projection; unpolarized per spin')
            self.assertFalse(meta['time_reversal_reduction'])
            self.assertTrue(meta['orbital_labels'])
            self.assertEqual(len(meta['orb_indx_sha256']), 64)

    def test_tshs_only_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            tshs, orbital_file = write_fixture(directory)
            raw = calculate_pdos(tshs, kgrid=[1,1,1], energy_range=[-2,2], points=41)
            self.assertNotIn('orbital_species', raw)
            np.testing.assert_array_equal(select_orbitals(raw, atoms=[1]), [0,1])
            with self.assertRaisesRegex(ValueError, 'atom selection'):
                select_orbitals(raw, atoms=[1.5])
            with self.assertRaisesRegex(ValueError, 'ORB_INDX'):
                select_orbitals(raw, element='C')
            labeled = calculate_pdos(tshs, kgrid=[1,1,1], energy_range=[-2,2], points=41,
                                      orbital_index=orbital_file)
            np.testing.assert_array_equal(select_orbitals(labeled, element='C', orbital='2pz'), [1])
            np.testing.assert_array_equal(select_orbitals(labeled, axis='y', coordinate=0., tolerance=.1), [0,1])
            with self.assertRaisesRegex(ValueError, 'no orbitals'):
                select_orbitals(labeled, element='W')
            with self.assertRaisesRegex(ValueError, 'together'):
                select_orbitals(labeled, axis='y')

    def test_orbital_index_mismatch_and_invalid_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            tshs, orbital_file = write_fixture(directory)
            with self.assertRaisesRegex(ValueError, 'atom order'):
                read_orbital_index(orbital_file, [1,2])
            for grid in ([0,1,1], [1.5,1,1], [1,1]):
                with self.assertRaises(ValueError):
                    calculate_pdos(tshs, kgrid=grid, energy_range=[-2,2])
            with self.assertRaises(ValueError):
                calculate_pdos(tshs, kgrid=[1,1,1], energy_range=[-2,2], sigma=0)
            fake = SimpleNamespace(spin=SimpleNamespace(kind=1))
            with patch('siesta_band_tools.pdos.read_hamiltonian', return_value=(fake, 0.)):
                with self.assertRaisesRegex(ValueError, 'unpolarized'):
                    calculate_pdos(tshs, kgrid=[1,1,1], energy_range=[-2,2])

    def test_cli_npz_plot_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            tshs, orbitals = write_fixture(directory)
            npz = Path(directory) / 'pdos.npz'
            png = Path(directory) / 'pdos.png'
            cli = [sys.executable, '-m', 'siesta_band_tools.cli', 'pdos']
            command = [*cli, 'calculate', str(tshs), str(npz), '--kgrid', '2','1','1',
                       '-er', '-2','2', '--points','101', '--sigma','.1', '--orb-indx',str(orbitals)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = npz.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(npz.read_bytes(), original)
            plot = [*cli, 'plot', str(npz), str(png), '--element','C', '--orbital','pz', '-dpi','40']
            result = subprocess.run(plot, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('selected orbitals=1', result.stdout)
            original_png = png.read_bytes()
            self.assertNotEqual(subprocess.run(plot, capture_output=True).returncode, 0)
            self.assertEqual(png.read_bytes(), original_png)

    def test_cli_tshs_only_atom_plot_and_missing_names(self):
        with tempfile.TemporaryDirectory() as directory:
            tshs, _ = write_fixture(directory)
            folder = Path(directory)
            npz, png, bad = folder / 'unlabeled.npz', folder / 'atom.png', folder / 'bad.png'
            cli = [sys.executable, '-m', 'siesta_band_tools.cli', 'pdos']
            calculate = [*cli, 'calculate', str(tshs), str(npz), '--kgrid', '1','1','1',
                         '-er','-2','2', '--points','41']
            result = subprocess.run(calculate, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            named = subprocess.run([*cli, 'plot', str(npz), str(bad), '--element','C'],
                                   capture_output=True, text=True)
            self.assertNotEqual(named.returncode, 0)
            self.assertIn('requires a matching ORB_INDX', named.stderr)
            self.assertFalse(bad.exists())
            atom = subprocess.run([*cli, 'plot', str(npz), str(png), '--atom','1', '-dpi','40'],
                                  capture_output=True, text=True)
            self.assertEqual(atom.returncode, 0, atom.stderr)
            self.assertTrue(png.is_file())


if __name__ == '__main__':
    unittest.main()
