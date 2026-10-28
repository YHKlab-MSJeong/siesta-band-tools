import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import FortranFile

from siesta_band_tools.calculation import make_path, load_path, calculate
from siesta_band_tools import legacy_plot as band
from siesta_band_tools.util import get_selector_groups, read_fermi_level, read_stdout_electrode_levels


class BandTests(unittest.TestCase):
    def test_calculate_absolute_energy_and_restore_units(self):
        import sisl.io.siesta.binaries as binary
        original = binary._Ry2eV
        class H:
            spin = SimpleNamespace(kind=0)
            geometry = SimpleNamespace(cell=np.eye(3))
            def eigh(self, k, gauge, eigvals_only):
                return np.array([k[0], k[0] + 2])
        class Source:
            def read_hamiltonian(self):
                self.assert_unit = binary._Ry2eV
                return H()
            def read_fermi_level(self):
                return -3.
        source = Source()
        with patch('sisl.get_sile', return_value=source):
            result = calculate('test.TSHS', [[0,0,0],[.5,0,0]], [1], legacy_siesta_units=True)
        self.assertEqual(source.assert_unit, 13.60580)
        self.assertEqual(binary._Ry2eV, original)
        np.testing.assert_allclose(result['energy_eV'], [[-3,-1],[-2.5,-.5]])

    def test_path_cartesian_distance(self):
        k, distance, ticks = make_path([[0, 0, 0], [.5, 0, 0], [.5, .5, 0]], [2, 1], np.diag([2., 4., 5.]))
        np.testing.assert_allclose(k, [[0,0,0],[.25,0,0],[.5,0,0],[.5,.5,0]])
        np.testing.assert_allclose(distance, [0, np.pi/4, np.pi/2, 3*np.pi/4])
        np.testing.assert_array_equal(ticks, [0,2,3])

    def test_bad_path(self):
        for n in ([0], [1, 2], [1.5]):
            with self.assertRaises(ValueError):
                make_path([[0,0,0],[1,0,0]], n, np.eye(3))

    def test_read_references_and_path(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'stdout.txt'
            file.write_text('siesta: Fermi = -2.0\nMSDFT: left, right efs -1.1 -2.2\nsiesta: Fermi = -1.5\n')
            self.assertEqual(read_fermi_level(file), -1.5)
            self.assertEqual(read_stdout_electrode_levels(file), (-1.1, -2.2))
            path = Path(folder) / 'path.json'
            path.write_text(json.dumps({'vertices': [[0,0,0],[1,0,0]], 'intervals': [2], 'labels': ['G']}))
            with self.assertRaises(ValueError):
                load_path(path)

    def test_selector(self):
        orbitals = [{'symbol': 'Mo', 'n': 4, 'l': 'd', 'name': 'dz2', 'atom_index': 1, 'position': [0, 2, 0]},
                    {'symbol': 'S', 'n': 3, 'l': 'p', 'name': 'px', 'atom_index': 2, 'position': [0, 8, 0]}]
        selection = [{'kind': 'axis', 'axis': 'y', 'coord': 2, 'symbol': 'Mo', 'orbital': '4d'}]
        np.testing.assert_array_equal(next(iter(get_selector_groups(orbitals, selection, 'symbol').values())), [0])

    def test_calculated_plot_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'bands.npz'
            png = Path(folder) / 'bands.png'
            np.savez(data, energy_eV=[[-1., 1.], [-.5, 1.5]],
                     k_distance_inv_ang=[0., 1.], tick_indices=[0, 1],
                     tick_labels=['Gamma', 'X'], fermi_eV=0.)
            command = [sys.executable, '-c', 'from siesta_band_tools.cli import plot_computed; plot_computed()',
                       str(data), str(png), '-dpi', '40']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(png.is_file())
            original = png.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(png.read_bytes(), original)

    def test_wfsx_read_and_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tiny.bands.WFSX'
            with FortranFile(path, 'w') as out:
                out.write_record(np.array([1, 1], dtype=np.int32))
                out.write_record(np.array([1], dtype=np.int32))
                out.write_record(np.array([2], dtype=np.int32))
                records = b''.join([band.WFSX_ORBITAL_RECORD.pack(i, symbol.ljust(20), 0, n, orbital.ljust(20))
                                    for i, symbol, n, orbital in [(1,b'Mo',4,b'dz2'),(2,b'S',3,b'px')]])
                out.write_record(np.frombuffer(records, dtype=np.uint8))
                out.write_record(np.frombuffer(band.WFSX_BLOCK_HEADER.pack(1,0.,0.,0.,1.), dtype=np.uint8))
                out.write_record(np.array([1], dtype=np.int32))
                out.write_record(np.array([1], dtype=np.int32))
                out.write_record(np.array([1], dtype=np.int32))
                out.write_record(np.array([2.5], dtype=np.float64))
                out.write_record(np.array([.6,.8], dtype=np.float32))
            meta = band.read_wfsx_metadata(path)
            result = band.read_wfsx_series(path, {'Mo': np.array([0])}, 1)[1]
            self.assertEqual(meta['symbols'], ['Mo','S'])
            np.testing.assert_allclose(result['eig_blocks'][0], [2.5])
            np.testing.assert_allclose(result['weight_blocks']['Mo'][0], [.36], rtol=1e-6)
            with self.assertRaisesRegex(ValueError, 'incomplete WFSX'):
                band.read_wfsx_series(path, {'Mo': np.array([0])}, 2)


if __name__ == '__main__':
    unittest.main()
