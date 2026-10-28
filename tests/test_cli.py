import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from siesta_band_tools import cli


class CLITests(unittest.TestCase):
    def test_calculate_dispatch_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'kpath.json'
            output = folder / 'bands.npz'
            path.write_text(json.dumps({'vertices': [[0,0,0], [.5,0,0]],
                                        'intervals': [1], 'labels': ['G', 'X']}))
            result = {'energy_eV': np.array([[0.], [1.]]), 'fermi_eV': np.array(0.),
                      'k_distance_inv_ang': np.array([0., 1.]), 'tick_indices': np.array([0, 1])}
            with patch('siesta_band_tools.cli.calculate', return_value=result) as calculate:
                cli.main(['calculate', 'model.TSHS', str(path), str(output), '--legacy-siesta-units'])
                calculate.assert_called_once_with(Path('model.TSHS'), [[0,0,0], [.5,0,0]], [1],
                                                   legacy_siesta_units=True)
                with np.load(output, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['energy_eV'], [[0.], [1.]])
                    np.testing.assert_array_equal(saved['tick_labels'], ['G', 'X'])
                original = output.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(['calculate', 'model.TSHS', str(path), str(output)])
                self.assertEqual(output.read_bytes(), original)
                self.assertEqual(calculate.call_count, 1)

    def test_plot_dispatch_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'bands.npz'
            output = Path(directory) / 'bands.png'
            np.savez(data, energy_eV=[[-1., 1.], [-.5, 1.5]],
                     k_distance_inv_ang=[0., 1.], tick_indices=[0, 1],
                     tick_labels=['Gamma', 'X'], fermi_eV=0.)
            command = [sys.executable, '-m', 'siesta_band_tools.cli', 'plot',
                       str(data), str(output), '-er', '-2', '2', '-dpi', '40']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = output.read_bytes()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_wfsx_arguments_forwarded(self):
        args = ['case/output', 'y', '23.6', 'W', '-eref', 'left', '-s1']
        with patch('siesta_band_tools.legacy_plot.main', return_value=0) as plot:
            self.assertEqual(cli.main(['wfsx', 'plot', *args]), 0)
        plot.assert_called_once_with(args, prog='siesta-band wfsx plot')

    def test_help_and_invalid_commands(self):
        for args, expected in [(['--help'], 'calculate'), (['calculate', '--help'], 'TSHS'),
                               (['plot', '--help'], 'NPZ'), (['wfsx', '--help'], 'plot'),
                               (['wfsx', 'plot', '--help'], 'siesta-band wfsx plot')]:
            result = subprocess.run([sys.executable, '-m', 'siesta_band_tools.cli', *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)
        for args in ([], ['bad'], ['wfsx'], ['wfsx', 'bad']):
            result = subprocess.run([sys.executable, '-m', 'siesta_band_tools.cli', *args],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
