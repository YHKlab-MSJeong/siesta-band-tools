import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from siesta_band_tools import data
from siesta_band_tools.plotting import PlotStyle, draw_bands, plot_band, style_axes


class PlottingTests(unittest.TestCase):
    def test_draw_on_caller_axes_without_global_configuration(self):
        fig, ax = plt.subplots()
        try:
            energy = np.array([[-1., 8.], [1., 9.]])
            self.assertEqual(draw_bands(ax, [0., 1.], energy, reference_energy=1.,
                                        energy_range=(-2., 2.), color='red'), 1)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [-2., 0.])
            self.assertEqual(ax.lines[0].get_color(), 'red')
            style_axes(ax, [0., 1.], np.array([0., 1.]), ['G', 'X'], 'Ef',
                       PlotStyle(energy_range=(-2., 2.), hide_labels=False))
            self.assertEqual(ax.get_ylim(), (-2., 2.))
            self.assertEqual(ax.get_ylabel(), 'Energy - Ef (eV)')
            with self.assertRaises(ValueError):
                draw_bands(ax, [0.], energy)
        finally:
            plt.close(fig)

    def test_direct_png_renderer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bands.png'
            plotted = plot_band(path, [(1, np.array([[0.], [1.]]))], True, 0., 'Ef',
                                [0., 1.], [0., 1.], ['G', 'X'], PlotStyle(dpi=40))
            self.assertEqual(plotted, 1)
            self.assertTrue(path.is_file())

    def test_separate_data_selection_and_normalization(self):
        meta = {'norb': 2, 'atom_indices': np.array([1, 2]), 'symbols': ['Mo', 'S'],
                'nvalues': [4, 3], 'orbital_names': ['dz2', 'px']}
        positions = {1: [0, 2, 0], 2: [0, 8, 0]}
        selectors = [{'kind': 'plain', 'symbol': 'Mo', 'orbital': '4d'}]
        groups = data.get_groups(meta, selectors, positions, combine=False)
        np.testing.assert_array_equal(groups['Mo 4d'], [0])
        self.assertEqual(data.get_weight_normalization_factors(meta, groups, positions,
                                                               normalization='cell')['Mo 4d'], 2)
        with self.assertRaises(ValueError):
            data.get_weight_normalization_factors(meta, groups, positions, normalization='wrong')


if __name__ == '__main__':
    unittest.main()
