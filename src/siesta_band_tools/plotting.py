"""Band drawing functions usable with Matplotlib axes or as PNG renderers."""
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from .util import get_color_map


@dataclass
class PlotStyle:
    width: float = 3.5
    height: float = 5.
    dpi: int = 280
    energy_range: tuple = (-3., 3.)
    energy_tick_count: int = 5
    marker_scale: float = 26.
    alpha: float = .70
    reference_linewidth: float = .1
    label_size: float = 16.
    tick_size: float = 16.
    hide_labels: bool = True
    show_legend: bool = False
    axis_linewidth: float = 2.2
    tick_width: float = 2.2
    tick_length: float = 12.
    group_colors: object = field(default_factory=lambda: ['k', 'g', 'b', 'k'])
    spin_group_colors: dict = field(default_factory=lambda: {1: ['b', 'g', 'b', 'k'], 2: ['r', 'orange', 'purple', 'r']})
    weight_tol: float = 1.e-4
    base_band_linewidth: float = .7
    default_marker_scale: float = 26.


def style_axes(ax, kdist, tick_positions, tick_labels, reference_label, style=None):
    """Apply the standard axes style to an existing Matplotlib axes."""
    style = style or PlotStyle()
    ax.tick_params(which='both', direction='in', top=False, right=False,
                   labelsize=style.tick_size, width=style.tick_width, length=style.tick_length)
    if style.hide_labels:
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        ax.set_xlabel('k-path', fontsize=style.label_size)
        ax.set_ylabel(f'Energy - {reference_label} (eV)', fontsize=style.label_size)
    ax.set_ylim(*style.energy_range)
    if style.energy_tick_count is not None:
        if style.energy_tick_count < 2:
            raise ValueError('energy_tick_count should be at least 2 or None')
        ax.set_yticks(np.linspace(*style.energy_range, style.energy_tick_count))
    if len(kdist) > 0:
        ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    if len(tick_positions):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        for xpos in tick_positions:
            ax.axvline(x=xpos, color='0.85', linewidth=0.7, zorder=0)
    else:
        ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_linewidth)
    ax.axhline(y=0.0, color='0.55', linewidth=style.reference_linewidth, zorder=0)


def draw_bands(ax, kdist, energy, *, reference_energy=0., energy_range=(-3., 3.),
               color='k', linewidth=.7):
    """Draw an (nk, nbands) energy array on an existing axes; return line count."""
    energy = np.asarray(energy)
    if energy.ndim != 2 or len(kdist) != len(energy):
        raise ValueError('energy must have shape (len(kdist), nbands)')
    ymin, ymax = energy_range
    plotted = 0
    for band_index in range(energy.shape[1]):
        y = energy[:, band_index] - reference_energy
        if not np.any((y >= ymin) & (y <= ymax)):
            continue
        ax.plot(kdist, y, color=color, linewidth=linewidth)
        plotted += 1
    return plotted


def draw_pdos(ax, energy, values, *, reference_energy=0., color='k'):
    """Draw a one-dimensional selected PDOS on an existing axes."""
    energy = np.asarray(energy, dtype=float)
    values = np.asarray(values, dtype=float)
    if (energy.ndim != 1 or values.shape != energy.shape or len(energy) < 2
            or not np.isfinite(energy).all() or not np.isfinite(values).all()
            or not np.isfinite(reference_energy)):
        raise ValueError('energy and PDOS must be finite one-dimensional arrays of equal length')
    return ax.plot(energy - reference_energy, values, color=color)[0]


def plot_pdos(output, energy, values, *, reference_energy=0., reference_label='Fermi',
              energy_range=None, figsize=(5., 3.5), dpi=280, color='k'):
    """Render a selected PDOS; energy is plotted on the horizontal axis."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        draw_pdos(ax, energy, values, reference_energy=reference_energy, color=color)
        if energy_range is not None:
            if (len(energy_range) != 2 or not np.isfinite(energy_range).all()
                    or energy_range[0] >= energy_range[1]):
                raise ValueError('energy_range must contain finite MIN < MAX')
            ax.set_xlim(*energy_range)
        ax.set_xlabel(f'Energy - {reference_label} (eV)')
        ax.set_ylabel('PDOS (states/eV per spin)')
        ax.axvline(0., color='0.55', linewidth=.5)
        fig.tight_layout()
        fig.savefig(output, dpi=dpi, transparent=True, format='png')
    finally:
        plt.close(fig)


def group_color_map(labels, spin_index, style):
    labels = list(labels)
    colors = style.group_colors
    if spin_index is not None:
        colors = style.spin_group_colors.get(spin_index, style.group_colors)
    if colors is None:
        return get_color_map(labels)
    if len(colors) == 0:
        raise ValueError('group color lists should not be empty; use None for the default palette')
    return {label: colors[index % len(colors)] for index, label in enumerate(labels)}


def band_color(spin_index, merge_spins, style):
    if not merge_spins:
        colors = style.spin_group_colors.get(spin_index, style.group_colors)
        if colors:
            return colors[0]
    if style.group_colors:
        return style.group_colors[0]
    return 'k'


def band_linewidth(style):
    return style.base_band_linewidth * style.marker_scale / style.default_marker_scale


def plot_fatband(output_png, spin_items, merge_spins, reference_energy, energy_reference_label,
                 kdist, tick_positions, tick_labels, style=None):
    """Render selected WFSX weights to a PNG using explicit plot settings."""
    style = style or PlotStyle()
    fig, ax = plt.subplots(figsize=(style.width, style.height))
    ymin, ymax = style.energy_range
    plotted = 0
    plotted_labels = set()
    try:
        for spin_index, data in spin_items:
            color_map = group_color_map(data.keys(), None if merge_spins else spin_index, style)
            for label, series in data.items():
                x, y, w = series['x'], series['y'] - reference_energy, series['w']
                mask = (y >= ymin) & (y <= ymax) & (w > style.weight_tol)
                if not np.any(mask):
                    continue
                size = np.maximum(w[mask], style.weight_tol) * style.marker_scale
                plot_label = '_nolegend_' if merge_spins and label in plotted_labels else label
                ax.scatter(x[mask], y[mask], s=size, color=color_map[label], alpha=style.alpha,
                           linewidths=0.0, label=plot_label)
                plotted_labels.add(label)
                plotted += 1
        style_axes(ax, kdist, tick_positions, tick_labels, energy_reference_label, style)
        if style.show_legend and plotted:
            ax.legend(frameon=False, fontsize=style.tick_size, loc='upper right')
        fig.tight_layout()
        fig.savefig(output_png, dpi=style.dpi, transparent=True)
    finally:
        plt.close(fig)
    return plotted


def plot_band(output_png, spin_items, merge_spins, reference_energy, energy_reference_label,
              kdist, tick_positions, tick_labels, style=None):
    """Render band energies to a PNG using explicit plot settings."""
    style = style or PlotStyle()
    fig, ax = plt.subplots(figsize=(style.width, style.height))
    try:
        plotted = sum(draw_bands(ax, kdist, energy, reference_energy=reference_energy,
                                 energy_range=style.energy_range,
                                 color=band_color(spin_index, merge_spins, style),
                                 linewidth=band_linewidth(style))
                      for spin_index, energy in spin_items)
        style_axes(ax, kdist, tick_positions, tick_labels, energy_reference_label, style)
        fig.tight_layout()
        fig.savefig(output_png, dpi=style.dpi, transparent=True)
    finally:
        plt.close(fig)
    return plotted
