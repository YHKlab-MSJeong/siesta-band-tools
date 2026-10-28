"""SIESTA band, WFSX and EIG readers adapted from the original band.py."""
from collections import OrderedDict
from pathlib import Path
import struct
import numpy as np
from scipy.io import FortranFile, FortranEOFError
from .util import LMAP

WFSX_ORBITAL_RECORD = struct.Struct('<i20sii20s')
WFSX_BLOCK_HEADER = struct.Struct('<idddd')

def normalize_k_label(label):
    token = label.strip()
    if token in [r'\Gamma', 'Gamma', 'G', 'Γ']:
        return r'$\Gamma$'
    if token.startswith('\\'):
        return '$' + token + '$'
    return token


def read_bands_path_data(bands_path):
    lines = Path(bands_path).read_text(encoding='utf-8', errors='replace').splitlines()
    if len(lines) < 5:
        raise ValueError(f'invalid or incomplete bands file [{bands_path}]')

    try:
        nbands, nspin, nk = [int(value) for value in lines[3].split()[:3]]
    except (ValueError, IndexError) as exc:
        raise ValueError(f'invalid bands dimensions in [{bands_path}]') from exc
    if nbands < 1 or nspin < 1 or nk < 1:
        raise ValueError(f'invalid bands dimensions [{nbands}, {nspin}, {nk}] in [{bands_path}]')

    line_index = 4
    values_per_kpoint = 1 + nbands * nspin
    kdist = []
    for kpoint_index in range(nk):
        record = []
        while len(record) < values_per_kpoint:
            if line_index >= len(lines):
                raise ValueError(
                    f'incomplete band record [{kpoint_index + 1}/{nk}] in [{bands_path}]'
                )
            fields = lines[line_index].split()
            line_index += 1
            if not fields:
                continue
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f'non-numeric band record [{kpoint_index + 1}/{nk}] in [{bands_path}]'
                ) from exc
            if len(record) + len(values) > values_per_kpoint:
                raise ValueError(
                    f'too many values in band record [{kpoint_index + 1}/{nk}] in [{bands_path}]'
                )
            record.extend(values)
        kdist.append(record[0])

    while line_index < len(lines) and not lines[line_index].strip():
        line_index += 1
    if line_index >= len(lines):
        raise ValueError(f'missing high-symmetry-point footer in [{bands_path}]')

    count_fields = lines[line_index].split()
    line_index += 1
    try:
        nlabels = int(count_fields[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f'invalid high-symmetry-point count in [{bands_path}]') from exc
    if nlabels < 0:
        raise ValueError(f'invalid high-symmetry-point count [{nlabels}] in [{bands_path}]')

    labels = []
    for label_index in range(nlabels):
        while line_index < len(lines) and not lines[line_index].strip():
            line_index += 1
        if line_index >= len(lines):
            raise ValueError(
                f'incomplete high-symmetry-point footer [{label_index + 1}/{nlabels}] '
                f'in [{bands_path}]'
            )
        fields = lines[line_index].split(maxsplit=1)
        line_index += 1
        if len(fields) < 2:
            raise ValueError(
                f'invalid high-symmetry-point record [{label_index + 1}/{nlabels}] '
                f'in [{bands_path}]'
            )
        try:
            position = float(fields[0])
        except ValueError as exc:
            raise ValueError(
                f'invalid high-symmetry-point position [{fields[0]}] in [{bands_path}]'
            ) from exc
        label = fields[1].strip().strip("'\"")
        if not label:
            raise ValueError(f'empty high-symmetry-point label in [{bands_path}]')
        labels.append((position, normalize_k_label(label)))

    kdist = np.asarray(kdist, dtype=float)
    if not np.all(np.isfinite(kdist)):
        raise ValueError(f'non-finite k-path coordinate in [{bands_path}]')
    if np.any(np.diff(kdist) < -1.0e-10):
        raise ValueError(f'non-monotonic k-path coordinates in [{bands_path}]')

    if labels:
        label_positions = np.asarray([position for position, _ in labels], dtype=float)
        if not np.all(np.isfinite(label_positions)):
            raise ValueError(f'non-finite high-symmetry-point position in [{bands_path}]')
        if np.any(np.diff(label_positions) < -1.0e-10):
            raise ValueError(f'non-monotonic high-symmetry-point positions in [{bands_path}]')
        tolerance = 1.0e-8 * max(1.0, abs(float(kdist[-1] - kdist[0])))
        if label_positions[0] < kdist[0] - tolerance or label_positions[-1] > kdist[-1] + tolerance:
            raise ValueError(f'high-symmetry-point position outside k-path range in [{bands_path}]')

    return {
        'nbands': nbands,
        'nspin': nspin,
        'nk': nk,
        'kdist': kdist,
        'labels': labels,
    }


def normalize_orbital_name(name):
    name = name.strip().lower()
    if len(name) > 1 and name[0] == 'p' and name[1] in LMAP.values():
        return name[1:]
    return name


def read_wfsx_metadata(wfsx_path):
    with FortranFile(str(wfsx_path), 'r') as handle:
        nk, gamma = [int(value) for value in handle.read_record(np.int32)]
        nspin = int(handle.read_record(np.int32)[0])
        norb = int(handle.read_record(np.int32)[0])
        raw = bytes(handle.read_record(np.uint8))

    if len(raw) != norb * WFSX_ORBITAL_RECORD.size:
        raise ValueError(f'invalid orbital metadata block in [{wfsx_path}]')

    atom_indices = np.zeros(norb, dtype=int)
    symbols = []
    nvalues = np.zeros(norb, dtype=int)
    orbital_names = []

    for index in range(norb):
        start = index * WFSX_ORBITAL_RECORD.size
        stop = start + WFSX_ORBITAL_RECORD.size
        atom_index, symbol, _, nvalue, orbital_name = WFSX_ORBITAL_RECORD.unpack(raw[start:stop])
        atom_indices[index] = int(atom_index)
        symbols.append(symbol.decode('ascii', errors='replace').strip())
        nvalues[index] = int(nvalue)
        orbital_names.append(normalize_orbital_name(orbital_name.decode('ascii', errors='replace')))

    return {
        'nk': nk,
        'gamma': gamma,
        'nspin': nspin,
        'norb': norb,
        'atom_indices': atom_indices,
        'symbols': symbols,
        'nvalues': nvalues,
        'orbital_names': orbital_names,
    }


def read_wfsx_series(wfsx_path, groups, nspin):
    spin_series = OrderedDict()
    for spin_index in range(1, nspin + 1):
        spin_series[spin_index] = {
            'eig_blocks': [],
            'weight_blocks': OrderedDict((label, []) for label in groups),
            'kpoints': [],
        }

    with FortranFile(str(wfsx_path), 'r') as handle:
        nk, _ = [int(value) for value in handle.read_record(np.int32)]
        _ = int(handle.read_record(np.int32)[0])
        norb = int(handle.read_record(np.int32)[0])
        _ = handle.read_record(np.uint8)

        while True:
            try:
                header = bytes(handle.read_record(np.uint8))
            except FortranEOFError:
                break

            if len(header) != WFSX_BLOCK_HEADER.size:
                raise ValueError(f'invalid block header in [{wfsx_path}]')

            _, kx, ky, kz, _ = WFSX_BLOCK_HEADER.unpack(header)
            spin_index = int(handle.read_record(np.int32)[0])
            if spin_index not in spin_series:
                raise ValueError(f'invalid spin index [{spin_index}] in [{wfsx_path}]')
            nband = int(handle.read_record(np.int32)[0])
            if nband < 1 or nband > norb:
                raise ValueError(f'invalid band count [{nband}] in [{wfsx_path}]')

            eig = np.zeros(nband, dtype=float)
            coeff2 = np.zeros((nband, norb), dtype=float)
            first_band_index = None

            for iband in range(nband):
                band_index = int(handle.read_record(np.int32)[0])
                if first_band_index is None:
                    first_band_index = band_index
                    if first_band_index < 1:
                        raise ValueError(f'invalid band index [{band_index}] in [{wfsx_path}]')
                expected_band_index = first_band_index + iband
                if band_index != expected_band_index:
                    raise ValueError(
                        f'unexpected band index [{band_index}], expected '
                        f'[{expected_band_index}] in [{wfsx_path}]'
                    )

                eig[iband] = float(handle.read_record(np.float64)[0])
                coeff = np.array(handle.read_record(np.float32), dtype=float)
                if len(coeff) == norb:
                    coeff2[iband, :] = coeff ** 2
                elif len(coeff) == 2 * norb:
                    coeff = coeff.reshape((2, norb), order='F')
                    coeff2[iband, :] = coeff[0] ** 2 + coeff[1] ** 2
                else:
                    raise ValueError(
                        f'unexpected WFSX coefficient length [{len(coeff)}] in [{wfsx_path}]'
                    )

            spin_series[spin_index]['eig_blocks'].append(eig)
            spin_series[spin_index]['kpoints'].append([kx, ky, kz])
            for label, orbital_indices in groups.items():
                spin_series[spin_index]['weight_blocks'][label].append(
                    coeff2[:, orbital_indices].sum(axis=1)
                )

    for spin_index, series in spin_series.items():
        if len(series['eig_blocks']) != nk:
            raise ValueError(f'incomplete WFSX spin {spin_index}: expected {nk} k-points, '
                             f'found {len(series["eig_blocks"])}')
        counts = {len(block) for block in series['eig_blocks']}
        if len(counts) != 1:
            raise ValueError(f'inconsistent WFSX band count for spin {spin_index}')
    return spin_series


def read_eig_vbm(eig_path):
    lines = Path(eig_path).read_text(encoding='utf-8', errors='replace').splitlines()
    if len(lines) < 2:
        raise ValueError(f'invalid EIG file [{eig_path}]')

    try:
        fermi = float(lines[0].split()[0])
        neig, nspin, nkpt = [int(value) for value in lines[1].split()[:3]]
    except (IndexError, ValueError) as exc:
        raise ValueError(f'invalid EIG header [{eig_path}]') from exc

    values_per_kpoint = neig * nspin
    homo_values = []
    line_index = 2

    for _ in range(nkpt):
        values = []
        first_line = True
        while len(values) < values_per_kpoint and line_index < len(lines):
            fields = lines[line_index].split()
            line_index += 1
            if not fields:
                continue
            if first_line:
                fields = fields[1:]
                first_line = False
            values.extend(float(value) for value in fields)

        if len(values) < values_per_kpoint:
            raise ValueError(f'incomplete EIG eigenvalue block [{eig_path}]')

        below_fermi = [value for value in values[:values_per_kpoint] if value <= fermi]
        if below_fermi:
            homo_values.append(max(below_fermi))

    if not homo_values:
        raise ValueError(f'failed to find VBM/HOMO below Fermi level in [{eig_path}]')

    return max(homo_values)
