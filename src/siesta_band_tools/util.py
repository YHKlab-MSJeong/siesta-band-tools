"""Helpers adapted from the original band.py utility functions."""
import glob
import os
import re
from collections import OrderedDict
from pathlib import Path
import numpy as np

AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
POSITION_TOL = 1.5
LMAP = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
SPECIES_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9_.+\-]*$')
ORBITAL_PATTERN = re.compile(r'^(\d+)?([spdf](?:x|y|z|xy|yz|z2|xz|x2-y2)?)$', re.IGNORECASE)

def read_structure(path):
    import sisl
    geometry = sisl.get_sile(str(path)).read_geometry()
    if geometry is None:
        raise ValueError(f'could not read structure from {path}')
    return geometry

def get_system_name(run_fdf_path):
    with open(run_fdf_path, 'r', encoding='utf-8', errors='replace') as run_handle:
        run_text = run_handle.read()

    label_match = re.search(r'^\s*SystemLabel\s+(\S+)', run_text, flags=re.M)
    if label_match is not None:
        return label_match.group(1)

    name_match = re.search(r'^\s*SystemName\s+(\S+)', run_text, flags=re.M)
    if name_match is not None:
        return name_match.group(1)

    return None


def get_output_file_path(output_dir, run_fdf_path, extension):
    system_name = get_system_name(run_fdf_path)
    if system_name is not None:
        guessed_path = os.path.join(output_dir, f'{system_name}{extension}')
        if os.path.isfile(guessed_path):
            return guessed_path

    paths = sorted(glob.glob(os.path.join(output_dir, f'*{extension}')))
    if not paths:
        return None
    return paths[0]


def find_single_file(directory, pattern):
    paths = sorted(Path(directory).glob(pattern))
    if len(paths) != 1:
        return None
    return paths[0]


def find_run_fdf(case_dir):
    case_dir = Path(case_dir)
    candidates = [case_dir / 'output' / 'RUN.fdf', case_dir / 'input' / 'RUN.fdf']
    for path in candidates:
        if path.is_file():
            return path
    return None


def read_fermi_level(stdout_path):
    fermi = None
    pattern = re.compile(r'siesta:\s*Fermi\s*=\s*([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)')

    with open(stdout_path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                fermi = float(match.group(1))

    if fermi is None:
        raise ValueError(f'failed to find Fermi level in [{stdout_path}]')

    return fermi


def read_stdout_electrode_levels(stdout_path):
    left_level = None
    right_level = None
    marker = 'MSDFT: left, right efs'

    with open(stdout_path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if marker not in line:
                continue

            values = line.split(marker, 1)[1].split()
            if len(values) < 2:
                continue

            left_level = float(values[0])
            right_level = float(values[1])

    if left_level is None or right_level is None:
        raise ValueError(f'failed to find MSDFT left/right electrode levels in [{stdout_path}]')

    return left_level, right_level


def parse_orbital(orbital):
    match = ORBITAL_PATTERN.fullmatch(orbital)
    if match is None:
        raise ValueError(f'invalid orbital token [{orbital}]')

    nvalue = match.group(1)
    if nvalue is not None:
        nvalue = int(nvalue)

    name = match.group(2).lower()
    return {
        'n': nvalue,
        'l': name[0],
        'name': name,
    }


def is_position_selector_token(token):
    return token.lower() in ['x', 'y', 'z']


def is_atom_selector_token(token):
    return token.lower() == 'atom'


def is_species_token(token):
    return SPECIES_PATTERN.fullmatch(token) is not None


def orbital_matches(orbital_info, orbital):
    parsed = parse_orbital(orbital)

    if parsed['n'] is not None and orbital_info['n'] != parsed['n']:
        return False

    if len(parsed['name']) == 1:
        return orbital_info['l'] == parsed['l']

    return orbital_info['name'] == parsed['name']


def get_selector_label(selector):
    if selector['kind'] == 'plain':
        if selector['orbital'] is None:
            return selector['symbol']
        return f"{selector['symbol']} {selector['orbital']}"

    if selector['kind'] == 'atom':
        label = f"atom {selector['atom_index']}"
        if selector['orbital'] is not None:
            label = f"{label} {selector['orbital']}"
        return label

    parts = [f"@{selector['axis']} {selector['coord']:.1f}"]
    if selector['symbol'] is not None:
        parts.append(selector['symbol'])
    if selector['orbital'] is not None:
        parts.append(selector['orbital'])
    return ' '.join(parts)


def get_position_matched_atom_indices(items, selector, symbol_key):
    axis = AXIS_INDEX[selector['axis']]
    atom_indices = []

    for one in items:
        if selector['symbol'] is not None and one[symbol_key] != selector['symbol']:
            continue

        if abs(float(one['position'][axis]) - selector['coord']) > POSITION_TOL:
            continue

        atom_indices.append(int(one['atom_index']))

    return sorted(set(atom_indices))


def get_selector_groups(items, selectors, symbol_key):
    symbols = []
    for one in items:
        if one[symbol_key] not in symbols:
            symbols.append(one[symbol_key])

    groups = OrderedDict()

    if not selectors:
        for symbol in symbols:
            indices = [idx for idx, one in enumerate(items) if one[symbol_key] == symbol]
            groups[symbol] = np.array(indices, dtype=int)
        return groups

    for selector in selectors:
        if selector['kind'] == 'plain':
            symbol = selector['symbol']
            orbital = selector['orbital']
            if symbol not in symbols:
                raise ValueError(f'cannot find species [{symbol}]')

            label = get_selector_label(selector)
            indices = []
            for idx, one in enumerate(items):
                if one[symbol_key] != symbol:
                    continue
                if orbital is None or orbital_matches(one, orbital):
                    indices.append(idx)

            if not indices:
                raise ValueError(f'cannot find orbital [{label}]')

        elif selector['kind'] == 'atom':
            label = get_selector_label(selector)
            atom_index = int(selector['atom_index'])
            orbital = selector['orbital']
            atom_exists = False
            indices = []
            for idx, one in enumerate(items):
                if int(one['atom_index']) != atom_index:
                    continue
                atom_exists = True
                if orbital is None or orbital_matches(one, orbital):
                    indices.append(idx)

            if not atom_exists:
                raise ValueError(f'cannot find atom [{atom_index}]')
            if not indices:
                raise ValueError(f'cannot find orbital [{label}]')

        elif selector['kind'] == 'axis':
            label = get_selector_label(selector)
            atom_indices = get_position_matched_atom_indices(items, selector, symbol_key)
            if not atom_indices:
                if selector['symbol'] is None:
                    raise ValueError(
                        f"cannot find atoms near {selector['axis']}={selector['coord']:.1f}"
                    )
                raise ValueError(
                    f"cannot find [{selector['symbol']}] atoms near "
                    f"{selector['axis']}={selector['coord']:.1f}"
                )

            indices = []
            for idx, one in enumerate(items):
                if int(one['atom_index']) not in atom_indices:
                    continue
                if selector['orbital'] is None or orbital_matches(one, selector['orbital']):
                    indices.append(idx)

            if not indices:
                raise ValueError(f'cannot find orbital [{label}]')

        else:
            raise ValueError(f"invalid selector kind [{selector['kind']}]")

        if label not in groups:
            groups[label] = []
        groups[label].extend(indices)

    for label in groups:
        groups[label] = np.array(sorted(set(groups[label])), dtype=int)

    return groups


def get_color_map(labels):
    color_map = {}
    for index, label in enumerate(labels):
        color_map[label] = COLORS[index % len(COLORS)]
    return color_map


def get_case_dirs(target_dir):
    target_dir = Path(target_dir).resolve()
    if not target_dir.is_dir():
        raise ValueError(f'cannot find root directory [{target_dir}]')

    if (target_dir / 'output').is_dir() or (target_dir / 'input').is_dir():
        return [target_dir], True

    case_dirs = sorted(path for path in target_dir.iterdir() if path.is_dir())
    return case_dirs, False


def parse_plot_arguments(arguments, script_name):
    if len(arguments) < 2:
        print(
            f'useage : python {script_name} [root dir] '
            '[element] [orbital] ... [atom index [orbital]] ... '
            '[x/y/z coord [element] [orbital]] ...'
        )
        raise SystemExit(1)

    target_dir = Path(arguments[1]).resolve()
    try:
        case_dirs, is_single_case = get_case_dirs(target_dir)
    except ValueError as exc:
        print(f'error : {exc}')
        raise SystemExit(1)

    selectors = []
    tokens = arguments[2:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if is_atom_selector_token(token):
            if index + 1 >= len(tokens):
                print(f'error : missing atom index after [{token}]')
                raise SystemExit(1)

            try:
                atom_index = int(tokens[index + 1])
            except ValueError:
                print(f'error : invalid atom index [{tokens[index + 1]}]')
                raise SystemExit(1)
            if atom_index < 1:
                print(f'error : atom index should be >= 1')
                raise SystemExit(1)

            index += 2
            orbital = None
            if index < len(tokens):
                next_token = tokens[index]
                next_orbital = next_token.lower()
                if ORBITAL_PATTERN.fullmatch(next_orbital) is not None and next_token == next_token.lower():
                    orbital = next_orbital
                    index += 1

            selectors.append({'kind': 'atom', 'atom_index': atom_index, 'orbital': orbital})
            continue

        if is_position_selector_token(token):
            axis = token.lower()
            if index + 1 >= len(tokens):
                print(f'error : missing coordinate after [{token}]')
                raise SystemExit(1)

            try:
                coord = float(tokens[index + 1])
            except ValueError:
                print(f'error : invalid coordinate [{tokens[index + 1]}]')
                raise SystemExit(1)

            index += 2
            element = None
            orbital = None

            if index < len(tokens):
                next_token = tokens[index]
                next_orbital = next_token.lower()
                if ORBITAL_PATTERN.fullmatch(next_orbital) and next_token == next_token.lower():
                    orbital = next_orbital
                    index += 1
                elif is_species_token(next_token):
                    element = next_token
                    index += 1

                    if index < len(tokens) and not is_position_selector_token(tokens[index]):
                        orbital = tokens[index].lower()
                        if ORBITAL_PATTERN.fullmatch(orbital) is None:
                            print(f'error : invalid orbital token [{tokens[index]}]')
                            raise SystemExit(1)
                        index += 1
                elif not is_position_selector_token(next_token):
                    print(f'error : invalid species token [{next_token}]')
                    raise SystemExit(1)

            selectors.append(
                {
                    'kind': 'axis',
                    'axis': axis,
                    'coord': coord,
                    'symbol': element,
                    'orbital': orbital,
                }
            )
            continue

        if not is_species_token(token):
            print(f'error : invalid species token [{token}]')
            raise SystemExit(1)

        element = token
        orbital = None
        if index + 1 < len(tokens) and not is_position_selector_token(tokens[index + 1]):
            next_token = tokens[index + 1]
            next_orbital = next_token.lower()
            if ORBITAL_PATTERN.fullmatch(next_orbital) is not None and next_token == next_token.lower():
                orbital = next_orbital
                index += 1

        selectors.append({'kind': 'plain', 'symbol': element, 'orbital': orbital})
        index += 1

    return target_dir, case_dirs, selectors, is_single_case
