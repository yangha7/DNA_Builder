"""
Parsers for reading DNA structures from PDB, mmCIF, and XYZ files.

Each parser returns a list of Atom objects (from builder.py) that can be
used for RMSD comparison with reference models.

Supported formats:
  - PDB: Standard ATOM records with chain/residue/atom name info
  - mmCIF: AlphaFold-style _atom_site loop with label_atom_id
  - XYZ: Element + coordinates only; atom names inferred by proximity
"""

import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .builder import Atom
from .fiber_data import RESIDUE_NAMES, WC_COMPLEMENT


# Reverse lookup: residue name -> single-letter base
_RESNAME_TO_BASE = {v: k for k, v in RESIDUE_NAMES.items()}

# Atom name normalization: maps alternative naming conventions to our standard
# Our reference models use: O1P, O2P, O5', C5', C4', O4', C3', O3', C2', C1'
# CIF/PDB v3 may use: OP1, OP2, same sugar names
_ATOM_NAME_ALIASES = {
    "OP1": "O1P",
    "OP2": "O2P",
    "OP3": "O5T",  # 5' terminal oxygen
}


def normalize_atom_name(name: str) -> str:
    """
    Normalize atom name to match our reference model convention.

    Handles:
    - OP1 -> O1P, OP2 -> O2P (phosphate oxygen naming)
    - OP3 -> O5T (5' terminal oxygen)
    """
    return _ATOM_NAME_ALIASES.get(name, name)


def detect_format(filepath: str) -> str:
    """
    Detect file format from extension.

    Returns one of: 'pdb', 'cif', 'xyz'
    """
    ext = Path(filepath).suffix.lower()
    if ext in ('.cif', '.mmcif'):
        return 'cif'
    elif ext == '.xyz':
        return 'xyz'
    elif ext in ('.pdb', '.ent'):
        return 'pdb'
    else:
        raise ValueError(f"Unknown file extension: {ext}. "
                         "Supported: .pdb, .cif, .mmcif, .xyz")


def parse_structure(filepath: str) -> List[Atom]:
    """
    Parse a structure file and return a list of Atom objects.

    Auto-detects format from file extension.
    """
    fmt = detect_format(filepath)
    if fmt == 'pdb':
        return parse_pdb(filepath)
    elif fmt == 'cif':
        return parse_cif(filepath)
    elif fmt == 'xyz':
        return parse_xyz(filepath)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def parse_pdb(filepath: str) -> List[Atom]:
    """
    Parse a PDB file and return Atom objects.

    Reads standard ATOM records (columns per PDB format specification):
      - Atom name: columns 13-16
      - Residue name: columns 18-20
      - Chain ID: column 22
      - Residue seq: columns 23-26
      - X, Y, Z: columns 31-54
      - Element: columns 77-78
    """
    atoms: List[Atom] = []

    with open(filepath, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip() or 'A'
            try:
                residue_seq = int(line[22:26].strip())
            except ValueError:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            # Element from columns 77-78, fallback to first letter of atom name
            element = ''
            if len(line) >= 78:
                element = line[76:78].strip()
            if not element:
                element = atom_name[0] if atom_name else 'X'

            # Normalize atom name
            atom_name = normalize_atom_name(atom_name)

            atoms.append(Atom(
                name=atom_name,
                element=element,
                x=x, y=y, z=z,
                residue_name=residue_name,
                residue_seq=residue_seq,
                chain_id=chain_id,
            ))

    return atoms


def parse_cif(filepath: str) -> List[Atom]:
    """
    Parse an mmCIF file and return Atom objects.

    Handles AlphaFold-style output with _atom_site loop containing:
      - group_PDB (ATOM/HETATM)
      - type_symbol (element)
      - label_atom_id (atom name)
      - label_comp_id (residue name)
      - label_asym_id or auth_asym_id (chain ID)
      - label_seq_id or auth_seq_id (residue number)
      - Cartn_x, Cartn_y, Cartn_z (coordinates)
    """
    atoms: List[Atom] = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Find the _atom_site loop
    # Parse column names and data rows
    lines = content.split('\n')
    in_atom_site = False
    column_names: List[str] = []
    data_rows: List[List[str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect start of _atom_site loop
        if line == 'loop_':
            # Check if next lines are _atom_site columns
            j = i + 1
            temp_cols = []
            while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                temp_cols.append(lines[j].strip())
                j += 1
            if temp_cols:
                column_names = temp_cols
                in_atom_site = True
                i = j
                continue

        if in_atom_site:
            if not line or line.startswith('loop_') or line.startswith('#') or line.startswith('_'):
                if data_rows:
                    break  # End of atom_site data
                i += 1
                continue

            # Parse data row — handle quoted strings
            tokens = _tokenize_cif_line(line)
            if tokens:
                data_rows.append(tokens)

        i += 1

    if not column_names or not data_rows:
        raise ValueError(f"No _atom_site data found in {filepath}")

    # Build column index map
    col_idx: Dict[str, int] = {}
    for idx, name in enumerate(column_names):
        # Strip the _atom_site. prefix
        short = name.replace('_atom_site.', '')
        col_idx[short] = idx

    # Required columns
    def _get(row: List[str], key: str, default: str = '') -> str:
        idx = col_idx.get(key)
        if idx is not None and idx < len(row):
            val = row[idx]
            return val if val != '?' and val != '.' else default
        return default

    for row in data_rows:
        group = _get(row, 'group_PDB', 'ATOM')
        if group not in ('ATOM', 'HETATM'):
            continue

        element = _get(row, 'type_symbol', 'X')
        atom_name = _get(row, 'label_atom_id', '')
        residue_name = _get(row, 'label_comp_id', '')

        # Prefer auth_asym_id for chain (matches PDB convention)
        chain_id = _get(row, 'auth_asym_id') or _get(row, 'label_asym_id', 'A')

        # Prefer auth_seq_id for residue number
        seq_str = _get(row, 'auth_seq_id') or _get(row, 'label_seq_id', '1')
        try:
            residue_seq = int(seq_str)
        except ValueError:
            continue

        try:
            x = float(_get(row, 'Cartn_x', '0'))
            y = float(_get(row, 'Cartn_y', '0'))
            z = float(_get(row, 'Cartn_z', '0'))
        except ValueError:
            continue

        # The tokenizer already handles quoted strings in CIF, so atom_name
        # should be clean (e.g., "O5'" -> O5'). Only strip double quotes
        # if they somehow remain, but preserve trailing primes (') which are
        # part of sugar atom names (C1', C2', O3', O4', O5', C3', C4', C5').
        if atom_name.startswith('"') and atom_name.endswith('"'):
            atom_name = atom_name[1:-1]

        # Normalize atom name to match our reference convention
        atom_name = normalize_atom_name(atom_name)

        atoms.append(Atom(
            name=atom_name,
            element=element,
            x=x, y=y, z=z,
            residue_name=residue_name,
            residue_seq=residue_seq,
            chain_id=chain_id,
        ))

    return atoms


def _tokenize_cif_line(line: str) -> List[str]:
    """
    Tokenize a CIF data line, handling quoted strings.

    Double-quoted strings like "O5'" are extracted with the quotes removed
    but internal content preserved (including prime characters).

    Examples:
        'ATOM 1 C "C1\\''" -> ['ATOM', '1', 'C', "C1'"]
    """
    tokens = []
    i = 0
    n = len(line)

    while i < n:
        # Skip whitespace
        while i < n and line[i] in (' ', '\t'):
            i += 1
        if i >= n:
            break

        # Double-quoted string
        if line[i] == '"':
            i += 1
            start = i
            while i < n and line[i] != '"':
                i += 1
            tokens.append(line[start:i])
            if i < n:
                i += 1  # skip closing quote
        # Single-quoted string (but be careful: single quote at end of
        # an unquoted token like O5' is NOT a quote delimiter)
        elif line[i] == "'" and (i == 0 or line[i-1] in (' ', '\t')):
            # Only treat as quoted if it looks like a proper quoted string
            # Find the closing quote (must be followed by space or end)
            j = i + 1
            found_close = False
            while j < n:
                if line[j] == "'" and (j + 1 >= n or line[j+1] in (' ', '\t')):
                    found_close = True
                    break
                j += 1
            if found_close and j > i + 1:
                tokens.append(line[i+1:j])
                i = j + 1
            else:
                # Not a proper quoted string, treat as unquoted
                start = i
                while i < n and line[i] not in (' ', '\t'):
                    i += 1
                tokens.append(line[start:i])
        else:
            # Unquoted token
            start = i
            while i < n and line[i] not in (' ', '\t'):
                i += 1
            tokens.append(line[start:i])

    return tokens


def parse_xyz(filepath: str) -> List[Atom]:
    """
    Parse an XYZ file and return Atom objects.

    XYZ format has no atom names or residue info — only element + coordinates.
    Atoms are assigned generic names and residue info based on position.

    Format:
        <num_atoms>
        <comment line>
        <element> <x> <y> <z>
        ...
    """
    atoms: List[Atom] = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"XYZ file too short: {filepath}")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"Invalid atom count in XYZ file: {lines[0].strip()}")

    # Parse atom lines (skip header 2 lines)
    for i, line in enumerate(lines[2:2 + num_atoms]):
        parts = line.split()
        if len(parts) < 4:
            continue

        element = parts[0].strip()
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue

        # For XYZ files, we don't have atom names or residue info
        # These will be matched by proximity in the classifier
        atoms.append(Atom(
            name=element + str(i + 1),  # Generic name
            element=element,
            x=x, y=y, z=z,
            residue_name='UNK',
            residue_seq=1,
            chain_id='A',
        ))

    return atoms


def extract_sequence_from_atoms(atoms: List[Atom]) -> Tuple[str, List[str]]:
    """
    Extract the DNA sequence from parsed atoms.

    Returns (sequence, chain_ids) where sequence is the strand A sequence
    in 5'->3' order and chain_ids are the unique chain identifiers found.

    For CIF/PDB files, uses residue names (DA, DT, DG, DC) to determine bases.
    """
    chain_ids = sorted(set(a.chain_id for a in atoms))

    # Use the first chain (typically 'A') for the sequence
    if not chain_ids:
        return '', []

    primary_chain = chain_ids[0]

    # Collect unique residues on the primary chain, ordered by seq number
    residues: Dict[int, str] = {}
    for atom in atoms:
        if atom.chain_id != primary_chain:
            continue
        if atom.residue_seq not in residues:
            residues[atom.residue_seq] = atom.residue_name

    # Convert residue names to single-letter bases
    sequence = ''
    for seq_num in sorted(residues.keys()):
        res_name = residues[seq_num]
        base = _RESNAME_TO_BASE.get(res_name, '')
        if base:
            sequence += base
        else:
            # Try stripping 'D' prefix for non-standard naming
            if res_name.startswith('D') and len(res_name) == 2:
                base_letter = res_name[1]
                if base_letter in 'ATGC':
                    sequence += base_letter
                    continue
            # Unknown residue
            sequence += '?'

    return sequence, chain_ids
