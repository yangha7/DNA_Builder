"""
Core DNA structure builder.

Uses nucleotide templates extracted from Avogadro-generated structures.
Each template contains the complete nucleotide (base + sugar + phosphate)
with correct atom positions and bond connectivity.

The building algorithm:
1. For each base pair position i, apply helical screw (twist*i, rise*i)
   to the template coordinates
2. Strand I atoms are output first, then strand II atoms
3. Templates are already in the helix frame at position 0
4. 5' terminal O atoms are generated to complete the phosphate group

Charge model:
  Each nucleotide has one phosphate group with formal charge -1.
  The 5' terminal phosphate gets an extra O (O5T) to complete the
  PO4 group, ensuring -1 charge per nucleotide systematically.
  Total charge = -(number of nucleotides) = -(2 * sequence_length)
"""

import numpy as np
import warnings
from typing import List, Optional
from .fiber_data import (
    HELICAL_PARAMS, WC_COMPLEMENT, RESIDUE_NAMES,
    PURINE_BASES, PYRIMIDINE_BASES,
    B_STRAND1, B_STRAND2,
    A_STRAND1, A_STRAND2,
    Z_STRAND1_POS1, Z_STRAND1_POS2,
    Z_STRAND2_POS1, Z_STRAND2_POS2,
)


class Atom:
    """Represents a single atom in the structure."""
    __slots__ = ("name", "element", "x", "y", "z",
                 "residue_name", "residue_seq", "chain_id")

    def __init__(self, name: str, element: str, x: float, y: float, z: float,
                 residue_name: str = "", residue_seq: int = 1,
                 chain_id: str = "A"):
        self.name = name
        self.element = element
        self.x = x
        self.y = y
        self.z = z
        self.residue_name = residue_name
        self.residue_seq = residue_seq
        self.chain_id = chain_id


def _rot_z(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Z axis."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _helical_screw(coords: np.ndarray, rise: float, twist_deg: float,
                   step: int) -> np.ndarray:
    """Apply helical screw: rotate by twist*step around Z, translate by rise*step along Z."""
    R = _rot_z(twist_deg * step)
    translation = np.array([0.0, 0.0, rise * step])
    return (R @ coords.T).T + translation


def _template_to_atoms(template, coords, res_name, res_seq, chain_id,
                       skip_extra=False) -> List[Atom]:
    """Convert template + transformed coordinates to Atom list."""
    atoms = []
    for i, (atom_name, element, _, _, _) in enumerate(template):
        if skip_extra and atom_name in ("OXT", "HTER"):
            continue
        atoms.append(Atom(
            name=atom_name, element=element,
            x=coords[i, 0], y=coords[i, 1], z=coords[i, 2],
            residue_name=res_name, residue_seq=res_seq, chain_id=chain_id,
        ))
    return atoms


def _find_atom_coords(atom_list: List[Atom], name: str) -> Optional[np.ndarray]:
    """Find coordinates of a named atom in a list."""
    for a in atom_list:
        if a.name == name:
            return np.array([a.x, a.y, a.z])
    return None


def _generate_5prime_terminal_O(atom_list: List[Atom],
                                 res_name: str, res_seq: int,
                                 chain_id: str) -> Optional[Atom]:
    """
    Generate the 5' terminal hydroxyl oxygen.

    Positioned along the P -> O5' bond direction, on the opposite side
    of P from O5'. This completes the PO4 group at the 5' end.

    The O5T is placed at ~1.48 Å from P (standard P-O bond length),
    in the direction opposite to O5' from P.
    """
    p_coord = _find_atom_coords(atom_list, "P")
    o5_coord = _find_atom_coords(atom_list, "O5'")

    if p_coord is None or o5_coord is None:
        return None

    # Direction from P to O5'
    direction = o5_coord - p_coord
    d = np.linalg.norm(direction)
    if d < 0.01:
        return None

    # Place O5T on the opposite side of P from O5', at P-O bond length
    p_o_bond = 1.48  # Å, standard P-O bond length
    o5t_coord = p_coord - (direction / d) * p_o_bond

    return Atom(
        name="O5T", element="O",
        x=o5t_coord[0], y=o5t_coord[1], z=o5t_coord[2],
        residue_name=res_name, residue_seq=res_seq, chain_id=chain_id,
    )


def build_b_dna(sequence: str) -> List[Atom]:
    """
    Build B-form double-stranded DNA.

    All nucleotides include the full phosphate group (P, O1P, O2P).
    5' terminal residues get an extra O (O5T) to complete the PO4 group,
    ensuring -1 formal charge per nucleotide.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    params = HELICAL_PARAMS["B"]
    rise = params["rise"]
    twist = params["twist"]
    n_bp = len(sequence)

    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # --- Strand I (5' -> 3'), all residues ---
    for i in range(n_bp):
        base_s1 = sequence[i]
        template_s1 = B_STRAND1[base_s1]
        coords_s1 = np.array([[a[2], a[3], a[4]] for a in template_s1])
        transformed_s1 = _helical_screw(coords_s1, rise, twist, i)

        res_name_s1 = RESIDUE_NAMES[base_s1]
        nuc_atoms = _template_to_atoms(
            template_s1, transformed_s1, res_name_s1, i + 1, "A",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for first residue
        if i == 0:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s1, i + 1, "A")
            if o5t:
                terminal_atoms.append(o5t)

    # --- Strand II (3' -> 5', antiparallel), all residues ---
    for i in range(n_bp):
        base_s2 = comp_sequence[i]
        template_s2 = B_STRAND2[base_s2]
        coords_s2 = np.array([[a[2], a[3], a[4]] for a in template_s2])
        transformed_s2 = _helical_screw(coords_s2, rise, twist, i)

        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        nuc_atoms = _template_to_atoms(
            template_s2, transformed_s2, res_name_s2, res_seq_s2, "B",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for strand II's 5' end
        if i == n_bp - 1:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s2, res_seq_s2, "B")
            if o5t:
                terminal_atoms.append(o5t)

    # Append terminal O atoms at the end (matching 3DNA convention)
    all_atoms.extend(terminal_atoms)

    return all_atoms


def build_a_dna(sequence: str) -> List[Atom]:
    """
    Build A-form double-stranded DNA.

    Uses nucleotide templates extracted from 3DNA fiber structures
    (Colin's A-form XYZ files). Templates were extracted by:
    1. Fitting the helical screw axis on ATATAT (RMSD 0.0009 Å)
    2. Transforming to helix frame (axis along Z)
    3. Unwinding each nucleotide to position 0
    4. G/C base atoms obtained via Kabsch superposition onto A/T backbone

    A-DNA parameters: rise = 2.548 Å, twist = 32.727° (11 bp/turn)

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    params = HELICAL_PARAMS["A"]
    rise = params["rise"]
    twist = params["twist"]
    n_bp = len(sequence)

    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # --- Strand I (5' -> 3'), all residues ---
    for i in range(n_bp):
        base_s1 = sequence[i]
        template_s1 = A_STRAND1[base_s1]
        coords_s1 = np.array([[a[2], a[3], a[4]] for a in template_s1])
        transformed_s1 = _helical_screw(coords_s1, rise, twist, i)

        res_name_s1 = RESIDUE_NAMES[base_s1]
        nuc_atoms = _template_to_atoms(
            template_s1, transformed_s1, res_name_s1, i + 1, "A",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for first residue
        if i == 0:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s1, i + 1, "A")
            if o5t:
                terminal_atoms.append(o5t)

    # --- Strand II (3' -> 5', antiparallel), all residues ---
    for i in range(n_bp):
        base_s2 = comp_sequence[i]
        template_s2 = A_STRAND2[base_s2]
        coords_s2 = np.array([[a[2], a[3], a[4]] for a in template_s2])
        transformed_s2 = _helical_screw(coords_s2, rise, twist, i)

        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        nuc_atoms = _template_to_atoms(
            template_s2, transformed_s2, res_name_s2, res_seq_s2, "B",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for strand II's 5' end
        if i == n_bp - 1:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s2, res_seq_s2, "B")
            if o5t:
                terminal_atoms.append(o5t)

    # Append terminal O atoms at the end
    all_atoms.extend(terminal_atoms)

    return all_atoms



def build_z_dna(sequence: str) -> List[Atom]:
    """
    Build Z-form double-stranded DNA.

    Z-DNA is a left-handed helix with a dinucleotide repeat unit.
    The canonical Z-DNA sequence alternates purine-pyrimidine (e.g., GCGCGC).

    The building algorithm uses a dinucleotide repeat:
    - Even positions (0,2,4...): purine in syn conformation (POS1 templates)
    - Odd positions (1,3,5...): pyrimidine in anti conformation (POS2 templates)
    - The helical screw operates on dinucleotide pairs:
      dinuc_index = position // 2
      cumulative_twist = dinuc_index * twist_dinuc
      cumulative_rise = dinuc_index * rise_dinuc

    Both nucleotides within a dinucleotide share the same screw transformation,
    preserving their relative geometry from the template.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').
        Must have even length. Canonical Z-DNA alternates pur-pyr.

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")
    if len(sequence) % 2 != 0:
        raise ValueError("Z-DNA sequence must have even length (dinucleotide repeat)")

    _comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_comp = "".join(_comp[b] for b in reversed(sequence))
    if rev_comp != sequence:
        raise ValueError(
            f"Z-DNA sequence '{sequence}' is not palindromic "
            f"(reverse complement is '{rev_comp}'). "
            f"Use a palindromic sequence such as GCGCGCGC."
        )

    # Check for canonical alternating purine-pyrimidine pattern
    is_canonical = all(
        (sequence[i] in PURINE_BASES) == (i % 2 == 0)
        for i in range(len(sequence))
    )
    if not is_canonical:
        warnings.warn(
            f"Z-DNA sequence '{sequence}' does not alternate purine-pyrimidine. "
            "Non-canonical Z-DNA may have distorted backbone geometry.",
            stacklevel=2,
        )

    params = HELICAL_PARAMS["Z"]
    rise_dinuc = params["rise_dinuc"]
    twist_dinuc = params["twist_dinuc"]
    n_bp = len(sequence)

    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # --- Strand I (5' -> 3'), all residues ---
    for i in range(n_bp):
        base_s1 = sequence[i]
        is_pos2 = i % 2
        dinuc_idx = i // 2

        # Select template based on position within dinucleotide
        if is_pos2:
            template_s1 = Z_STRAND1_POS2[base_s1]
        else:
            template_s1 = Z_STRAND1_POS1[base_s1]

        coords_s1 = np.array([[a[2], a[3], a[4]] for a in template_s1])
        # Apply dinucleotide screw (same for both positions in a dinucleotide)
        cum_twist = dinuc_idx * twist_dinuc
        cum_rise = dinuc_idx * rise_dinuc
        transformed_s1 = _helical_screw(coords_s1, cum_rise, cum_twist, 1)

        res_name_s1 = RESIDUE_NAMES[base_s1]
        nuc_atoms = _template_to_atoms(
            template_s1, transformed_s1, res_name_s1, i + 1, "A",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for first residue
        if i == 0:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s1, i + 1, "A")
            if o5t:
                terminal_atoms.append(o5t)

    # --- Strand II (3' -> 5', antiparallel), all residues ---
    for i in range(n_bp):
        base_s2 = comp_sequence[i]
        is_pos2 = i % 2
        dinuc_idx = i // 2

        # Select template based on position within dinucleotide
        if is_pos2:
            template_s2 = Z_STRAND2_POS2[base_s2]
        else:
            template_s2 = Z_STRAND2_POS1[base_s2]

        coords_s2 = np.array([[a[2], a[3], a[4]] for a in template_s2])
        cum_twist = dinuc_idx * twist_dinuc
        cum_rise = dinuc_idx * rise_dinuc
        transformed_s2 = _helical_screw(coords_s2, cum_rise, cum_twist, 1)

        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        nuc_atoms = _template_to_atoms(
            template_s2, transformed_s2, res_name_s2, res_seq_s2, "B",
            skip_extra=True)
        all_atoms.extend(nuc_atoms)

        # Generate 5' terminal O for strand II's 5' end
        if i == n_bp - 1:
            o5t = _generate_5prime_terminal_O(
                nuc_atoms, res_name_s2, res_seq_s2, "B")
            if o5t:
                terminal_atoms.append(o5t)

    # Append terminal O atoms at the end (matching 3DNA convention)
    all_atoms.extend(terminal_atoms)

    return all_atoms

def build_dna(sequence: str, form: str = "B", method: str = "template") -> List[Atom]:
    """
    Build double-stranded DNA in the specified form.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').
    form : str
        DNA form: "A", "B", or "Z".
    method : str
        Building method: "template" (v1, default) or "zmatrix" (v2).
        - template: Uses pre-extracted nucleotide templates with helical screw.
        - zmatrix: Builds backbone atom-by-atom using internal coordinates.

    Returns
    -------
    List[Atom]
        Complete list of atoms for the double-stranded DNA.
    """
    form = form.upper().strip()
    method = method.lower().strip()

    if form not in ("A", "B", "Z"):
        raise ValueError(f"Unknown DNA form: {form}. Must be A, B, or Z.")

    if method == "zmatrix":
        from .zmatrix_builder import build_dna_v2
        return build_dna_v2(sequence, form)

    # Default: template method (v1)
    if form == "B":
        return build_b_dna(sequence)
    elif form == "A":
        return build_a_dna(sequence)
    else:
        return build_z_dna(sequence)
