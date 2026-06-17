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
"""

import numpy as np
from typing import List
from .fiber_data import (
    HELICAL_PARAMS, WC_COMPLEMENT, RESIDUE_NAMES,
    PURINE_BASES, PYRIMIDINE_BASES,
    B_STRAND1, B_STRAND2,
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
                       skip_phosphate=False, skip_extra=False) -> List[Atom]:
    """Convert template + transformed coordinates to Atom list."""
    atoms = []
    for i, (atom_name, element, _, _, _) in enumerate(template):
        if skip_phosphate and atom_name in ("P", "O1P", "O2P", "OP1", "OP2"):
            continue
        if skip_extra and atom_name in ("OXT", "HTER"):
            continue
        atoms.append(Atom(
            name=atom_name, element=element,
            x=coords[i, 0], y=coords[i, 1], z=coords[i, 2],
            residue_name=res_name, residue_seq=res_seq, chain_id=chain_id,
        ))
    return atoms


def build_b_dna(sequence: str) -> List[Atom]:
    """
    Build B-form double-stranded DNA.

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

    # --- Strand I (5' -> 3'), all residues first ---
    for i in range(n_bp):
        base_s1 = sequence[i]
        template_s1 = B_STRAND1[base_s1]
        coords_s1 = np.array([[a[2], a[3], a[4]] for a in template_s1])
        transformed_s1 = _helical_screw(coords_s1, rise, twist, i)

        res_name_s1 = RESIDUE_NAMES[base_s1]
        skip_p_s1 = (i == 0)
        all_atoms.extend(_template_to_atoms(
            template_s1, transformed_s1, res_name_s1, i + 1, "A",
            skip_phosphate=skip_p_s1, skip_extra=True))

    # --- Strand II (3' -> 5', antiparallel), all residues ---
    for i in range(n_bp):
        base_s2 = comp_sequence[i]
        template_s2 = B_STRAND2[base_s2]
        coords_s2 = np.array([[a[2], a[3], a[4]] for a in template_s2])
        transformed_s2 = _helical_screw(coords_s2, rise, twist, i)

        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        skip_p_s2 = (i == n_bp - 1)
        all_atoms.extend(_template_to_atoms(
            template_s2, transformed_s2, res_name_s2, res_seq_s2, "B",
            skip_phosphate=skip_p_s2, skip_extra=True))

    return all_atoms


def build_dna(sequence: str, form: str = "B") -> List[Atom]:
    """
    Build double-stranded DNA in the specified form.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').
    form : str
        DNA form: "A", "B", or "Z".

    Returns
    -------
    List[Atom]
        Complete list of atoms for the double-stranded DNA.
    """
    form = form.upper().strip()
    if form not in ("A", "B", "Z"):
        raise ValueError(f"Unknown DNA form: {form}. Must be A, B, or Z.")

    if form == "B":
        return build_b_dna(sequence)
    elif form == "A":
        raise NotImplementedError(
            "A-DNA builder requires Avogadro A-DNA templates. "
            "Please generate an A-DNA structure in Avogadro and run extract_from_avogadro.py.")
    else:
        raise NotImplementedError(
            "Z-DNA builder requires Avogadro Z-DNA templates. "
            "Please generate a Z-DNA structure in Avogadro and run extract_from_avogadro.py.")
