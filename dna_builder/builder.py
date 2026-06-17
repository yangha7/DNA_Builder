"""
Core DNA structure builder.

Constructs double-stranded DNA using the base pair reference frame approach:

1. Build a Watson-Crick base pair at the origin
   - Strand I base in the +y half-plane
   - Strand II (complement) base in the -y half-plane (related by pseudo-dyad)
   - Backbone atoms attached to each base

2. Position the base pair relative to the helix axis
   - Apply x-displacement (shift perpendicular to helix axis)
   - Apply inclination (tilt of base pair plane)

3. Apply helical screw operation for each successive base pair
   - Rotate by twist angle around Z (helix axis)
   - Translate by rise along Z

The pseudo-dyad operation for Watson-Crick base pairs:
   (x, y, z) -> (-x, -y, -z)
This maps strand I to strand II within a base pair.
"""

import numpy as np
from typing import List, Tuple
from .fiber_data import (
    HELICAL_PARAMS, WC_COMPLEMENT, RESIDUE_NAMES,
    PURINE_BASES, PYRIMIDINE_BASES,
    BASE_COORDS,
    BACKBONE_C2ENDO_ANTI, BACKBONE_C3ENDO_ANTI, BACKBONE_C3ENDO_SYN,
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


def _rot_y(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(angle_deg: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _build_nucleotide(base_type: str, backbone_template: list,
                      is_strand2: bool = False,
                      propeller: float = 0.0) -> list:
    """
    Build a single nucleotide (base + backbone) in the base pair frame.

    Parameters
    ----------
    base_type : str
        One of A, T, G, C.
    backbone_template : list
        Backbone atom coordinates [(name, element, x, y, z), ...].
    is_strand2 : bool
        If True, apply the pseudo-dyad to place on strand II.
    propeller : float
        Propeller twist angle in degrees.

    Returns
    -------
    list of (name, element, x, y, z) tuples
    """
    base_atoms = BASE_COORDS[base_type]

    # Combine base and backbone atoms
    all_atoms = []

    # Apply propeller twist: rotate base around x-axis
    # Strand I gets +propeller/2, strand II gets -propeller/2
    prop_angle = propeller / 2.0 if not is_strand2 else -propeller / 2.0
    R_prop = _rot_x(prop_angle)

    # Base atoms
    for (name, elem, bx, by, bz) in base_atoms:
        coord = R_prop @ np.array([bx, by, bz])
        all_atoms.append((name, elem, coord[0], coord[1], coord[2]))

    # Backbone atoms (positioned relative to glycosidic N)
    for (name, elem, bx, by, bz) in backbone_template:
        coord = R_prop @ np.array([bx, by, bz])
        all_atoms.append((name, elem, coord[0], coord[1], coord[2]))

    if is_strand2:
        # Apply pseudo-dyad: (x, y, z) -> (-x, -y, -z)
        all_atoms = [(n, e, -x, -y, -z) for (n, e, x, y, z) in all_atoms]

    return all_atoms


def _position_bp_on_helix(atoms_s1: list, atoms_s2: list,
                          x_disp: float, y_disp: float,
                          inclination: float, tip: float,
                          rise: float, twist: float,
                          step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Position a base pair on the helix.

    1. Apply inclination (rotation around y-axis of bp frame)
    2. Apply x-displacement and y-displacement
    3. Apply helical screw (twist * step around Z, rise * step along Z)

    Returns transformed coordinate arrays for both strands.
    """
    coords_s1 = np.array([[x, y, z] for (_, _, x, y, z) in atoms_s1])
    coords_s2 = np.array([[x, y, z] for (_, _, x, y, z) in atoms_s2])

    # 1. Apply inclination (tilt bp plane around y-axis)
    if abs(inclination) > 0.01:
        R_inc = _rot_y(inclination)
        coords_s1 = (R_inc @ coords_s1.T).T
        coords_s2 = (R_inc @ coords_s2.T).T

    # 2. Apply tip (tilt around x-axis)
    if abs(tip) > 0.01:
        R_tip = _rot_x(tip)
        coords_s1 = (R_tip @ coords_s1.T).T
        coords_s2 = (R_tip @ coords_s2.T).T

    # 3. Apply displacement from helix axis
    displacement = np.array([x_disp, y_disp, 0.0])
    coords_s1 = coords_s1 + displacement
    coords_s2 = coords_s2 + displacement

    # 4. Apply helical screw for this step
    angle = twist * step
    R_twist = _rot_z(angle)
    z_trans = np.array([0.0, 0.0, rise * step])

    coords_s1 = (R_twist @ coords_s1.T).T + z_trans
    coords_s2 = (R_twist @ coords_s2.T).T + z_trans

    return coords_s1, coords_s2


def _atoms_to_list(template, coords, res_name, res_seq, chain_id,
                   skip_phosphate=False) -> List[Atom]:
    """Convert template names + transformed coords to Atom objects."""
    atoms = []
    for i, (name, elem, _, _, _) in enumerate(template):
        if skip_phosphate and name in ("P", "OP1", "OP2"):
            continue
        atoms.append(Atom(
            name=name, element=elem,
            x=coords[i, 0], y=coords[i, 1], z=coords[i, 2],
            residue_name=res_name, residue_seq=res_seq, chain_id=chain_id,
        ))
    return atoms


def build_ab_dna(sequence: str, form: str = "B") -> List[Atom]:
    """
    Build A-form or B-form double-stranded DNA.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').
    form : str
        "A" or "B".

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    params = HELICAL_PARAMS[form]
    rise = params["rise"]
    twist = params["twist"]
    x_disp = params["x_disp"]
    y_disp = params["y_disp"]
    inclination = params["inclination"]
    tip = params["tip"]
    propeller = params["propeller"]
    n_bp = len(sequence)

    # Select backbone template based on sugar pucker
    if form == "B":
        backbone = BACKBONE_C2ENDO_ANTI
    else:  # A-DNA
        backbone = BACKBONE_C3ENDO_ANTI

    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    all_atoms: List[Atom] = []

    for i in range(n_bp):
        base_s1 = sequence[i]
        base_s2 = comp_sequence[i]

        # Build strand I nucleotide
        nuc_s1 = _build_nucleotide(base_s1, backbone,
                                    is_strand2=False, propeller=propeller)
        # Build strand II nucleotide (complement, on opposite side)
        nuc_s2 = _build_nucleotide(base_s2, backbone,
                                    is_strand2=True, propeller=propeller)

        # Position on helix
        coords_s1, coords_s2 = _position_bp_on_helix(
            nuc_s1, nuc_s2,
            x_disp, y_disp, inclination, tip,
            rise, twist, i
        )

        # Create atoms for strand I
        res_name_s1 = RESIDUE_NAMES[base_s1]
        skip_p_s1 = (i == 0)  # No phosphate at 5' end
        all_atoms.extend(_atoms_to_list(
            nuc_s1, coords_s1, res_name_s1, i + 1, "A", skip_p_s1))

        # Create atoms for strand II
        # Strand II is antiparallel: numbered from n_bp down to 1
        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        skip_p_s2 = (i == n_bp - 1)  # 5' end of strand II
        all_atoms.extend(_atoms_to_list(
            nuc_s2, coords_s2, res_name_s2, res_seq_s2, "B", skip_p_s2))

    return all_atoms


def build_z_dna(sequence: str) -> List[Atom]:
    """
    Build Z-form double-stranded DNA.

    Z-DNA has a dinucleotide repeat with alternating conformations:
    - Purines: syn glycosidic angle, C3'-endo sugar
    - Pyrimidines: anti glycosidic angle, C2'-endo sugar

    The helical parameters alternate between two step types.

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

    params = HELICAL_PARAMS["Z"]
    x_disp = params["x_disp"]
    y_disp = params["y_disp"]
    inclination = params["inclination"]
    tip = params["tip"]
    propeller = params["propeller"]
    n_bp = len(sequence)

    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    all_atoms: List[Atom] = []

    # Accumulate rise and twist
    cum_rise = 0.0
    cum_twist = 0.0

    for i in range(n_bp):
        base_s1 = sequence[i]
        base_s2 = comp_sequence[i]

        # Select backbone based on base type
        if base_s1 in PURINE_BASES:
            backbone_s1 = BACKBONE_C3ENDO_SYN
        else:
            backbone_s1 = BACKBONE_C2ENDO_ANTI

        if base_s2 in PURINE_BASES:
            backbone_s2 = BACKBONE_C3ENDO_SYN
        else:
            backbone_s2 = BACKBONE_C2ENDO_ANTI

        # Build nucleotides
        nuc_s1 = _build_nucleotide(base_s1, backbone_s1,
                                    is_strand2=False, propeller=propeller)
        nuc_s2 = _build_nucleotide(base_s2, backbone_s2,
                                    is_strand2=True, propeller=propeller)

        # Get coordinates
        coords_s1 = np.array([[x, y, z] for (_, _, x, y, z) in nuc_s1])
        coords_s2 = np.array([[x, y, z] for (_, _, x, y, z) in nuc_s2])

        # Apply inclination
        if abs(inclination) > 0.01:
            R_inc = _rot_y(inclination)
            coords_s1 = (R_inc @ coords_s1.T).T
            coords_s2 = (R_inc @ coords_s2.T).T

        # Apply displacement
        displacement = np.array([x_disp, y_disp, 0.0])
        coords_s1 = coords_s1 + displacement
        coords_s2 = coords_s2 + displacement

        # Apply cumulative helical transformation
        R_twist = _rot_z(cum_twist)
        z_trans = np.array([0.0, 0.0, cum_rise])
        coords_s1 = (R_twist @ coords_s1.T).T + z_trans
        coords_s2 = (R_twist @ coords_s2.T).T + z_trans

        # Create atoms
        res_name_s1 = RESIDUE_NAMES[base_s1]
        skip_p_s1 = (i == 0)
        all_atoms.extend(_atoms_to_list(
            nuc_s1, coords_s1, res_name_s1, i + 1, "A", skip_p_s1))

        res_name_s2 = RESIDUE_NAMES[base_s2]
        res_seq_s2 = n_bp - i
        skip_p_s2 = (i == n_bp - 1)
        all_atoms.extend(_atoms_to_list(
            nuc_s2, coords_s2, res_name_s2, res_seq_s2, "B", skip_p_s2))

        # Accumulate for next step
        if i < n_bp - 1:
            # Determine step type based on position in dinucleotide
            if i % 2 == 0:
                cum_rise += params["rise_1"]
                cum_twist += params["twist_1"]
            else:
                cum_rise += params["rise_2"]
                cum_twist += params["twist_2"]

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

    if form in ("A", "B"):
        return build_ab_dna(sequence, form)
    else:
        return build_z_dna(sequence)
