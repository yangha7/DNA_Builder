"""
V2 DNA builder using internal coordinates (Z-matrix method).

Builds DNA structures by growing the sugar-phosphate backbone chain
atom-by-atom using internal coordinates (bond lengths, bond angles,
dihedral angles), then inserting bases.

The core algorithm:
1. place_atom() — Z-matrix operation: place atom given distance, angle, dihedral
2. grow_backbone() — Build backbone chain residue by residue
3. insert_base() — Place base atoms relative to sugar using template geometry
4. build_strand2() — Build complementary strand using Watson-Crick pairing

Internal coordinates are extracted from Colin's 3DNA fiber structures
(see extract_internal_coords.py and internal_coords.py).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .builder import Atom
from .internal_coords import (
    B_DNA_PARAMS, A_DNA_PARAMS,
    Z_DNA_POS1_PARAMS, Z_DNA_POS2_PARAMS,
    INTERNAL_COORDS,
    B_BASE_TEMPLATES, A_BASE_TEMPLATES, Z_BASE_TEMPLATES,
)
from .fiber_data import (
    HELICAL_PARAMS, WC_COMPLEMENT, RESIDUE_NAMES,
    PURINE_BASES, PYRIMIDINE_BASES,
)


# =============================================================================
# Core Z-matrix operation
# =============================================================================

def place_atom(d: float, theta: float, phi: float,
               ref1: np.ndarray, ref2: np.ndarray, ref3: np.ndarray) -> np.ndarray:
    """
    Place a new atom using internal coordinates (Z-matrix operation).

    The new atom is placed at:
    - Distance `d` from ref1
    - Bond angle `theta` (degrees) at ref1 in the ref2-ref1-new plane
    - Dihedral angle `phi` (degrees) for ref3-ref2-ref1-new

    This is the standard NERF (Natural Extension Reference Frame) algorithm.

    Parameters
    ----------
    d : float
        Bond length from ref1 to new atom (Å).
    theta : float
        Bond angle ref2-ref1-new (degrees).
    phi : float
        Dihedral angle ref3-ref2-ref1-new (degrees).
    ref1 : np.ndarray
        Position of bonded atom (closest reference).
    ref2 : np.ndarray
        Position of second reference atom.
    ref3 : np.ndarray
        Position of third reference atom.

    Returns
    -------
    np.ndarray
        Position of the new atom.
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Build local coordinate system at ref1
    # bc = unit vector from ref2 to ref1
    bc = ref1 - ref2
    bc = bc / np.linalg.norm(bc)

    # n = normal to plane ref3-ref2-ref1
    ab = ref2 - ref3
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate case: ref3, ref2, ref1 are collinear
        # Pick an arbitrary perpendicular direction
        if abs(bc[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        n = np.cross(bc, perp)
        n = n / np.linalg.norm(n)
    else:
        n = n / n_norm

    # m = perpendicular to bc in the bc-n plane
    m = np.cross(n, bc)

    # New atom position in local frame
    # d * [-cos(theta), cos(phi)*sin(theta), sin(phi)*sin(theta)]
    new_local = np.array([
        -d * np.cos(theta_rad),
        d * np.cos(phi_rad) * np.sin(theta_rad),
        d * np.sin(phi_rad) * np.sin(theta_rad),
    ])

    # Transform to global frame
    new_pos = ref1 + new_local[0] * bc + new_local[1] * m + new_local[2] * n

    return new_pos


# =============================================================================
# Backbone chain growth
# =============================================================================

def _get_params(form: str, position: int = 0) -> dict:
    """Get internal coordinate parameters for a given DNA form and position."""
    if form == "Z":
        if position % 2 == 0:
            return Z_DNA_POS1_PARAMS
        else:
            return Z_DNA_POS2_PARAMS
    elif form == "A":
        return A_DNA_PARAMS
    else:
        return B_DNA_PARAMS


def grow_backbone(n_residues: int, form: str = "B",
                  custom_torsions: Optional[List[Dict]] = None) -> List[Dict[str, np.ndarray]]:
    """
    Build the sugar-phosphate backbone chain atom-by-atom.

    For each residue, places atoms in this order:
    1. P (phosphorus) — connected to previous O3' via O3'-P bond
    2. O5' — from P, using alpha torsion
    3. C5' — from O5', using beta torsion
    4. C4' — from C5', using gamma torsion
    5. C3' — from C4', using delta torsion
    6. O3' — from C3', using epsilon torsion (connects to next P)
    7. O4' — branch from C4' (sugar ring)
    8. C2' — branch from C3' (sugar ring)
    9. C1' — branch from C2' (sugar ring, connects to O4')
    10. OP1, OP2 — branches from P

    Parameters
    ----------
    n_residues : int
        Number of nucleotide residues to build.
    form : str
        DNA form: "A", "B", or "Z".
    custom_torsions : list of dict, optional
        Per-residue torsion angle overrides.

    Returns
    -------
    list of dict
        Each dict maps atom name -> np.ndarray position.
    """
    residues = []

    # Initialize: place first 3 atoms to establish the coordinate frame
    # P at origin, O5' along +x, C5' in xy-plane
    params0 = _get_params(form, 0)
    bl = params0["bond_lengths"]
    ba = params0["bond_angles"]
    ta = params0["torsion_angles"]

    # First P at origin
    P0 = np.array([0.0, 0.0, 0.0])

    # O5' along +x at P-O5' bond length
    d_P_O5 = bl["P-O5'"]
    O5_0 = np.array([d_P_O5, 0.0, 0.0])

    # C5' in xy-plane
    d_O5_C5 = bl["O5'-C5'"]
    angle_P_O5_C5 = ba["P-O5'-C5'"]
    theta_rad = np.radians(angle_P_O5_C5)
    C5_0 = O5_0 + np.array([
        -d_O5_C5 * np.cos(theta_rad),
        d_O5_C5 * np.sin(theta_rad),
        0.0,
    ])

    # Now we have 3 reference points. Build the rest of residue 0.
    # We need a "virtual" previous O3' for the first residue's alpha torsion.
    # Place it using the zeta torsion from the previous (virtual) residue.
    # For the first residue, we create a virtual O3' behind P.
    # Use the O3'-P-O5' angle and place O3' in a reasonable position.
    d_O3_P = bl.get("O3'-P", 1.601)
    angle_O3_P_O5 = ba.get("O3'-P-O5'", 101.4)
    # Place virtual O3' using a default dihedral
    # We need a 4th reference point. Use a point along -z as virtual ref.
    virtual_ref = P0 + np.array([0.0, 0.0, -1.0])
    prev_O3 = place_atom(d_O3_P, angle_O3_P_O5, 0.0, P0, O5_0, virtual_ref)

    for i in range(n_residues):
        params = _get_params(form, i)
        bl = params["bond_lengths"]
        ba = params["bond_angles"]
        ta = params["torsion_angles"]

        if custom_torsions and i < len(custom_torsions):
            ta = {**ta, **custom_torsions[i]}

        nuc = {}

        if i == 0:
            # Use pre-computed positions for first residue
            nuc["P"] = P0
            nuc["O5'"] = O5_0
            nuc["C5'"] = C5_0
        else:
            # Place P from previous O3'
            prev_nuc = residues[i - 1]
            prev_O3 = prev_nuc["O3'"]
            prev_C3 = prev_nuc["C3'"]

            # Get previous residue params for zeta
            prev_params = _get_params(form, i - 1)
            prev_ta = prev_params["torsion_angles"]

            # P: bonded to prev_O3, angle at prev_O3 with prev_C3
            d_O3_P = bl.get("O3'-P", 1.601)
            angle_C3_O3_P = prev_params["bond_angles"].get("C3'-O3'-P+", 119.0)
            zeta = prev_ta.get("zeta", 160.5)
            # Dihedral: C4'(prev) - C3'(prev) - O3'(prev) - P(new)
            # But we use zeta = C3'(prev) - O3'(prev) - P(new) - O5'(new)
            # So for placing P, we need: epsilon = C4'(prev) - C3'(prev) - O3'(prev) - P(new)
            epsilon = prev_ta.get("epsilon", 140.8)
            prev_C4 = prev_nuc["C4'"]
            nuc["P"] = place_atom(d_O3_P, angle_C3_O3_P, epsilon,
                                  prev_O3, prev_C3, prev_C4)

            # O5': bonded to P, angle O3'-P-O5'
            d_P_O5 = bl["P-O5'"]
            angle_O3_P_O5 = ba.get("O3'-P-O5'", 101.4)
            # Dihedral: C3'(prev) - O3'(prev) - P - O5' = zeta
            nuc["O5'"] = place_atom(d_P_O5, angle_O3_P_O5, zeta,
                                    nuc["P"], prev_O3, prev_C3)

            # C5': bonded to O5', angle P-O5'-C5'
            d_O5_C5 = bl["O5'-C5'"]
            angle_P_O5_C5 = ba["P-O5'-C5'"]
            alpha = ta.get("alpha", 29.9)
            # Dihedral: O3'(prev) - P - O5' - C5' = alpha
            nuc["C5'"] = place_atom(d_O5_C5, angle_P_O5_C5, alpha,
                                    nuc["O5'"], nuc["P"], prev_O3)

        # C4': bonded to C5', angle O5'-C5'-C4'
        d_C5_C4 = bl["C5'-C4'"]
        angle_O5_C5_C4 = ba["O5'-C5'-C4'"]
        beta = ta.get("beta", -136.3)
        # Dihedral: P - O5' - C5' - C4' = beta
        nuc["C4'"] = place_atom(d_C5_C4, angle_O5_C5_C4, beta,
                                nuc["C5'"], nuc["O5'"], nuc["P"])

        # C3': bonded to C4', angle C5'-C4'-C3'
        d_C4_C3 = bl["C4'-C3'"]
        angle_C5_C4_C3 = ba["C5'-C4'-C3'"]
        gamma = ta.get("gamma", -31.1)
        # Dihedral: O5' - C5' - C4' - C3' = gamma
        nuc["C3'"] = place_atom(d_C4_C3, angle_C5_C4_C3, gamma,
                                nuc["C4'"], nuc["C5'"], nuc["O5'"])

        # O3': bonded to C3', angle C4'-C3'-O3'
        d_C3_O3 = bl["C3'-O3'"]
        angle_C4_C3_O3 = ba["C4'-C3'-O3'"]
        delta = ta.get("delta", -143.4)
        # Dihedral: C5' - C4' - C3' - O3' = delta
        nuc["O3'"] = place_atom(d_C3_O3, angle_C4_C3_O3, delta,
                                nuc["C3'"], nuc["C4'"], nuc["C5'"])

        # O4': branch from C4', using out-of-ring dihedral O5'-C5'-C4'-O4'
        d_C4_O4 = bl["C4'-O4'"]
        angle_C5_C4_O4 = ba.get("C5'-C4'-O4'", 112.4)
        dih_O5_C5_C4_O4 = ta.get("O5'-C5'-C4'-O4'", 88.1)
        nuc["O4'"] = place_atom(d_C4_O4, angle_C5_C4_O4, dih_O5_C5_C4_O4,
                                nuc["C4'"], nuc["C5'"], nuc["O5'"])

        # C2': branch from C3', using out-of-ring dihedral C5'-C4'-C3'-C2'
        d_C3_C2 = bl["C3'-C2'"]
        angle_C4_C3_C2 = ba.get("C2'-C3'-C4'", 104.7)
        dih_C5_C4_C3_C2 = ta.get("C5'-C4'-C3'-C2'", 100.7)
        nuc["C2'"] = place_atom(d_C3_C2, angle_C4_C3_C2, dih_C5_C4_C3_C2,
                                nuc["C3'"], nuc["C4'"], nuc["C5'"])

        # C1': from C2', using endocyclic dihedral C4'-C3'-C2'-C1'
        d_C2_C1 = bl["C2'-C1'"]
        angle_C3_C2_C1 = ba.get("C1'-C2'-C3'", 96.6)
        dih_C4_C3_C2_C1 = ta.get("C4'-C3'-C2'-C1'", 40.2)
        nuc["C1'"] = place_atom(d_C2_C1, angle_C3_C2_C1, dih_C4_C3_C2_C1,
                                nuc["C2'"], nuc["C3'"], nuc["C4'"])

        # OP1 (O1P): branch from P, using dihedral C5'-O5'-P-O1P
        d_P_OP1 = bl["P-O1P"]
        angle_O5_P_OP1 = ba.get("O1P-P-O5'", 109.6)
        dih_C5_O5_P_OP1 = ta.get("C5'-O5'-P-O1P", -85.9)
        nuc["O1P"] = place_atom(d_P_OP1, angle_O5_P_OP1, dih_C5_O5_P_OP1,
                                nuc["P"], nuc["O5'"], nuc["C5'"])

        # OP2 (O2P): branch from P, using dihedral C5'-O5'-P-O2P
        d_P_OP2 = bl["P-O2P"]
        angle_O5_P_OP2 = ba.get("O2P-P-O5'", 109.6)
        dih_C5_O5_P_OP2 = ta.get("C5'-O5'-P-O2P", 145.6)
        nuc["O2P"] = place_atom(d_P_OP2, angle_O5_P_OP2, dih_C5_O5_P_OP2,
                                nuc["P"], nuc["O5'"], nuc["C5'"])

        residues.append(nuc)

    return residues


# =============================================================================
# Base insertion
# =============================================================================

def _get_base_templates(form: str) -> dict:
    """Get base atom templates for a given DNA form."""
    if form == "Z":
        return Z_BASE_TEMPLATES
    elif form == "A":
        return A_BASE_TEMPLATES
    else:
        return B_BASE_TEMPLATES


def insert_base(base_type: str, nuc: Dict[str, np.ndarray],
                form: str = "B", position: int = 0) -> Dict[str, np.ndarray]:
    """
    Place base atoms relative to the sugar using template geometry.

    Uses a Kabsch-like superposition: align the template sugar atoms
    (C1', O4', C2') to the built backbone sugar atoms, then apply
    the same transformation to the base atoms.

    Parameters
    ----------
    base_type : str
        Base type: A, T, G, or C.
    nuc : dict
        Backbone atom positions (must contain C1', O4', C2').
    form : str
        DNA form.
    position : int
        Position index (for Z-DNA alternating conformations).

    Returns
    -------
    dict
        Base atom name -> position mapping.
    """
    templates = _get_base_templates(form)
    if base_type not in templates:
        raise ValueError(f"Unknown base type: {base_type}")

    bt = templates[base_type]

    # Template reference points (sugar atoms)
    t_c1 = np.array(bt["C1'"])
    t_o4 = np.array(bt["O4'"])
    t_c2 = np.array(bt["C2'"])

    # Built reference points
    b_c1 = nuc["C1'"]
    b_o4 = nuc["O4'"]
    b_c2 = nuc["C2'"]

    # Compute transformation: template -> built
    # Using Kabsch algorithm (SVD-based superposition)
    template_pts = np.array([t_c1, t_o4, t_c2])
    built_pts = np.array([b_c1, b_o4, b_c2])

    # Center both sets
    t_center = np.mean(template_pts, axis=0)
    b_center = np.mean(built_pts, axis=0)

    t_centered = template_pts - t_center
    b_centered = built_pts - b_center

    # SVD for optimal rotation
    H = t_centered.T @ b_centered
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T

    # Apply transformation to base atoms
    base_atoms = {}
    for atom_name, coords in bt["atoms"].items():
        t_pos = np.array(coords)
        # Transform: rotate centered template, then translate to built center
        b_pos = R @ (t_pos - t_center) + b_center
        base_atoms[atom_name] = b_pos

    return base_atoms


# =============================================================================
# Strand II construction
# =============================================================================

def build_strand2_from_templates(strand1_residues: List[Dict[str, np.ndarray]],
                                  sequence: str, form: str = "B") -> List[Dict[str, np.ndarray]]:
    """
    Build strand II by applying the helical dyad symmetry.

    For B-DNA and A-DNA, the dyad axis is perpendicular to the helix axis
    and passes through the center of each base pair. Strand II is generated
    by rotating strand I by 180° around this dyad axis.

    For Z-DNA, the dinucleotide repeat requires special handling.

    This function uses the v1 template approach for strand II: it builds
    strand I using Z-matrix, then uses the known helical parameters to
    generate strand II positions.

    Parameters
    ----------
    strand1_residues : list of dict
        Strand I backbone + base atom positions.
    sequence : str
        Strand I sequence (5'→3').
    form : str
        DNA form.

    Returns
    -------
    list of dict
        Strand II residue atom positions (in 5'→3' order of strand II,
        which is 3'→5' relative to strand I).
    """
    n_bp = len(sequence)
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    if form in ("B", "A"):
        return _build_strand2_helical(strand1_residues, sequence, comp_sequence, form)
    else:
        return _build_strand2_z_dna(strand1_residues, sequence, comp_sequence)


def _build_strand2_helical(strand1_residues, sequence, comp_sequence, form):
    """Build strand II for B-DNA or A-DNA using helical symmetry."""
    params = HELICAL_PARAMS[form]
    rise = params["rise"]
    twist = params["twist"]
    n_bp = len(sequence)

    # The dyad operation for a helix:
    # For base pair at step i, the dyad axis is in the xy-plane
    # at angle (twist * i / 2) from x-axis.
    # Strand II atom at step i = rotate strand I atom by 180° around dyad axis
    # then translate by rise * i along z.

    # But we need to figure out the dyad axis from the built structure.
    # Alternative approach: use the helical screw relationship.
    # Strand II residue at position i pairs with strand I residue at position i.
    # The strand II residue is related to strand I by a 180° rotation around
    # the dyad axis plus the helical parameters.

    # Simpler approach: build strand II backbone independently using the same
    # internal coordinates but with the complementary sequence, and position
    # it using the known helical relationship.

    # For now, use the dyad symmetry approach:
    # 1. Find the helix axis (should be along Z for our built structure)
    # 2. For each base pair, apply the dyad rotation

    # The helix axis is along Z (by construction).
    # The dyad axis for step i is perpendicular to Z, at angle twist*i/2.
    # Dyad rotation = 180° around this axis.

    # For the first base pair (i=0), the dyad axis is along x.
    # We need to find the correct dyad axis orientation from the structure.

    # Actually, let's use a more robust approach:
    # Build strand II using the same grow_backbone but with reversed parameters,
    # then align it using the Watson-Crick base pair geometry.

    # Most robust: use the template-based strand II from fiber_data
    # and superpose it onto our Z-matrix strand I.

    # For now, let's use the direct dyad approach with the helix axis along Z.
    # The dyad for step 0 needs to be determined from the structure.

    # Find the center of the first base pair
    # C1' of strand I residue 0 should be roughly at the expected position
    s1_c1_0 = strand1_residues[0]["C1'"]

    # The dyad axis passes through the helix axis (Z) and is perpendicular to it.
    # For step i, the dyad axis direction in xy-plane is at angle:
    #   phi_dyad = twist * i + phi_0
    # where phi_0 is the initial orientation.

    # Determine phi_0 from the first residue's C1' position
    phi_0 = np.arctan2(s1_c1_0[1], s1_c1_0[0])
    # The dyad axis bisects the angle between strand I and strand II C1' positions
    # For B-DNA, the C1'-C1' vector is roughly perpendicular to the dyad axis

    strand2_residues = []

    for i in range(n_bp):
        # Dyad axis direction for step i
        dyad_angle = np.radians(twist * i) + phi_0 + np.pi / 2  # perpendicular to C1' direction
        dyad_dir = np.array([np.cos(dyad_angle), np.sin(dyad_angle), 0.0])

        # Dyad point on helix axis at height rise * i
        dyad_point = np.array([0.0, 0.0, rise * i])

        # Apply 180° rotation around dyad axis through dyad_point
        s1_nuc = strand1_residues[i]
        s2_nuc = {}

        for atom_name, pos in s1_nuc.items():
            s2_nuc[atom_name] = _rotate_180_around_axis(pos, dyad_point, dyad_dir)

        strand2_residues.append(s2_nuc)

    # Strand II is in 3'→5' order relative to strand I pairing
    # We need to reverse it to get 5'→3' of strand II
    # Actually, the residues are already paired: s2[i] pairs with s1[i]
    # But strand II runs antiparallel, so s2[0] is the 3' end of strand II

    return strand2_residues


def _build_strand2_z_dna(strand1_residues, sequence, comp_sequence):
    """Build strand II for Z-DNA using dinucleotide dyad symmetry."""
    params = HELICAL_PARAMS["Z"]
    rise_dinuc = params["rise_dinuc"]
    twist_dinuc = params["twist_dinuc"]
    n_bp = len(sequence)

    s1_c1_0 = strand1_residues[0]["C1'"]
    phi_0 = np.arctan2(s1_c1_0[1], s1_c1_0[0])

    strand2_residues = []

    for i in range(n_bp):
        dinuc_idx = i // 2
        cum_twist = dinuc_idx * twist_dinuc
        cum_rise = dinuc_idx * rise_dinuc

        dyad_angle = np.radians(cum_twist) + phi_0 + np.pi / 2
        dyad_dir = np.array([np.cos(dyad_angle), np.sin(dyad_angle), 0.0])
        dyad_point = np.array([0.0, 0.0, cum_rise])

        s1_nuc = strand1_residues[i]
        s2_nuc = {}

        for atom_name, pos in s1_nuc.items():
            s2_nuc[atom_name] = _rotate_180_around_axis(pos, dyad_point, dyad_dir)

        strand2_residues.append(s2_nuc)

    return strand2_residues


def _rotate_180_around_axis(point: np.ndarray, axis_point: np.ndarray,
                             axis_dir: np.ndarray) -> np.ndarray:
    """Rotate a point 180° around an axis defined by a point and direction."""
    # Translate so axis passes through origin
    p = point - axis_point
    # 180° rotation around axis_dir: R = 2 * (n ⊗ n) - I
    n = axis_dir / np.linalg.norm(axis_dir)
    # Rodrigues' formula for 180°: R*p = 2*(n·p)*n - p
    rotated = 2.0 * np.dot(n, p) * n - p
    return rotated + axis_point


# =============================================================================
# Complete DNA builders (v2)
# =============================================================================

def _residues_to_atoms(residues: List[Dict[str, np.ndarray]],
                       sequence: str, chain_id: str = "A",
                       start_seq: int = 1,
                       reverse_numbering: bool = False) -> List[Atom]:
    """Convert residue dicts to Atom list with proper naming."""
    all_atoms = []

    # Define atom output order (matching v1 template order)
    backbone_order = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'",
                      "C3'", "O3'", "C2'", "C1'"]

    base_atom_order = {
        "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
        "T": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"],
        "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
        "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    }

    element_map = {
        "P": "P", "O1P": "O", "O2P": "O", "O5'": "O", "C5'": "C",
        "C4'": "C", "O4'": "O", "C3'": "C", "O3'": "O", "C2'": "C",
        "C1'": "C", "N9": "N", "C8": "C", "N7": "N", "C5": "C",
        "C6": "C", "N6": "N", "N1": "N", "C2": "C", "N3": "N",
        "C4": "C", "O2": "O", "O4": "O", "C5M": "C", "O6": "O",
        "N2": "N", "N4": "N",
    }

    n = len(residues)
    for idx, nuc in enumerate(residues):
        base = sequence[idx]
        res_name = RESIDUE_NAMES[base]
        if reverse_numbering:
            res_seq = start_seq + (n - 1 - idx)
        else:
            res_seq = start_seq + idx

        # Output backbone atoms
        for atom_name in backbone_order:
            if atom_name in nuc:
                pos = nuc[atom_name]
                elem = element_map.get(atom_name, atom_name[0])
                all_atoms.append(Atom(
                    name=atom_name, element=elem,
                    x=pos[0], y=pos[1], z=pos[2],
                    residue_name=res_name, residue_seq=res_seq,
                    chain_id=chain_id,
                ))

        # Output base atoms
        for atom_name in base_atom_order.get(base, []):
            if atom_name in nuc:
                pos = nuc[atom_name]
                elem = element_map.get(atom_name, atom_name[0])
                all_atoms.append(Atom(
                    name=atom_name, element=elem,
                    x=pos[0], y=pos[1], z=pos[2],
                    residue_name=res_name, residue_seq=res_seq,
                    chain_id=chain_id,
                ))

    return all_atoms


def _generate_5prime_terminal_O(nuc: Dict[str, np.ndarray],
                                 res_name: str, res_seq: int,
                                 chain_id: str) -> Optional[Atom]:
    """Generate 5' terminal oxygen (O5T) to complete the PO4 group."""
    if "P" not in nuc or "O5'" not in nuc:
        return None

    p_coord = nuc["P"]
    o5_coord = nuc["O5'"]

    direction = o5_coord - p_coord
    d = np.linalg.norm(direction)
    if d < 0.01:
        return None

    p_o_bond = 1.48
    o5t_coord = p_coord - (direction / d) * p_o_bond

    return Atom(
        name="O5T", element="O",
        x=o5t_coord[0], y=o5t_coord[1], z=o5t_coord[2],
        residue_name=res_name, residue_seq=res_seq, chain_id=chain_id,
    )


def build_b_dna_v2(sequence: str) -> List[Atom]:
    """
    Build B-form DNA using the Z-matrix (internal coordinate) method.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    return _build_dna_v2(sequence, "B")


def build_a_dna_v2(sequence: str) -> List[Atom]:
    """
    Build A-form DNA using the Z-matrix (internal coordinate) method.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    return _build_dna_v2(sequence, "A")


def build_z_dna_v2(sequence: str) -> List[Atom]:
    """
    Build Z-form DNA using the Z-matrix (internal coordinate) method.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for strand I (5' to 3').

    Returns
    -------
    List[Atom]
        Complete list of atoms for both strands.
    """
    return _build_dna_v2(sequence, "Z")


def _build_dna_v2(sequence: str, form: str) -> List[Atom]:
    """
    Internal implementation for building DNA using Z-matrix method.

    Steps:
    1. Grow strand I backbone using internal coordinates
    2. Insert bases onto strand I
    3. Build strand II using helical symmetry
    4. Insert bases onto strand II
    5. Generate terminal oxygens
    """
    import warnings

    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    if form == "Z":
        if len(sequence) % 2 != 0:
            raise ValueError("Z-DNA sequence must have even length (dinucleotide repeat)")
        is_canonical = all(
            (sequence[i] in PURINE_BASES) == (i % 2 == 0)
            for i in range(len(sequence))
        )
        if not is_canonical:
            warnings.warn(
                f"Z-DNA sequence '{sequence}' does not alternate purine-pyrimidine.",
                stacklevel=2,
            )

    n_bp = len(sequence)
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    # Step 1: Grow strand I backbone
    s1_backbone = grow_backbone(n_bp, form)

    # Step 2: Insert bases onto strand I
    for i in range(n_bp):
        base = sequence[i]
        base_atoms = insert_base(base, s1_backbone[i], form, i)
        s1_backbone[i].update(base_atoms)

    # Step 3: Build strand II using helical symmetry
    s2_residues = build_strand2_from_templates(s1_backbone, sequence, form)

    # Step 4: Re-insert correct bases for strand II
    # The dyad operation copies all atoms including bases from strand I,
    # but strand II has complementary bases. We need to replace them.
    for i in range(n_bp):
        comp_base = comp_sequence[i]
        # Remove strand I base atoms from s2
        base_atom_names = set()
        for b in ["A", "T", "G", "C"]:
            if b in B_BASE_TEMPLATES:
                base_atom_names.update(B_BASE_TEMPLATES[b]["atoms"].keys())

        s2_backbone = {k: v for k, v in s2_residues[i].items()
                       if k not in base_atom_names}

        # Insert correct complementary base
        comp_base_atoms = insert_base(comp_base, s2_backbone, form, i)
        s2_backbone.update(comp_base_atoms)
        s2_residues[i] = s2_backbone

    # Step 5: Convert to atom lists
    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # Strand I atoms
    s1_atoms = _residues_to_atoms(s1_backbone, sequence, "A", start_seq=1)
    all_atoms.extend(s1_atoms)

    # Generate 5' terminal O for strand I
    o5t = _generate_5prime_terminal_O(s1_backbone[0],
                                       RESIDUE_NAMES[sequence[0]], 1, "A")
    if o5t:
        terminal_atoms.append(o5t)

    # Strand II atoms (antiparallel: residue i pairs with strand I residue i)
    # Strand II runs 3'→5' relative to strand I, so numbering is reversed
    s2_atoms = _residues_to_atoms(s2_residues, comp_sequence, "B",
                                   start_seq=1, reverse_numbering=True)
    all_atoms.extend(s2_atoms)

    # Generate 5' terminal O for strand II (5' end is the last residue in pairing order)
    o5t_s2 = _generate_5prime_terminal_O(s2_residues[-1],
                                          RESIDUE_NAMES[comp_sequence[-1]],
                                          1, "B")
    if o5t_s2:
        terminal_atoms.append(o5t_s2)

    all_atoms.extend(terminal_atoms)

    return all_atoms


def build_dna_v2(sequence: str, form: str = "B") -> List[Atom]:
    """
    Build double-stranded DNA using the Z-matrix method.

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
        return build_b_dna_v2(sequence)
    elif form == "A":
        return build_a_dna_v2(sequence)
    else:
        return build_z_dna_v2(sequence)
