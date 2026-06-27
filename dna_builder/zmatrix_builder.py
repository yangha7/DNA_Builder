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
import warnings
from typing import List, Dict, Optional, Tuple
from .builder import Atom
from .internal_coords import (
    B_DNA_PARAMS, A_DNA_PARAMS,
    Z_DNA_POS1_PARAMS, Z_DNA_POS2_PARAMS,
    INTERNAL_COORDS,
    B_BASE_TEMPLATES, A_BASE_TEMPLATES,
    Z_BASE_TEMPLATES, Z_BASE_TEMPLATES_POS1, Z_BASE_TEMPLATES_POS2,
    B_CROSS_STRAND, A_CROSS_STRAND,
    Z_CROSS_STRAND_POS1, Z_CROSS_STRAND_POS2,
    STRAND2_BACKBONE_DIHEDRALS,
    BASE_GROWTH_IC,
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
    phi_rad = np.radians(-phi)  # negate to match standard dihedral convention

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


def _dihedral(a: np.ndarray, b: np.ndarray,
              c: np.ndarray, d: np.ndarray) -> float:
    """Return A-B-C-D dihedral angle in degrees."""
    b1 = b - a; b2 = c - b; b3 = d - c
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    norm1 = np.linalg.norm(n1); norm2 = np.linalg.norm(n2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    n1 /= norm1; n2 /= norm2
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


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
# Base growth from internal coordinates
# =============================================================================

def grow_base(base_type: str, nuc: Dict[str, np.ndarray],
              form: str = "B", position: int = 0) -> Dict[str, np.ndarray]:
    """
    Grow base atoms atom-by-atom from the sugar using internal coordinates.

    Uses the Z-matrix approach: each base atom is placed using a bond length,
    bond angle, and dihedral angle relative to three reference atoms that
    are already placed.

    The glycosidic torsion angle (chi) is taken from the form-specific
    internal coordinate parameters.

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
    if base_type not in BASE_GROWTH_IC:
        raise ValueError(f"Unknown base type: {base_type}")

    ic = BASE_GROWTH_IC[base_type]
    growth_steps = ic["growth"]

    # Get chi angle from form-specific parameters
    params = _get_params(form, position)
    ta = params["torsion_angles"]
    if base_type in PURINE_BASES:
        chi = ta.get("chi_pur", -58.5)
    else:
        chi = ta.get("chi_pyr", 154.2)

    # Start with sugar atoms
    atoms = dict(nuc)  # copy backbone atoms

    for step in growth_steps:
        new_atom, ref1_name, ref2_name, ref3_name, dist, ang, dih = step

        # Skip if already placed (e.g., C6 for pyrimidines placed twice)
        if new_atom in atoms:
            continue

        ref1 = atoms[ref1_name]
        ref2 = atoms[ref2_name]
        ref3 = atoms[ref3_name]

        # For the first two atoms (glycosidic bond), use chi
        if dih is None:
            dih = chi

        atoms[new_atom] = place_atom(dist, ang, dih, ref1, ref2, ref3)

    # Return only base atoms (not backbone)
    backbone_set = {"P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'",
                    "C3'", "O3'", "C2'", "C1'"}
    return {k: v for k, v in atoms.items() if k not in backbone_set}


# =============================================================================
# Base insertion (template-based, for A/B-DNA)
# =============================================================================

def _get_base_templates(form: str, position: int = 0) -> dict:
    """Get base atom templates for a given DNA form and position."""
    if form == "Z":
        if position % 2 == 0:
            return Z_BASE_TEMPLATES_POS1
        else:
            return Z_BASE_TEMPLATES_POS2
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

    For Z-DNA, uses a two-step alignment for non-canonical bases:
    1. Align the base-specific template to the canonical template
       (purine at pos1, pyrimidine at pos2)
    2. Align the canonical template to the built backbone
    This handles the fact that purines and pyrimidines have different
    C1'/O4'/C2' positions in the templates, but the Z-matrix backbone
    uses the same geometry for all bases.

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
    templates = _get_base_templates(form, position)
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

    # For Z-DNA non-canonical bases, use two-step alignment
    if form == "Z":
        is_pos2 = (position % 2 == 1)
        is_canonical = (base_type in PURINE_BASES and not is_pos2) or \
                       (base_type in PYRIMIDINE_BASES and is_pos2)

        if not is_canonical:
            # Get canonical template for this position
            if is_pos2:
                # Canonical pos2 = pyrimidine (T)
                canonical_bt = templates["T"]
            else:
                # Canonical pos1 = purine (A)
                canonical_bt = templates["A"]

            cn_c1 = np.array(canonical_bt["C1'"])
            cn_o4 = np.array(canonical_bt["O4'"])
            cn_c2 = np.array(canonical_bt["C2'"])

            # Step 1: Align base template sugar -> canonical sugar
            t_pts = np.array([t_c1, t_o4, t_c2])
            cn_pts = np.array([cn_c1, cn_o4, cn_c2])
            t_ctr = np.mean(t_pts, axis=0)
            cn_ctr = np.mean(cn_pts, axis=0)
            H1 = (t_pts - t_ctr).T @ (cn_pts - cn_ctr)
            U1, S1, Vt1 = np.linalg.svd(H1)
            d1 = np.linalg.det(Vt1.T @ U1.T)
            R1 = Vt1.T @ np.diag([1.0, 1.0, d1]) @ U1.T

            # Step 2: Align canonical sugar -> built sugar
            b_pts = np.array([b_c1, b_o4, b_c2])
            b_ctr = np.mean(b_pts, axis=0)
            H2 = (cn_pts - cn_ctr).T @ (b_pts - b_ctr)
            U2, S2, Vt2 = np.linalg.svd(H2)
            d2 = np.linalg.det(Vt2.T @ U2.T)
            R2 = Vt2.T @ np.diag([1.0, 1.0, d2]) @ U2.T

            # Apply combined transformation to base atoms
            base_atoms = {}
            for atom_name, coords in bt["atoms"].items():
                t_pos = np.array(coords)
                # Step 1: template -> canonical
                cn_pos = R1 @ (t_pos - t_ctr) + cn_ctr
                # Step 2: canonical -> built
                b_pos = R2 @ (cn_pos - cn_ctr) + b_ctr
                base_atoms[atom_name] = b_pos

            return base_atoms

    # Standard case (A/B-DNA, or canonical Z-DNA)
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
        b_pos = R @ (t_pos - t_center) + b_center
        base_atoms[atom_name] = b_pos

    return base_atoms


# =============================================================================
# Strand II construction — cross-strand Z-matrix method
# =============================================================================

def build_strand2_from_templates(strand1_residues: List[Dict[str, np.ndarray]],
                                  sequence: str, form: str = "B") -> List[Dict[str, np.ndarray]]:
    """
    Build strand II using cross-strand Z-matrix placement.

    For each base pair position i:
    1. Use strand 1's base H-bond atoms to place strand 2's C1' via Z-matrix
    2. Place O4' and C2' relative to C1' using cross-strand references
    3. Build the rest of the sugar ring from C1'/O4'/C2'
    4. Grow the backbone (C3', O3', C4', C5', O5', P) outward from the sugar
    5. Insert the complementary base

    The cross-strand Z-matrix parameters are extracted from Colin's 3DNA
    structures and stored in B_CROSS_STRAND.

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
        Strand II residue atom positions. s2[i] pairs with s1[i]
        (antiparallel: strand II runs 3'→5' relative to strand I).
    """
    n_bp = len(sequence)
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    strand2_residues = []

    for i in range(n_bp):
        s1_base = sequence[i]
        s2_base = comp_sequence[i]
        s1_nuc = strand1_residues[i]

        # Get cross-strand parameters (form-specific)
        bp_key = "{}->{}".format(s1_base, s2_base)
        if form == "Z":
            # Z-DNA: position-specific cross-strand params
            # s2 residue pairing with s1[i] is at position (n_bp-1-i) in strand 2
            s2_pos = (n_bp - 1 - i) % 2  # position type of s2 residue
            if s2_pos == 0:
                cs = Z_CROSS_STRAND_POS1[bp_key]
            else:
                cs = Z_CROSS_STRAND_POS2[bp_key]
        elif form == "A":
            cs = A_CROSS_STRAND[bp_key]
        else:
            cs = B_CROSS_STRAND[bp_key]

        # Reference atoms on strand 1's base
        ref_atom = cs["ref_atom"]      # N1 for purines, N3 for pyrimidines
        angle_ref = cs["angle_ref"]    # C2
        dihedral_ref = cs["dihedral_ref"]  # C6 for purines, C4 for pyrimidines

        ref1 = s1_nuc[ref_atom]
        ref2 = s1_nuc[angle_ref]
        ref3 = s1_nuc[dihedral_ref]

        # Step 1: Place C1' of strand 2 using cross-strand Z-matrix
        s2_c1 = place_atom(cs["C1'_dist"], cs["C1'_angle"], cs["C1'_dihedral"],
                           ref1, ref2, ref3)

        # Step 2: Place O4' from C1', using strand 1 H-bond atom as reference
        # For Z-DNA, use the s2 residue's own position params
        if form == "Z":
            s2_params = _get_params(form, n_bp - 1 - i)
        else:
            s2_params = _get_params(form, i)
        bl = s2_params["bond_lengths"]

        d_c1_o4 = bl["C1'-O4'"]
        s2_o4 = place_atom(d_c1_o4, cs["O4'_angle"], cs["O4'_dihedral"],
                           s2_c1, ref1, ref2)

        # Step 3: Place C2' from C1'
        d_c1_c2 = bl["C2'-C1'"]
        s2_c2 = place_atom(d_c1_c2, cs["C2'_angle"], cs["C2'_dihedral"],
                           s2_c1, ref1, ref2)

        # Step 4: Build the rest of the sugar and backbone
        s2_nuc = _build_sugar_and_backbone_from_c1(
            s2_c1, s2_o4, s2_c2, form, n_bp - 1 - i if form == "Z" else i,
            s2_params)

        strand2_residues.append(s2_nuc)

    # Connect backbone: for strand 2, the chain runs in the opposite direction
    # s2[0] pairs with s1[0] but is the 3' end of strand 2
    # s2[n-1] pairs with s1[n-1] and is the 5' end of strand 2
    # The backbone connectivity is: s2[n-1] -> s2[n-2] -> ... -> s2[0]
    # So O3' of s2[i] connects to P of s2[i-1] (reversed direction)

    return strand2_residues


def _build_sugar_and_backbone_from_c1(c1_pos, o4_pos, c2_pos,
                                       form, position, params):
    """
    Build sugar ring and backbone outward from C1', O4', C2'.

    Given the sugar frame (C1', O4', C2'), build:
    - C3' from C2' (using C1' as reference)
    - C4' from C3' (using C2' as reference)
    - O3' from C3' (branch)
    - C5' from C4' (branch)
    - O5' from C5' (branch)
    - P from O5' (branch)
    - OP1, OP2 from P (branches)
    """
    bl = params["bond_lengths"]
    ba = params["bond_angles"]
    ta = params["torsion_angles"]

    nuc = {}
    nuc["C1'"] = c1_pos
    nuc["O4'"] = o4_pos
    nuc["C2'"] = c2_pos

    # C3' from C2', angle C1'-C2'-C3', dihedral O4'-C1'-C2'-C3'
    d_c2_c3 = bl["C3'-C2'"]
    a_c1_c2_c3 = ba.get("C1'-C2'-C3'", 96.6)
    # Use endocyclic torsion: O4'-C1'-C2'-C3' (nu1)
    # nu1 can be derived from sugar pucker
    sp = params["sugar_pucker"]
    P_pseudo = sp.get("P", -26.1)
    tau_m = sp.get("tau_m", 44.8)
    # nu1 = tau_m * cos(P - 144°)  [Altona-Sundaralingam]
    nu1 = tau_m * np.cos(np.radians(P_pseudo - 144.0))
    nuc["C3'"] = place_atom(d_c2_c3, a_c1_c2_c3, nu1,
                            nuc["C2'"], nuc["C1'"], nuc["O4'"])

    # C4' from C3', angle C2'-C3'-C4', dihedral C1'-C2'-C3'-C4'
    d_c3_c4 = bl["C4'-C3'"]
    a_c2_c3_c4 = ba.get("C2'-C3'-C4'", 104.7)
    # nu2 = tau_m * cos(P)
    nu2 = tau_m * np.cos(np.radians(P_pseudo))
    nuc["C4'"] = place_atom(d_c3_c4, a_c2_c3_c4, nu2,
                            nuc["C3'"], nuc["C2'"], nuc["C1'"])

    # O3' from C3', angle C4'-C3'-O3'
    # Dihedral: C2'-C4'-C3'-O3' (form-specific, extracted from Colin's structures)
    d_c3_o3 = bl["C3'-O3'"]
    a_c4_c3_o3 = ba.get("C4'-C3'-O3'", 108.9)
    # place_atom(d, theta, phi, ref1=C3', ref2=C4', ref3=C2')
    # -> dihedral is C2'-C4'-C3'-O3'
    if form == "Z":
        # Use position-specific strand2 backbone dihedrals for Z-DNA
        pos_key = "Z_pos1" if position % 2 == 0 else "Z_pos2"
        s2_dihedrals = STRAND2_BACKBONE_DIHEDRALS[pos_key]
    else:
        s2_dihedrals = STRAND2_BACKBONE_DIHEDRALS.get(form, STRAND2_BACKBONE_DIHEDRALS["B"])
    dih_c2_c4_c3_o3 = s2_dihedrals["C2'-C4'-C3'-O3'"]
    nuc["O3'"] = place_atom(d_c3_o3, a_c4_c3_o3, dih_c2_c4_c3_o3,
                            nuc["C3'"], nuc["C4'"], nuc["C2'"])

    # C5' from C4', angle C3'-C4'-C5'
    # Dihedral: O4'-C3'-C4'-C5' (form-specific, extracted from Colin's structures)
    d_c4_c5 = bl["C5'-C4'"]
    a_c3_c4_c5 = ba.get("C5'-C4'-C3'", 115.8)
    # place_atom(d, theta, phi, ref1=C4', ref2=C3', ref3=O4')
    # -> dihedral is O4'-C3'-C4'-C5'
    dih_o4_c3_c4_c5 = s2_dihedrals["O4'-C3'-C4'-C5'"]
    nuc["C5'"] = place_atom(d_c4_c5, a_c3_c4_c5, dih_o4_c3_c4_c5,
                            nuc["C4'"], nuc["C3'"], nuc["O4'"])

    # O5' from C5', angle C4'-C5'-O5', dihedral from C3'
    d_c5_o5 = bl["O5'-C5'"]
    a_c4_c5_o5 = ba.get("O5'-C5'-C4'", 110.0)
    gamma = ta.get("gamma", -31.1)
    # gamma = O5'-C5'-C4'-C3'
    nuc["O5'"] = place_atom(d_c5_o5, a_c4_c5_o5, gamma,
                            nuc["C5'"], nuc["C4'"], nuc["C3'"])

    # P from O5', angle C5'-O5'-P, dihedral from C4'
    d_o5_p = bl["P-O5'"]
    a_c5_o5_p = ba.get("P-O5'-C5'", 119.0)
    beta = ta.get("beta", -136.3)
    # beta = P-O5'-C5'-C4'
    nuc["P"] = place_atom(d_o5_p, a_c5_o5_p, beta,
                          nuc["O5'"], nuc["C5'"], nuc["C4'"])

    # OP1 from P
    d_p_op1 = bl["P-O1P"]
    a_o5_p_op1 = ba.get("O1P-P-O5'", 109.6)
    dih_op1 = ta.get("C5'-O5'-P-O1P", -85.9)
    nuc["O1P"] = place_atom(d_p_op1, a_o5_p_op1, dih_op1,
                            nuc["P"], nuc["O5'"], nuc["C5'"])

    # OP2 from P
    d_p_op2 = bl["P-O2P"]
    a_o5_p_op2 = ba.get("O2P-P-O5'", 109.6)
    dih_op2 = ta.get("C5'-O5'-P-O2P", 145.6)
    nuc["O2P"] = place_atom(d_p_op2, a_o5_p_op2, dih_op2,
                            nuc["P"], nuc["O5'"], nuc["C5'"])

    return nuc


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
                                 chain_id: str,
                                 prev_O3: Optional[np.ndarray] = None) -> Optional[Atom]:
    """Generate 5' terminal oxygen (O5T) to complete the PO4 group.

    If prev_O3 is given (the phantom residue's O3' from the grow-and-chop
    approach), it is used directly — giving the correct ~101° O5T-P-O5'
    angle instead of the degenerate 180° from a linear projection.
    """
    if "P" not in nuc:
        return None

    p_coord = nuc["P"]

    if prev_O3 is not None:
        o5t_coord = prev_O3
    else:
        if "O5'" not in nuc:
            return None
        o5_coord = nuc["O5'"]
        direction = o5_coord - p_coord
        d = np.linalg.norm(direction)
        if d < 0.01:
            return None
        p_o_bond = 1.48
        o5t_coord = p_coord - (direction / d) * p_o_bond

    return Atom(
        name="O5T", element="O",
        x=float(o5t_coord[0]), y=float(o5t_coord[1]), z=float(o5t_coord[2]),
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


def _build_z_dna_v2(sequence: str) -> List[Atom]:
    """
    Build Z-form DNA using the Z-matrix (internal coordinate) method.

    Strategy: build a canonical alternating purine-pyrimidine structure
    first (which achieves 0.039 Å RMSD), then replace the base atoms
    for any non-canonical positions using grow_base().

    Steps:
    1. Construct a canonical sequence (A at pos1, T at pos2)
    2. Grow strand I backbone from internal coordinates
    3. Grow canonical bases (A/T) onto strand I
    4. Build strand II using cross-strand Z-matrix (canonical bases)
    5. Grow canonical bases onto strand II
    6. Replace base atoms at each position with the actual sequence's
       bases using grow_base() with the correct chi angle

    The backbone is identical for all base types at a given position
    (confirmed: 0.005 Å RMSD). Only the base atoms differ.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    if len(sequence) % 2 != 0:
        raise ValueError("Z-DNA sequence must have even length (dinucleotide repeat)")

    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_comp = "".join(comp[b] for b in reversed(sequence))
    if rev_comp != sequence:
        raise ValueError(
            f"Z-DNA sequence '{sequence}' is not palindromic. "
            f"Z-DNA builder currently requires palindromic sequences "
            f"(reverse complement equals the sequence). "
            f"The reverse complement is '{rev_comp}'."
        )

    is_canonical = all(
        (sequence[i] in PURINE_BASES) == (i % 2 == 0)
        for i in range(len(sequence))
    )
    if not is_canonical:
        warnings.warn(
            f"Z-DNA sequence '{sequence}' does not alternate purine-pyrimidine. "
            "Non-canonical Z-DNA may have reduced quality.",
            stacklevel=2,
        )

    n_bp = len(sequence)
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    # Build a canonical reference sequence: A at pos1, T at pos2
    canonical_s1_seq = "".join("A" if i % 2 == 0 else "T" for i in range(n_bp))
    canonical_comp = "".join(WC_COMPLEMENT[b] for b in canonical_s1_seq)

    # Step 1: Grow strand I backbone using internal coordinates.
    # Build 2 extra phantom residues at the 5' end so that the first actual
    # residue's P is placed using a real O3'(prev) rather than a virtual one.
    # 2 phantoms are required (not 1) to preserve Z-DNA's even/odd position parity:
    # s1_extended[i+2] uses backbone position i+2, parity (i+2)%2 == i%2.
    s1_extended = grow_backbone(n_bp + 2, "Z")
    phantom_O3_s1 = s1_extended[1]["O3'"]   # phantom residue 1's O3' = correct O5T position
    s1_residues = s1_extended[2:]  # discard phantom residues 0 and 1

    # Step 2: Grow canonical bases (A at pos1, T at pos2) onto strand I
    for i in range(n_bp):
        canonical_base = canonical_s1_seq[i]
        base_atoms = grow_base(canonical_base, s1_residues[i], "Z", i)
        s1_residues[i].update(base_atoms)

    # Step 3: Build strand II using cross-strand Z-matrix (canonical bases)
    s2_residues = build_strand2_from_templates(
        s1_residues, canonical_s1_seq, "Z")

    # Step 4: Grow canonical bases onto strand II
    for i in range(n_bp):
        canonical_comp_base = canonical_comp[i]
        s2_pos_idx = n_bp - 1 - i
        base_atoms = grow_base(canonical_comp_base, s2_residues[i], "Z", s2_pos_idx)
        s2_residues[i].update(base_atoms)

    # Step 5: Replace base atoms with actual sequence bases
    # The backbone remains unchanged; only base atoms are replaced.
    # Define which atoms are base atoms (not backbone)
    backbone_set = {"P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'",
                    "C3'", "O3'", "C2'", "C1'"}

    for i in range(n_bp):
        actual_base = sequence[i]
        canonical_base = canonical_s1_seq[i]
        if actual_base != canonical_base:
            # Remove old base atoms
            for key in list(s1_residues[i].keys()):
                if key not in backbone_set:
                    del s1_residues[i][key]
            # Grow new base atoms
            base_atoms = grow_base(actual_base, s1_residues[i], "Z", i)
            s1_residues[i].update(base_atoms)

    for i in range(n_bp):
        actual_comp = comp_sequence[i]
        canonical_comp_base = canonical_comp[i]
        if actual_comp != canonical_comp_base:
            s2_pos_idx = n_bp - 1 - i
            for key in list(s2_residues[i].keys()):
                if key not in backbone_set:
                    del s2_residues[i][key]
            base_atoms = grow_base(actual_comp, s2_residues[i], "Z", s2_pos_idx)
            s2_residues[i].update(base_atoms)

    # Step 7: Convert to atom lists
    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # Strand I atoms
    s1_atoms = _residues_to_atoms(s1_residues, sequence, "A", start_seq=1)
    all_atoms.extend(s1_atoms)

    # Generate 5' terminal O for strand I (use phantom O3' for correct ~101° angle)
    o5t = _generate_5prime_terminal_O(s1_residues[0],
                                       RESIDUE_NAMES[sequence[0]], 1, "A",
                                       prev_O3=phantom_O3_s1)
    if o5t:
        terminal_atoms.append(o5t)

    # Strand II atoms (antiparallel)
    s2_atoms = _residues_to_atoms(s2_residues, comp_sequence, "B",
                                   start_seq=1, reverse_numbering=True)
    all_atoms.extend(s2_atoms)

    # Generate 5' terminal O for strand II.
    # Backbone connectivity: O3'(s2[i+1]) links to P(s2[i]).
    # Compute O5T dihedral from a same-parity internal reference (offset -2 from
    # terminal preserves Z-DNA even/odd alternation: n_bp-3 has same parity as n_bp-1).
    s2_term = s2_residues[-1]
    ref_idx = max(0, n_bp - 3)
    s2_ref_O5T_dih = _dihedral(
        s2_residues[ref_idx]["C5'"], s2_residues[ref_idx]["O5'"],
        s2_residues[ref_idx]["P"],   s2_residues[ref_idx + 1]["O3'"],
    )
    s2_term_params = _get_params("Z", n_bp - 1)
    phantom_O3_s2 = place_atom(
        s2_term_params["bond_lengths"].get("O3'-P", 1.601),
        s2_term_params["bond_angles"].get("O3'-P-O5'", 101.4),
        s2_ref_O5T_dih,
        s2_term["P"], s2_term["O5'"], s2_term["C5'"],
    )
    o5t_s2 = _generate_5prime_terminal_O(s2_term,
                                          RESIDUE_NAMES[comp_sequence[-1]],
                                          1, "B", prev_O3=phantom_O3_s2)
    if o5t_s2:
        terminal_atoms.append(o5t_s2)

    all_atoms.extend(terminal_atoms)

    return all_atoms


def _build_dna_v2(sequence: str, form: str) -> List[Atom]:
    """
    Internal implementation for building DNA using Z-matrix method.

    Steps:
    1. Grow strand I backbone using internal coordinates
    2. Insert bases onto strand I
    3. Build strand II using cross-strand Z-matrix
    4. Insert bases onto strand II
    5. Generate terminal oxygens

    For Z-DNA, uses a separate code path with dinucleotide helical screw.
    """
    sequence = sequence.upper().strip()
    if not all(b in "ATGC" for b in sequence):
        raise ValueError(f"Invalid bases in sequence: {sequence}")

    # Z-DNA uses a separate builder
    if form == "Z":
        return _build_z_dna_v2(sequence)

    n_bp = len(sequence)
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)

    # Step 1: Grow strand I backbone.
    # Build 2 extra phantom residues so the first actual residue's P is placed
    # using a real O3'(prev) — fixes terminal phosphate orientation.
    s1_extended = grow_backbone(n_bp + 2, form)
    phantom_O3_s1 = s1_extended[1]["O3'"]   # for correct O5T placement
    s1_backbone = s1_extended[2:]  # discard phantom residues

    # Step 2: Insert bases onto strand I
    for i in range(n_bp):
        base = sequence[i]
        base_atoms = insert_base(base, s1_backbone[i], form, i)
        s1_backbone[i].update(base_atoms)

    # Step 3: Build strand II using cross-strand Z-matrix
    # s2_residues[i] pairs with s1_backbone[i] (antiparallel)
    s2_residues = build_strand2_from_templates(s1_backbone, sequence, form)

    # Step 4: Insert complementary bases onto strand II
    for i in range(n_bp):
        comp_base = comp_sequence[i]
        base_atoms = insert_base(comp_base, s2_residues[i], form, i)
        s2_residues[i].update(base_atoms)

    # Step 5: Convert to atom lists
    all_atoms: List[Atom] = []
    terminal_atoms: List[Atom] = []

    # Strand I atoms
    s1_atoms = _residues_to_atoms(s1_backbone, sequence, "A", start_seq=1)
    all_atoms.extend(s1_atoms)

    # Generate 5' terminal O for strand I (use phantom O3' for correct ~101° angle)
    o5t = _generate_5prime_terminal_O(s1_backbone[0],
                                       RESIDUE_NAMES[sequence[0]], 1, "A",
                                       prev_O3=phantom_O3_s1)
    if o5t:
        terminal_atoms.append(o5t)

    # Strand II atoms (antiparallel: residue i pairs with strand I residue i)
    # Strand II runs 3'→5' relative to strand I, so numbering is reversed
    s2_atoms = _residues_to_atoms(s2_residues, comp_sequence, "B",
                                   start_seq=1, reverse_numbering=True)
    all_atoms.extend(s2_atoms)

    # Generate 5' terminal O for strand II (5' end is the last residue in pairing order).
    # Backbone connectivity: O3'(s2[i+1]) links to P(s2[i]).
    # Compute O5T dihedral from an adjacent internal reference residue.
    s2_term = s2_residues[-1]
    # O3'(s2[-1]) is the prev O3' for P(s2[-2]) per connectivity rules
    s2_ref_O5T_dih = _dihedral(
        s2_residues[-2]["C5'"], s2_residues[-2]["O5'"],
        s2_residues[-2]["P"],   s2_residues[-1]["O3'"],
    )
    s2_term_params = _get_params(form, n_bp - 1)
    phantom_O3_s2 = place_atom(
        s2_term_params["bond_lengths"].get("O3'-P", 1.601),
        s2_term_params["bond_angles"].get("O3'-P-O5'", 101.4),
        s2_ref_O5T_dih,
        s2_term["P"], s2_term["O5'"], s2_term["C5'"],
    )
    o5t_s2 = _generate_5prime_terminal_O(s2_term,
                                          RESIDUE_NAMES[comp_sequence[-1]],
                                          1, "B", prev_O3=phantom_O3_s2)
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
