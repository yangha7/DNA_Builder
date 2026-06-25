"""
DNA conformation classifier.

Classifies a DNA structure as A-DNA, B-DNA, or Z-DNA by computing RMSD
against reference models built with our builder for the same sequence.

Algorithm:
1. Parse the input structure (PDB, mmCIF, or XYZ)
2. Extract the DNA sequence from the structure
3. Build reference A, B, and Z-DNA models for that sequence
4. Compute RMSD (Kabsch superposition) between input and each reference
5. Classify as the form with the lowest RMSD
6. Report helical parameters (rise, twist, handedness) as supporting evidence
"""

import re
import sys
import warnings
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .builder import Atom, build_dna
from .io_parser import parse_structure, extract_sequence_from_atoms, detect_format
from .fiber_data import HELICAL_PARAMS, WC_COMPLEMENT

# ---------------------------------------------------------------------------
# Canonical helical parameters
# rise/twist: from HELICAL_PARAMS in fiber_data.py (fiber diffraction values).
# nu2: ν₂ endocyclic torsion C2'–C3'–C4'–O4' measured directly from the
#      builder's own reference templates.  Positive = C3'-endo-like (north,
#      A-form); negative = C2'-endo-like (south, B-form); Z alternates,
#      averaging near zero.  These values are internally consistent with the
#      torsion convention used in _torsion() below.
# ---------------------------------------------------------------------------
_CANONICAL_HELICAL: Dict[str, Dict[str, float]] = {
    'A': {'rise': 2.548, 'twist': 32.727, 'nu2':  40.0},
    'B': {'rise': 3.375, 'twist': 36.0,   'nu2': -33.0},
    'Z': {'rise': 3.625, 'twist': -30.0,  'nu2':   4.0},  # average of alternating ≈ +24° / −16°
}

# Within-form standard deviations used to normalise deviations for the
# helical penalty.  These are larger than ideal-model spreads to accommodate
# the natural variability of real (non-ideal) structures.
_SIGMA_RISE = 0.40   # Å
_SIGMA_NU2  = 8.0    # °  (ν₂ torsion)

# 1 σ combined deviation contributes this many Å to the classification score.
_HELICAL_SCALE  = 0.5   # Å per σ unit
# Weight of the helical penalty relative to the RMSD term in the combined score.
_HELICAL_WEIGHT = 1.0


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute RMSD between two point sets using Kabsch algorithm.

    Finds the optimal rotation and translation to superimpose P onto Q,
    then returns the RMSD.

    Parameters
    ----------
    P : np.ndarray, shape (N, 3)
        First point set (will be rotated/translated).
    Q : np.ndarray, shape (N, 3)
        Second point set (reference).

    Returns
    -------
    rmsd : float
        Root mean square deviation after optimal superposition.
    R : np.ndarray, shape (3, 3)
        Optimal rotation matrix.
    t : np.ndarray, shape (3,)
        Optimal translation vector.
    """
    assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"
    n = P.shape[0]

    # Center both point sets
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    P_rotated = (R @ P_centered.T).T
    diff = P_rotated - Q_centered
    rmsd = np.sqrt((diff ** 2).sum() / n)

    # Translation
    t = centroid_Q - R @ centroid_P

    return rmsd, R, t


def _match_atoms_by_identity(input_atoms: List[Atom],
                              ref_atoms: List[Atom],
                              chain: Optional[str] = None
                              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match atoms between input and reference structures by
    (chain_id, residue_seq, atom_name).
    """
    # Build lookup for reference atoms: (chain, seq, name) -> coords
    ref_lookup: Dict[Tuple[str, int, str], np.ndarray] = {}
    for atom in ref_atoms:
        if chain and atom.chain_id != chain:
            continue
        key = (atom.chain_id, atom.residue_seq, atom.name)
        ref_lookup[key] = np.array([atom.x, atom.y, atom.z])

    # Match input atoms
    input_coords = []
    ref_coords = []

    for atom in input_atoms:
        if chain and atom.chain_id != chain:
            continue
        key = (atom.chain_id, atom.residue_seq, atom.name)
        if key in ref_lookup:
            input_coords.append([atom.x, atom.y, atom.z])
            ref_coords.append(ref_lookup[key])

    if not input_coords:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), 0

    return (np.array(input_coords), np.array(ref_coords), len(input_coords))


def _match_atoms_by_residue_offset(input_atoms: List[Atom],
                                    ref_atoms: List[Atom],
                                    chain: Optional[str] = None
                                    ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match atoms between input and reference when residue numbering differs.

    Maps residue sequences by order (1st residue in input -> 1st in ref, etc.)
    then matches by atom name within each residue.
    """
    def _group_by_residue(atoms, ch):
        groups: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        order: List[Tuple[str, int]] = []
        for atom in atoms:
            if ch and atom.chain_id != ch:
                continue
            key = (atom.chain_id, atom.residue_seq)
            if key not in groups:
                groups[key] = {}
                order.append(key)
            groups[key][atom.name] = np.array([atom.x, atom.y, atom.z])
        return groups, order

    input_groups, input_order = _group_by_residue(input_atoms, chain)
    ref_groups, ref_order = _group_by_residue(ref_atoms, chain)

    n_residues = min(len(input_order), len(ref_order))

    input_coords = []
    ref_coords = []

    for i in range(n_residues):
        inp_key = input_order[i]
        ref_key = ref_order[i]
        inp_atoms = input_groups[inp_key]
        ref_atoms_dict = ref_groups[ref_key]

        for atom_name, inp_coord in inp_atoms.items():
            if atom_name in ref_atoms_dict:
                input_coords.append(inp_coord)
                ref_coords.append(ref_atoms_dict[atom_name])

    if not input_coords:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), 0

    return (np.array(input_coords), np.array(ref_coords), len(input_coords))


def _greedy_element_match(inp_coords_by_elem: Dict[str, np.ndarray],
                           ref_coords_by_elem: Dict[str, np.ndarray],
                           threshold: float = 5.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbor matching by element type."""
    input_matched = []
    ref_matched = []

    for elem in inp_coords_by_elem:
        if elem not in ref_coords_by_elem:
            continue
        inp_arr = inp_coords_by_elem[elem]
        ref_arr = ref_coords_by_elem[elem]

        used_ref = set()
        for inp_coord in inp_arr:
            dists = np.linalg.norm(ref_arr - inp_coord, axis=1)
            for idx in np.argsort(dists):
                if idx not in used_ref:
                    if dists[idx] < threshold:
                        input_matched.append(inp_coord)
                        ref_matched.append(ref_arr[idx])
                        used_ref.add(idx)
                    break

    if not input_matched:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    return np.array(input_matched), np.array(ref_matched)


def _match_atoms_by_element_proximity(input_atoms: List[Atom],
                                       ref_atoms: List[Atom]
                                       ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match atoms between input and reference by element type and proximity.

    Used for XYZ files where atom names and residue info are not available.
    Uses a two-pass approach:
    1. Rough centroid alignment + greedy matching
    2. Kabsch superposition on initial matches
    3. Re-match with tighter threshold after alignment
    """
    inp_coords_all = np.array([[a.x, a.y, a.z] for a in input_atoms])
    ref_coords_all = np.array([[a.x, a.y, a.z] for a in ref_atoms])

    # Center both sets
    inp_centroid = inp_coords_all.mean(axis=0)
    ref_centroid = ref_coords_all.mean(axis=0)
    offset = ref_centroid - inp_centroid

    # Group by element
    elements = set(a.element for a in input_atoms) & set(a.element for a in ref_atoms)

    # Build element-grouped coordinate arrays
    inp_orig_by_elem: Dict[str, np.ndarray] = {}
    ref_by_elem: Dict[str, np.ndarray] = {}
    for elem in elements:
        inp_orig_by_elem[elem] = np.array(
            [[a.x, a.y, a.z] for a in input_atoms if a.element == elem])
        ref_by_elem[elem] = np.array(
            [[a.x, a.y, a.z] for a in ref_atoms if a.element == elem])

    # Pass 1: rough match with centroid offset
    inp_shifted = {e: coords + offset for e, coords in inp_orig_by_elem.items()}
    pass1_inp, pass1_ref = _greedy_element_match(inp_shifted, ref_by_elem,
                                                  threshold=8.0)

    if pass1_inp.shape[0] < 4:
        if pass1_inp.shape[0] == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), 0
        return pass1_inp, pass1_ref, pass1_inp.shape[0]

    # Pass 2: Kabsch alignment on pass-1 matches, then re-match
    _, R, t = _kabsch_rmsd(pass1_inp, pass1_ref)

    # Transform ALL original input coords using the Kabsch rotation+translation
    inp_aligned = {}
    for elem in elements:
        orig = inp_orig_by_elem[elem]
        # Apply: x_aligned = R @ (x_orig + offset) + (t - R @ offset)
        # Simplify: x_aligned = R @ x_orig + t
        # Wait, t was computed from pass1 which used shifted coords
        # pass1_inp = orig + offset, so t = centroid_ref - R @ centroid_pass1
        # For original coords: x_aligned = R @ x_orig + (R @ offset + t - R @ offset)
        # Actually just: x_aligned = R @ (x_orig + offset) + (t - R @ (inp_centroid + offset))...
        # Simpler: just apply R and t to the shifted coords
        shifted = orig + offset
        inp_aligned[elem] = (R @ shifted.T).T + (t - R @ np.zeros(3))
        # Actually t already accounts for the centroid shift in Kabsch
        # Let me just do it properly:
        inp_aligned[elem] = (R @ shifted.T).T + t - (R @ pass1_inp.mean(axis=0) - pass1_ref.mean(axis=0) + pass1_inp.mean(axis=0))
        # This is getting complicated. Let me just recompute properly.

    # Simpler approach: compute R and t that maps original coords to ref frame
    # pass1_inp are shifted coords, pass1_ref are ref coords
    # _kabsch_rmsd centers both, so R maps centered_P to centered_Q
    # t = centroid_Q - R @ centroid_P
    # So for any shifted coord x: x_aligned = R @ x + t
    inp_aligned = {}
    for elem in elements:
        shifted = inp_orig_by_elem[elem] + offset
        inp_aligned[elem] = (R @ shifted.T).T + t

    pass2_inp, pass2_ref = _greedy_element_match(inp_aligned, ref_by_elem,
                                                  threshold=3.0)

    if pass2_inp.shape[0] < 4:
        return pass1_inp, pass1_ref, pass1_inp.shape[0]

    return pass2_inp, pass2_ref, pass2_inp.shape[0]


def _torsion(a: np.ndarray, b: np.ndarray,
             c: np.ndarray, d: np.ndarray) -> float:
    """Signed torsion angle (degrees) for the four-atom sequence a–b–c–d."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def _compute_helical_params(atoms: List[Atom],
                             chain: str = 'A',
                             chain_b: Optional[str] = None
                             ) -> Dict[str, float]:
    """
    Estimate helical parameters from a DNA structure.

    Rise is computed from the midpoints of complementary C1'–C1' pairs when a
    second chain is supplied (dsDNA path); this is accurate to ~0.03 Å for
    sequences ≥ 10 bp.  For single-strand input the rise is estimated from
    C1' PCA, which becomes reliable only for longer sequences.

    Twist is computed from strand-A C1' atoms projected onto the
    midpoint-derived axis, or the PCA axis for single-strand input.

    ν₂ (C2'–C3'–C4'–O4') is the endocyclic ring torsion that directly
    encodes sugar pucker: positive values indicate C3'-endo-like (A-form)
    geometry; negative values indicate C2'-endo-like (B-form) geometry.
    It requires no axis estimation and is reliable at any sequence length.

    Phosphate radius (mean distance from P atoms to the helix axis) is
    included for display only.
    """
    # ── Collect C1' atoms for the primary strand ──────────────────────────
    c1_A = sorted(
        [(a.residue_seq, np.array([a.x, a.y, a.z]))
         for a in atoms if a.chain_id == chain and a.name == "C1'"],
        key=lambda x: x[0]
    )
    n_A = len(c1_A)
    if n_A < 3:
        return {'rise': 0.0, 'twist': 0.0, 'handedness': 0, 'n_bp': n_A,
                'nu2_mean': None, 'nu2_std': None, 'phosphate_radius': 0.0}

    # ── Rise: midpoints of complementary pairs (dsDNA) or PCA (ssDNA) ──────
    axis: Optional[np.ndarray] = None

    if chain_b:
        c1_B = sorted(
            [(a.residue_seq, np.array([a.x, a.y, a.z]))
             for a in atoms if a.chain_id == chain_b and a.name == "C1'"],
            key=lambda x: x[0]
        )
        n_bp = min(n_A, len(c1_B))
        if n_bp >= 6:
            # Strand B runs antiparallel → reverse its order before pairing
            mids = np.array([
                (c1_A[i][1] + c1_B[n_bp - 1 - i][1]) / 2.0
                for i in range(n_bp)
            ])
            centroid_m = mids.mean(axis=0)
            centered_m = mids - centroid_m
            _, _, Vt_m = np.linalg.svd(centered_m)
            axis = Vt_m[0]
            if np.dot(axis, mids[-1] - mids[0]) < 0:
                axis = -axis
            proj_m = centered_m @ axis
            avg_rise = float(abs(np.mean(np.diff(proj_m))))
        else:
            chain_b = None   # fall through to single-strand path

    if axis is None:
        coords_arr = np.array([c[1] for c in c1_A])
        centroid = coords_arr.mean(axis=0)
        centered = coords_arr - centroid
        _, _, Vt = np.linalg.svd(centered)
        # For n ≤ 10 bp the axial variance is smaller than radial variance,
        # so the helix axis is the LAST (smallest) principal component.
        # For n > 10 the axial variance dominates and it is the FIRST.
        axis = Vt[0] if n_A > 10 else Vt[-1]
        if np.dot(axis, coords_arr[-1] - coords_arr[0]) < 0:
            axis = -axis
        proj = centered @ axis
        avg_rise = float(abs(np.mean(np.diff(proj))))
        n_bp = n_A

    # ── Twist: strand-A C1' atoms projected onto the derived axis ─────────
    coords_A = np.array([c[1] for c in c1_A])
    centroid_A = coords_A.mean(axis=0)
    centered_A = coords_A - centroid_A
    perp_A = centered_A - np.outer(centered_A @ axis, axis)

    twists = []
    for i in range(len(perp_A) - 1):
        v1, v2 = perp_A[i], perp_A[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 0.01 or n2 < 0.01:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        if np.dot(np.cross(v1, v2), axis) < 0:
            angle = -angle
        twists.append(angle)

    avg_twist = float(np.mean(twists)) if twists else 0.0
    handedness = 1 if avg_twist > 0 else (-1 if avg_twist < 0 else 0)

    # ── ν₂ sugar pucker: C2'–C3'–C4'–O4' per residue ────────────────────
    by_res: Dict[int, Dict[str, np.ndarray]] = {}
    for atom in atoms:
        if atom.chain_id != chain:
            continue
        by_res.setdefault(atom.residue_seq, {})[atom.name] = \
            np.array([atom.x, atom.y, atom.z])

    nu2_vals = []
    for res_dict in by_res.values():
        if all(k in res_dict for k in ("C2'", "C3'", "C4'", "O4'")):
            nu2 = _torsion(res_dict["C2'"], res_dict["C3'"],
                           res_dict["C4'"], res_dict["O4'"])
            nu2_vals.append(nu2)

    nu2_mean = float(np.mean(nu2_vals)) if nu2_vals else None
    nu2_std  = float(np.std(nu2_vals))  if nu2_vals else None

    # ── Phosphate radius (display only) ───────────────────────────────────
    p_coords = np.array([
        [a.x, a.y, a.z] for a in atoms
        if a.chain_id == chain and a.name == 'P'
    ])
    if len(p_coords) >= 2:
        centroid_ref = coords_A.mean(axis=0)
        p_cent = p_coords - centroid_ref
        p_proj = np.outer(p_cent @ axis, axis)
        p_perp = p_cent - p_proj
        phosphate_radius = float(np.mean(np.linalg.norm(p_perp, axis=1)))
    else:
        phosphate_radius = 0.0

    return {
        'rise':              avg_rise,
        'twist':             avg_twist,
        'handedness':        handedness,
        'n_bp':              n_bp,
        'nu2_mean':          nu2_mean,
        'nu2_std':           nu2_std,
        'phosphate_radius':  phosphate_radius,
    }


def _helical_penalty(helical: Dict, form: str) -> float:
    """
    Helical-parameter penalty for *form* ('A', 'B', or 'Z') given measured
    *helical* parameters.

    Combines normalised rise and ν₂ deviations from canonical values into a
    single Å-equivalent penalty that can be added directly to RMSD.

    Returns 0.0 if helical parameters are unavailable.
    """
    canonical = _CANONICAL_HELICAL.get(form)
    if canonical is None or helical.get('n_bp', 0) < 3:
        return 0.0

    terms: List[float] = []

    rise = helical.get('rise')
    if rise is not None and rise > 0:
        terms.append(((rise - canonical['rise']) / _SIGMA_RISE) ** 2)

    nu2 = helical.get('nu2_mean')
    if nu2 is not None:
        terms.append(((nu2 - canonical['nu2']) / _SIGMA_NU2) ** 2)

    if not terms:
        return 0.0

    return _HELICAL_SCALE * float(np.sqrt(sum(terms)))


def _assess_confidence(rmsds: Dict[str, float],
                       combined: Dict[str, float]) -> Tuple[str, str]:
    """
    Assess classification confidence from RMSD values and the combined scores.

    Returns (confidence_level, explanation).
    """
    best_form = min(combined, key=combined.get)
    best_rmsd_form = min(rmsds, key=rmsds.get)

    sorted_combined = sorted(combined.items(), key=lambda x: x[1])
    best_score = sorted_combined[0][1]
    second_score = sorted_combined[1][1] if len(sorted_combined) > 1 else best_score
    gap = second_score - best_score

    # Flag when RMSD alone would pick a different form
    rmsd_conflict = (best_form != best_rmsd_form)
    conflict_note = (
        f"; note RMSD alone favours {best_rmsd_form} — "
        f"helical parameters override"
        if rmsd_conflict else ""
    )

    if best_score < 1.5:
        if gap > 2.0:
            return "High", f"{best_form} score very low and clearly best{conflict_note}"
        elif gap > 0.8:
            return "High", f"{best_form} score low and clearly lower than alternatives{conflict_note}"
        else:
            return "Medium", f"{best_form} score low but close to {sorted_combined[1][0]}{conflict_note}"
    elif best_score < 4.0:
        if gap > 2.5:
            return "High", f"{best_form} score significantly lower than alternatives{conflict_note}"
        elif gap > 1.2:
            return "Medium", f"{best_form} score moderately lower than {sorted_combined[1][0]}{conflict_note}"
        else:
            return "Low", f"{best_form} and {sorted_combined[1][0]} scores are close{conflict_note}"
    else:
        if gap > 2.5:
            return "Medium", f"{best_form} score lower than alternatives but all scores are high{conflict_note}"
        else:
            return "Low", f"All scores are high; structure may not match canonical forms well{conflict_note}"


def _extract_sequence_from_filename(filepath: str) -> Optional[str]:
    """
    Try to extract a DNA sequence from the filename.

    Handles patterns like:
      - B_ATATAT.xyz -> ATATAT
      - A_GCGCGC.pdb -> GCGCGC
      - fold_atatat_model_0.cif -> ATATAT
    """
    stem = Path(filepath).stem.upper()

    # Pattern: X_SEQUENCE (e.g., B_ATATAT, A_GCGCGC)
    match = re.match(r'^[ABZ]_([ATGC]{2,})$', stem)
    if match:
        return match.group(1)

    # Pattern: anything_SEQUENCE_anything (e.g., fold_atatat_model_0)
    # Find the longest run of ATGC characters
    runs = re.findall(r'[ATGC]{2,}', stem)
    if runs:
        return max(runs, key=len)

    return None


def classify_structure(filepath: str, verbose: bool = True,
                       sequence: Optional[str] = None) -> Dict:
    """
    Classify a DNA structure as A-DNA, B-DNA, or Z-DNA.

    Parameters
    ----------
    filepath : str
        Path to the input structure file (PDB, CIF, or XYZ).
    verbose : bool
        If True, print detailed results to stdout.
    sequence : str, optional
        DNA sequence override. If not provided, extracted from the structure
        or filename.

    Returns
    -------
    dict with keys:
        'classification': str ('A-DNA', 'B-DNA', or 'Z-DNA')
        'rmsds': dict mapping form name to RMSD value
        'sequence': str
        'chain_ids': list of str
        'n_matched': dict mapping form name to number of matched atoms
        'confidence': str
        'confidence_detail': str
        'helical_params': dict with rise, twist, handedness
    """
    # Step 1: Parse input structure
    input_atoms = parse_structure(filepath)
    if not input_atoms:
        raise ValueError(f"No atoms found in {filepath}")

    # Step 2: Extract sequence
    fmt = detect_format(filepath)
    is_xyz = (fmt == 'xyz')

    if sequence:
        # User-provided sequence
        sequence = sequence.upper().strip()
        chain_ids = sorted(set(a.chain_id for a in input_atoms))
    else:
        seq_from_atoms, chain_ids = extract_sequence_from_atoms(input_atoms)

        if seq_from_atoms and '?' not in seq_from_atoms:
            sequence = seq_from_atoms
        else:
            # Try extracting from filename
            seq_from_name = _extract_sequence_from_filename(filepath)
            if seq_from_name:
                sequence = seq_from_name
                if verbose:
                    print(f"  Note: Sequence extracted from filename: {sequence}",
                          file=sys.stderr)
            else:
                raise ValueError(
                    f"Could not extract valid DNA sequence from {filepath}. "
                    f"Got: '{seq_from_atoms}'. "
                    "Use --sequence-hint to specify the sequence manually."
                )

    n_bp = len(sequence)

    # Step 3: Build reference models
    forms_to_test = ['A', 'B']

    # Z-DNA requires even-length sequence
    if n_bp % 2 == 0:
        forms_to_test.append('Z')

    ref_models: Dict[str, List[Atom]] = {}
    for form in forms_to_test:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_models[form] = build_dna(sequence, form)
        except ValueError as e:
            if verbose:
                print(f"  Note: Cannot build {form}-DNA for this sequence: {e}",
                      file=sys.stderr)

    if not ref_models:
        raise ValueError("Could not build any reference models for the sequence")

    # Step 4: Compute RMSD for each form
    rmsds: Dict[str, float] = {}
    n_matched: Dict[str, int] = {}

    primary_chain = chain_ids[0] if chain_ids else 'A'

    for form, ref_atoms in ref_models.items():
        if is_xyz:
            # XYZ files: use element-based proximity matching
            inp_coords, ref_coords, matched = _match_atoms_by_element_proximity(
                input_atoms, ref_atoms)
        else:
            # PDB/CIF: try direct matching by (chain, seq, name)
            inp_coords, ref_coords, matched = _match_atoms_by_identity(
                input_atoms, ref_atoms, chain=primary_chain)

            # If direct matching fails, try residue-offset matching
            if matched < 5:
                inp_coords, ref_coords, matched = _match_atoms_by_residue_offset(
                    input_atoms, ref_atoms, chain=primary_chain)

        if matched < 3:
            if verbose:
                print(f"  Warning: Only {matched} atoms matched for {form}-DNA",
                      file=sys.stderr)
            continue

        rmsd, _, _ = _kabsch_rmsd(inp_coords, ref_coords)
        rmsds[f"{form}-DNA"] = rmsd
        n_matched[f"{form}-DNA"] = matched

    if not rmsds:
        raise ValueError("Could not match enough atoms for RMSD computation")

    # Step 5: Helical parameters
    # Use the second chain (if present) for the more-accurate midpoint rise estimate
    second_chain = chain_ids[1] if len(chain_ids) >= 2 else None
    helical = _compute_helical_params(
        input_atoms, chain=primary_chain, chain_b=second_chain)

    # Step 6: Combined score = RMSD + helical penalty
    helical_penalties: Dict[str, float] = {}
    combined_scores:   Dict[str, float] = {}
    for form_key in rmsds:
        form = form_key.split('-')[0]          # 'A', 'B', or 'Z'
        pen = _helical_penalty(helical, form)
        helical_penalties[form_key] = pen
        combined_scores[form_key] = rmsds[form_key] + _HELICAL_WEIGHT * pen

    # Step 7: Classify on combined score
    best_form = min(combined_scores, key=combined_scores.get)

    # Confidence assessment
    confidence, confidence_detail = _assess_confidence(rmsds, combined_scores)

    result = {
        'classification':    best_form,
        'rmsds':             rmsds,
        'helical_penalties': helical_penalties,
        'combined_scores':   combined_scores,
        'sequence':          sequence,
        'chain_ids':         chain_ids,
        'n_matched':         n_matched,
        'confidence':        confidence,
        'confidence_detail': confidence_detail,
        'helical_params':    helical,
        'filepath':          filepath,
        'n_bp':              n_bp,
    }

    if verbose:
        _print_report(result)

    return result


def _print_report(result: Dict) -> None:
    """Print a formatted classification report."""
    filepath        = result['filepath']
    sequence        = result['sequence']
    n_bp            = result['n_bp']
    chain_ids       = result['chain_ids']
    rmsds           = result['rmsds']
    penalties       = result['helical_penalties']
    combined        = result['combined_scores']
    n_matched       = result['n_matched']
    classification  = result['classification']
    confidence      = result['confidence']
    confidence_detail = result['confidence_detail']
    helical         = result['helical_params']

    chain_str = '+'.join(chain_ids) if chain_ids else '?'

    print()
    print("DNA Conformation Classifier")
    print(f"Input:    {Path(filepath).name}")
    print(f"Sequence: {sequence} ({n_bp} bp, chains {chain_str})")
    print()

    # \u2500\u2500 Helical parameters comparison table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if helical['n_bp'] >= 3:
        rise    = helical['rise']
        twist   = helical['twist']
        nu2     = helical['nu2_mean']
        p_rad   = helical['phosphate_radius']

        bp_turn = abs(360.0 / twist) if abs(twist) > 0.5 else float('inf')
        pitch   = rise * bp_turn if bp_turn < 100 else float('nan')

        hand_str = {1: 'right', -1: 'left', 0: '?'}
        hand_label = hand_str.get(helical['handedness'], '?')

        # Pre-define label strings to avoid backslash-in-f-string issues
        sep    = "\u2500"
        lbl_rise  = "Rise (\u00c5/bp)"
        lbl_twist = "Twist (\u00b0/bp)"
        lbl_pitch = "Helix pitch (\u00c5)"
        lbl_nu2   = "\u03bd\u2082 C2'-C3'-C4'-O4' (\u00b0)"
        lbl_prad  = "P-axis radius (\u00c5)"

        # Header row
        print(f"{'Parameter':<22}{'Measured':>10}  {'A-DNA':>8}  {'B-DNA':>8}  {'Z-DNA':>8}")
        print(sep * 22 + sep * 12 + "  " + sep * 10 + "  " + sep * 10 + "  " + sep * 10)
        print(f"{lbl_rise:<22}{rise:>10.2f}  {'2.55':>8}  {'3.38':>8}  {'3.63':>8}")
        print(f"{lbl_twist:<22}{twist:>10.1f}  {'+32.7':>8}  {'+36.0':>8}  {'-30.0':>8}")
        if bp_turn < 100:
            print(f"{'bp/turn':<22}{bp_turn:>10.1f}  {'11.0':>8}  {'10.0':>8}  {'12.0':>8}")
        if not (pitch != pitch):   # isnan check
            print(f"{lbl_pitch:<22}{pitch:>10.1f}  {'28.0':>8}  {'33.8':>8}  {'43.5':>8}")
        print(f"{'Handedness':<22}{hand_label:>10}  {'right':>8}  {'right':>8}  {'left':>8}")
        if nu2 is not None:
            std_str = ("\u00b1" + f"{helical['nu2_std']:.1f}") if helical['nu2_std'] is not None else ""
            print(f"{lbl_nu2:<22}{nu2:>+9.1f}{std_str:<3}  {'+40':>8}  {'-33':>8}  {'~+4':>8}")
        if p_rad > 0:
            print(f"{lbl_prad:<22}{p_rad:>10.1f}  {'~9.5':>8}  {'~9.0':>8}  {'~8.5':>8}")

    # \u2500\u2500 Scores \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print()
    sorted_combined = sorted(combined.items(), key=lambda x: x[1])
    hdr_rmsd = "RMSD (\u00c5)"
    print(f"{'Form':<10}{hdr_rmsd:>10}  {'Helical pen.':>13}  {'Combined':>9}")
    print(sep * 10 + sep * 12 + "  " + sep * 15 + "  " + sep * 11)
    for form_key, score in sorted_combined:
        rmsd = rmsds[form_key]
        pen  = penalties[form_key]
        matched = n_matched.get(form_key, 0)
        marker = "  \u2190 BEST" if form_key == classification else ""
        print(f"{form_key:<10}{rmsd:>10.2f}  {pen:>13.2f}  {score:>9.2f}{marker}  ({matched} atoms)")

    print()
    print(f"Classification: {classification}")
    print(f"Confidence:     {confidence} ({confidence_detail})")
    print()
