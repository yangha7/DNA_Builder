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


def _compute_helical_params(atoms: List[Atom], chain: str = 'A'
                            ) -> Dict[str, float]:
    """
    Estimate helical parameters (rise, twist, handedness) from a structure.

    Uses C1' atoms along the specified chain to estimate:
    - Rise: average Z-displacement between consecutive base pairs
    - Twist: average rotation angle between consecutive base pairs
    - Handedness: +1 for right-handed, -1 for left-handed
    """
    # Collect C1' atoms ordered by residue sequence
    c1_atoms = []
    for atom in atoms:
        if atom.chain_id == chain and atom.name == "C1'":
            c1_atoms.append((atom.residue_seq, np.array([atom.x, atom.y, atom.z])))

    c1_atoms.sort(key=lambda x: x[0])
    coords = [c[1] for c in c1_atoms]

    if len(coords) < 3:
        return {'rise': 0.0, 'twist': 0.0, 'handedness': 0, 'n_bp': len(coords)}

    coords_arr = np.array(coords)
    centroid = coords_arr.mean(axis=0)
    centered = coords_arr - centroid

    # SVD to find principal axis
    _, _, Vt = np.linalg.svd(centered)
    axis = Vt[0]  # First principal component = helix axis

    # Ensure axis points in the direction of increasing residue number
    if np.dot(axis, coords[-1] - coords[0]) < 0:
        axis = -axis

    # Project onto axis and perpendicular plane
    projections = centered @ axis
    perp = centered - np.outer(projections, axis)

    # Compute rise: average displacement along axis between consecutive residues
    rises = []
    for i in range(len(projections) - 1):
        rises.append(projections[i + 1] - projections[i])
    avg_rise = np.mean(rises) if rises else 0.0

    # Compute twist: angle between consecutive perpendicular projections
    twists = []
    for i in range(len(perp) - 1):
        v1 = perp[i]
        v2 = perp[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 0.01 or n2 < 0.01:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Determine sign (handedness) using cross product
        cross = np.cross(v1, v2)
        if np.dot(cross, axis) < 0:
            angle = -angle
        twists.append(angle)

    avg_twist = np.mean(twists) if twists else 0.0
    handedness = 1 if avg_twist > 0 else (-1 if avg_twist < 0 else 0)

    return {
        'rise': abs(avg_rise),
        'twist': avg_twist,
        'handedness': handedness,
        'n_bp': len(coords),
    }


def _assess_confidence(rmsds: Dict[str, float]) -> Tuple[str, str]:
    """
    Assess classification confidence based on RMSD differences.

    Returns (confidence_level, explanation).
    """
    sorted_forms = sorted(rmsds.items(), key=lambda x: x[1])
    best_form, best_rmsd = sorted_forms[0]
    second_form, second_rmsd = sorted_forms[1]

    gap = second_rmsd - best_rmsd

    if best_rmsd < 1.0:
        if gap > 1.5:
            return "High", f"{best_form} RMSD very low and significantly lower than alternatives"
        elif gap > 0.5:
            return "High", f"{best_form} RMSD low and clearly lower than alternatives"
        else:
            return "Medium", f"{best_form} RMSD low but close to {second_form}"
    elif best_rmsd < 2.5:
        if gap > 2.0:
            return "High", f"{best_form} RMSD significantly lower than alternatives"
        elif gap > 1.0:
            return "Medium", f"{best_form} RMSD moderately lower than {second_form}"
        else:
            return "Low", f"{best_form} and {second_form} RMSDs are close"
    else:
        if gap > 2.0:
            return "Medium", f"{best_form} RMSD lower than alternatives but all RMSDs are high"
        else:
            return "Low", f"All RMSDs are high; structure may not match canonical forms well"


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

    # Step 5: Classify
    best_form = min(rmsds, key=rmsds.get)

    # Step 6: Helical parameters
    helical = _compute_helical_params(input_atoms, chain=primary_chain)

    # Confidence assessment
    confidence, confidence_detail = _assess_confidence(rmsds)

    result = {
        'classification': best_form,
        'rmsds': rmsds,
        'sequence': sequence,
        'chain_ids': chain_ids,
        'n_matched': n_matched,
        'confidence': confidence,
        'confidence_detail': confidence_detail,
        'helical_params': helical,
        'filepath': filepath,
        'n_bp': n_bp,
    }

    if verbose:
        _print_report(result)

    return result


def _print_report(result: Dict) -> None:
    """Print a formatted classification report."""
    filepath = result['filepath']
    sequence = result['sequence']
    n_bp = result['n_bp']
    chain_ids = result['chain_ids']
    rmsds = result['rmsds']
    n_matched = result['n_matched']
    classification = result['classification']
    confidence = result['confidence']
    confidence_detail = result['confidence_detail']
    helical = result['helical_params']

    chain_str = '+'.join(chain_ids) if chain_ids else '?'

    print()
    print("DNA Conformation Classifier")
    print(f"Input: {Path(filepath).name}")
    print(f"Sequence: {sequence} ({n_bp} bp, chains {chain_str})")
    print()

    # Sort RMSDs: best first
    sorted_rmsds = sorted(rmsds.items(), key=lambda x: x[1])

    print(f"RMSD vs reference models (strand {chain_ids[0] if chain_ids else 'A'}):")
    for form, rmsd in sorted_rmsds:
        matched = n_matched.get(form, 0)
        marker = "  \u2190 BEST MATCH" if form == classification else ""
        print(f"  {form}: {rmsd:.2f} \u00c5 ({matched} atoms){marker}")

    print()
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence} ({confidence_detail})")

    # Helical parameters
    if helical['n_bp'] >= 3:
        print()
        print("Helical parameters (estimated from C1' atoms):")
        print(f"  Rise:       {helical['rise']:.2f} \u00c5/bp")
        print(f"  Twist:      {helical['twist']:.1f}\u00b0/bp")
        hand_str = {1: "right-handed", -1: "left-handed", 0: "undetermined"}
        print(f"  Handedness: {hand_str.get(helical['handedness'], 'undetermined')}")

        # Compare with canonical values
        print()
        print("Reference helical parameters:")
        print(f"  B-DNA: rise=3.38 \u00c5, twist=36.0\u00b0, right-handed")
        print(f"  A-DNA: rise=2.55 \u00c5, twist=32.7\u00b0, right-handed")
        print(f"  Z-DNA: rise=3.63 \u00c5, twist=-30.0\u00b0, left-handed")
    print()
