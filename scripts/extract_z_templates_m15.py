#!/usr/bin/env python3
"""
Extract Z-DNA templates from 3DNA Model 15 (poly d(GC)) PDB file.

Model 15 gives 0.965 Å RMSD against 1DCG crystal structure (vs 1.065 Å for Model 16).

The PDB file has:
- Chain A residues 1-12: GCGCGCGCGCGC (strand 1)
- Chain B residues 1-12: GCGCGCGCGCGC (strand 2, antiparallel)
- Standard PDB atom names (OP1, OP2 not O1P, O2P)

Z-DNA dinucleotide repeat:
- pos1 (odd residues 1,3,5...): G in syn conformation
- pos2 (even residues 2,4,6...): C in anti conformation
- twist_dinuc = -60.0°, rise_dinuc = 7.250 Å

This script:
1. Parses the PDB file
2. Fits helix axis from P atoms (both strands)
3. Transforms to helix frame (axis along Z)
4. Unwinds each nucleotide to position 0 using the dinucleotide repeat
5. Extracts 4 template sets
6. Derives A and T templates by base superposition
7. Extracts internal coordinates for V2 builder
"""

import numpy as np
import sys
import os

# =============================================================================
# PDB Parsing
# =============================================================================

ATOM_NAME_MAP = {"OP1": "O1P", "OP2": "O2P"}

BACKBONE_NAMES = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]

BASE_ATOM_NAMES = {
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "T": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"],
}

ELEMENT_MAP = {
    "P": "P", "O1P": "O", "O2P": "O", "O5'": "O", "C5'": "C",
    "C4'": "C", "O4'": "O", "C3'": "C", "O3'": "O", "C2'": "C", "C1'": "C",
    "N9": "N", "C8": "C", "N7": "N", "C5": "C", "C6": "C", "O6": "O",
    "N1": "N", "C2": "C", "N2": "N", "N3": "N", "C4": "C",
    "O2": "O", "N4": "N", "N6": "N", "O4": "O", "C5M": "C",
}


def parse_pdb(filepath):
    """Parse PDB file, return dict of (chain, resseq) -> {atom_name: (x,y,z)}."""
    residues = {}
    res_names = {}
    
    with open(filepath) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_seq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            if atom_name in ATOM_NAME_MAP:
                atom_name = ATOM_NAME_MAP[atom_name]
            
            key = (chain_id, res_seq)
            if key not in residues:
                residues[key] = {}
                res_names[key] = res_name
            residues[key][atom_name] = np.array([x, y, z])
    
    return residues, res_names


# =============================================================================
# Geometry functions
# =============================================================================

def rot_z(angle_deg):
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def distance(a, b):
    return np.linalg.norm(a - b)


def angle_deg(a, b, c):
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def dihedral_deg(a, b, c, d):
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.degrees(np.arctan2(y, x))


def kabsch_rotation(P, Q):
    """Compute optimal rotation R minimizing ||R*P - Q||."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    return R


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition."""
    n = P.shape[0]
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    Pr = (R @ Pc.T).T
    diff = Pr - Qc
    return np.sqrt(np.sum(diff ** 2) / n)


# =============================================================================
# Helix axis fitting
# =============================================================================

def fit_helix_axis(p_coords):
    """Fit helix axis from P atom coordinates using SVD."""
    centroid = np.mean(p_coords, axis=0)
    centered = p_coords - centroid
    U, S, Vt = np.linalg.svd(centered)
    axis_dir = Vt[0]
    return centroid, axis_dir


def rotation_to_align_z(axis_dir):
    """Rotation matrix that aligns axis_dir with Z axis."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis_dir, z)
    s = np.linalg.norm(v)
    c = np.dot(axis_dir, z)
    if s < 1e-10:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)


# =============================================================================
# Base superposition for deriving A/T from G/C
# =============================================================================

PURINE_RING_ATOMS = ["N9", "C4", "C5", "N7", "C8"]
PYRIMIDINE_RING_ATOMS = ["N1", "C2", "N3", "C4", "C5", "C6"]

IDEAL_BASES = {
    "A": {
        "N9": [0.000, 0.000, 0.000], "C8": [1.239, 0.678, 0.000],
        "N7": [1.188, 2.014, 0.000], "C5": [-0.076, 2.199, 0.000],
        "C6": [-0.813, 3.393, 0.000], "N6": [-0.186, 4.571, 0.000],
        "N1": [-2.152, 3.340, 0.000], "C2": [-2.714, 2.126, 0.000],
        "N3": [-2.076, 0.963, 0.000], "C4": [-0.739, 1.020, 0.000],
    },
    "G": {
        "N9": [0.000, 0.000, 0.000], "C8": [1.239, 0.678, 0.000],
        "N7": [1.188, 2.014, 0.000], "C5": [-0.076, 2.199, 0.000],
        "C6": [-0.813, 3.393, 0.000], "O6": [-0.186, 4.571, 0.000],
        "N1": [-2.152, 3.340, 0.000], "C2": [-2.714, 2.126, 0.000],
        "N2": [-4.060, 2.076, 0.000], "N3": [-2.076, 0.963, 0.000],
        "C4": [-0.739, 1.020, 0.000],
    },
    "C": {
        "N1": [0.000, 0.000, 0.000], "C2": [-1.268, -0.530, 0.000],
        "O2": [-1.360, -1.756, 0.000], "N3": [-2.310, 0.316, 0.000],
        "C4": [-2.148, 1.641, 0.000], "N4": [-3.210, 2.434, 0.000],
        "C5": [-0.862, 2.199, 0.000], "C6": [0.155, 1.361, 0.000],
    },
    "T": {
        "N1": [0.000, 0.000, 0.000], "C2": [-1.268, -0.530, 0.000],
        "O2": [-1.360, -1.756, 0.000], "N3": [-2.310, 0.316, 0.000],
        "C4": [-2.148, 1.641, 0.000], "O4": [-3.210, 2.434, 0.000],
        "C5": [-0.862, 2.199, 0.000], "C5M": [-0.700, 3.685, 0.000],
        "C6": [0.155, 1.361, 0.000],
    },
}


def derive_base_template(source_residue, source_base, target_base):
    """
    Derive target base atoms from source base by superimposing common ring atoms.
    For same-type (purine->purine, pyrimidine->pyrimidine): use ring atoms.
    For cross-type: use glycosidic bond frame (C1', N9/N1 direction).
    """
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}
    
    if source_base in purines and target_base in purines:
        common_atoms = PURINE_RING_ATOMS
    elif source_base in pyrimidines and target_base in pyrimidines:
        common_atoms = PYRIMIDINE_RING_ATOMS
    else:
        # Cross-type: use sugar frame
        C1 = source_residue.get("C1'")
        O4 = source_residue.get("O4'")
        C2p = source_residue.get("C2'")
        
        src_glyc = "N9" if source_base in purines else "N1"
        src_N = source_residue.get(src_glyc)
        
        if any(x is None for x in [C1, O4, C2p, src_N]):
            raise ValueError("Missing atoms for cross-type derivation")
        
        # Build local frame at C1'
        x_axis = src_N - C1
        x_axis = x_axis / np.linalg.norm(x_axis)
        v_o4 = O4 - C1
        z_axis = np.cross(x_axis, v_o4)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_local = np.column_stack([x_axis, y_axis, z_axis])
        
        glyc_dist = np.linalg.norm(src_N - C1)
        tgt_glyc = "N9" if target_base in purines else "N1"
        tgt_N_pos = C1 + x_axis * glyc_dist
        ideal_tgt_N = np.array(IDEAL_BASES[target_base][tgt_glyc])
        
        target_atoms = {}
        for atom_name in BASE_ATOM_NAMES[target_base]:
            if atom_name in IDEAL_BASES[target_base]:
                ideal_pos = np.array(IDEAL_BASES[target_base][atom_name])
                rel_pos = ideal_pos - ideal_tgt_N
                target_atoms[atom_name] = tgt_N_pos + R_local @ rel_pos
        return target_atoms
    
    # Same-type: superimpose common ring atoms
    src_common = np.array([source_residue[a] for a in common_atoms])
    ideal_src = np.array([IDEAL_BASES[source_base][a] for a in common_atoms])
    ideal_tgt = np.array([IDEAL_BASES[target_base][a] for a in common_atoms])
    
    src_centroid = src_common.mean(axis=0)
    ideal_src_centroid = ideal_src.mean(axis=0)
    R = kabsch_rotation(ideal_src - ideal_src_centroid, src_common - src_centroid)
    
    target_atoms = {}
    for atom_name in BASE_ATOM_NAMES[target_base]:
        if atom_name in IDEAL_BASES[target_base]:
            ideal_pos = np.array(IDEAL_BASES[target_base][atom_name])
            target_atoms[atom_name] = R @ (ideal_pos - ideal_src_centroid) + src_centroid
    return target_atoms


# =============================================================================
# Main extraction
# =============================================================================

def main():
    pdb_file = "3DNA_Model15_ZDNA.pdb"
    if not os.path.exists(pdb_file):
        print(f"ERROR: {pdb_file} not found")
        sys.exit(1)
    
    print(f"Parsing {pdb_file}...")
    residues, res_names = parse_pdb(pdb_file)
    
    # Collect P atoms for axis fitting
    p_coords = []
    for (chain, resseq), atoms in sorted(residues.items()):
        if "P" in atoms:
            p_coords.append(atoms["P"])
    p_coords = np.array(p_coords)
    print(f"Found {len(p_coords)} P atoms")
    
    # Fit helix axis
    centroid, axis_dir = fit_helix_axis(p_coords)
    print(f"Helix axis direction: {axis_dir}")
    print(f"Helix axis centroid: {centroid}")
    
    # Align axis with Z
    R_align = rotation_to_align_z(axis_dir)
    
    # Transform all atoms to helix frame
    transformed = {}
    for key, atoms in residues.items():
        transformed[key] = {}
        for atom_name, coords in atoms.items():
            transformed[key][atom_name] = R_align @ (coords - centroid)
    
    # Check strand A P z-coordinates to determine helix direction
    strand_a_pz = []
    for resseq in range(1, 13):
        key = ("A", resseq)
        if key in transformed and "P" in transformed[key]:
            strand_a_pz.append((resseq, transformed[key]["P"][2]))
    
    print("\nStrand A P z-coordinates:")
    for resseq, z in strand_a_pz:
        print(f"  Res {resseq}: z = {z:.3f}")
    
    # The helix progresses in -Z direction (z decreases with increasing resseq)
    # Measure actual rise and twist per dinucleotide
    rises = []
    twists = []
    for i in range(0, len(strand_a_pz) - 2, 2):
        r1, z1 = strand_a_pz[i]
        r2, z2 = strand_a_pz[i + 2]
        rises.append(z2 - z1)
        
        p1 = transformed[("A", r1)]["P"]
        p2 = transformed[("A", r2)]["P"]
        a1 = np.degrees(np.arctan2(p1[1], p1[0]))
        a2 = np.degrees(np.arctan2(p2[1], p2[0]))
        tw = a2 - a1
        while tw > 180: tw -= 360
        while tw < -180: tw += 360
        twists.append(tw)
    
    measured_rise = np.mean(rises)
    measured_twist = np.mean(twists)
    print(f"\nMeasured rise per dinucleotide: {measured_rise:.3f} Å")
    print(f"Measured twist per dinucleotide: {measured_twist:.1f}°")
    
    # Use measured values for unwinding
    # rise_dinuc is negative (helix goes in -Z), twist is positive (left-handed in this frame)
    # For the builder, we want rise=+7.250 and twist=-60.0
    # So we need to flip the Z axis to make the helix go in +Z direction
    
    # Actually, let's just unwind using the measured parameters
    # Dinucleotide index for strand A: dinuc_idx = (resseq - 1) // 2
    # To unwind dinuc n to position 0: rotate by -n*twist and translate by -n*rise
    
    unwound = {}
    
    # Unwind strand A
    for resseq in range(1, 13):
        key = ("A", resseq)
        dinuc_idx = (resseq - 1) // 2
        
        R_unwind = rot_z(-dinuc_idx * measured_twist)
        t_unwind = np.array([0.0, 0.0, -dinuc_idx * measured_rise])
        
        unwound[key] = {}
        for atom_name, coords in transformed[key].items():
            unwound[key][atom_name] = R_unwind @ coords + t_unwind
    
    # Unwind strand B
    # B:resseq partners with A:(13-resseq)
    # So B:resseq has dinuc_idx = (13 - resseq - 1) // 2 = (12 - resseq) // 2
    for resseq in range(1, 13):
        key = ("B", resseq)
        partner_a = 13 - resseq
        dinuc_idx = (partner_a - 1) // 2
        
        R_unwind = rot_z(-dinuc_idx * measured_twist)
        t_unwind = np.array([0.0, 0.0, -dinuc_idx * measured_rise])
        
        unwound[key] = {}
        for atom_name, coords in transformed[key].items():
            unwound[key][atom_name] = R_unwind @ coords + t_unwind
    
    # Check unwinding quality
    print("\n--- Checking unwinding quality ---")
    
    # All strand A pos1 (G) residues should now overlap
    s1_pos1_keys = [("A", r) for r in range(1, 13, 2)]
    s1_pos2_keys = [("A", r) for r in range(2, 13, 2)]
    
    for label, keys, base in [("S1_POS1", s1_pos1_keys, "G"), ("S1_POS2", s1_pos2_keys, "C")]:
        atom_names = BACKBONE_NAMES + BASE_ATOM_NAMES[base]
        max_dev = 0.0
        ref_key = keys[0]
        for key in keys[1:]:
            for aname in atom_names:
                if aname in unwound[ref_key] and aname in unwound[key]:
                    dev = np.linalg.norm(unwound[key][aname] - unwound[ref_key][aname])
                    max_dev = max(max_dev, dev)
        print(f"  {label}: max deviation from first = {max_dev:.4f} Å")
    
    # Average the unwound templates
    def average_template(keys, base_type, label):
        atom_names = BACKBONE_NAMES + BASE_ATOM_NAMES[base_type]
        avg = {}
        counts = {}
        for key in keys:
            atoms = unwound.get(key, {})
            for aname in atom_names:
                if aname in atoms:
                    if aname not in avg:
                        avg[aname] = np.zeros(3)
                        counts[aname] = 0
                    avg[aname] += atoms[aname]
                    counts[aname] += 1
        for aname in avg:
            avg[aname] /= counts[aname]
        
        max_dev = 0.0
        for key in keys:
            atoms = unwound.get(key, {})
            for aname in atom_names:
                if aname in atoms and aname in avg:
                    dev = np.linalg.norm(atoms[aname] - avg[aname])
                    max_dev = max(max_dev, dev)
        print(f"  {label}: max deviation from mean = {max_dev:.4f} Å ({len(keys)} residues)")
        return avg
    
    print("\n--- Averaging unwound templates ---")
    
    # Strand 2 pos1 partners: B:12,10,8,6,4,2 (C residues)
    s2_pos1_keys = [("B", r) for r in range(12, 0, -2)]
    # Strand 2 pos2 partners: B:11,9,7,5,3,1 (G residues)
    s2_pos2_keys = [("B", r) for r in range(11, 0, -2)]
    
    s1_pos1_avg = average_template(s1_pos1_keys, "G", "S1_POS1 (G)")
    s1_pos2_avg = average_template(s1_pos2_keys, "C", "S1_POS2 (C)")
    s2_pos1_avg = average_template(s2_pos1_keys, "C", "S2_POS1 (C)")
    s2_pos2_avg = average_template(s2_pos2_keys, "G", "S2_POS2 (G)")
    
    # Now we need to flip Z so the helix goes in +Z direction
    # In the builder, rise is positive and twist is negative
    # Currently our templates have the helix going in -Z
    # We need to negate Z coordinates of all templates
    # But we also need to be careful about the handedness
    
    # Actually, let's check what the existing Model 16 templates look like
    # In the existing code, Z_STRAND1_POS1["G"] has P at z=-16.607
    # and the helix goes in -Z direction (rise is applied as +7.250 along Z
    # but twist is -60°). Let me check the builder code...
    
    # Looking at builder.py, for Z-DNA it uses:
    #   rise_per_step = rise_dinuc / 2  (for each nucleotide)
    #   twist_per_step = twist_dinuc / 2  (for each nucleotide)
    # Wait, actually it uses the dinucleotide repeat differently.
    # Let me check the builder more carefully.
    
    # For now, let's just output the templates as-is and see if they work.
    # The key insight is that the templates should be at "position 0" of the
    # dinucleotide repeat, and the builder applies the helical screw to generate
    # subsequent positions.
    
    # Derive A and T templates
    print("\n--- Deriving A and T templates ---")
    
    def make_derived(avg_dict, src_base, tgt_base, label):
        tgt_base_atoms = derive_base_template(avg_dict, src_base, tgt_base)
        result = {}
        for aname in BACKBONE_NAMES:
            if aname in avg_dict:
                result[aname] = avg_dict[aname].copy()
        for aname in BASE_ATOM_NAMES[tgt_base]:
            if aname in tgt_base_atoms:
                result[aname] = tgt_base_atoms[aname]
        print(f"  {label}: {len(tgt_base_atoms)} base atoms derived")
        return result
    
    # S1_POS1: G backbone, derive A (purine->purine)
    s1_pos1_A = make_derived(s1_pos1_avg, "G", "A", "S1_POS1 A<-G")
    # S1_POS1: G backbone, derive C (cross-type)
    s1_pos1_C = make_derived(s1_pos1_avg, "G", "C", "S1_POS1 C<-G")
    # S1_POS1: G backbone, derive T (cross-type)
    s1_pos1_T = make_derived(s1_pos1_avg, "G", "T", "S1_POS1 T<-G")
    
    # S1_POS2: C backbone, derive T (pyrimidine->pyrimidine)
    s1_pos2_T = make_derived(s1_pos2_avg, "C", "T", "S1_POS2 T<-C")
    # S1_POS2: C backbone, derive A (cross-type)
    s1_pos2_A = make_derived(s1_pos2_avg, "C", "A", "S1_POS2 A<-C")
    # S1_POS2: C backbone, derive G (cross-type)
    s1_pos2_G = make_derived(s1_pos2_avg, "C", "G", "S1_POS2 G<-C")
    
    # S2_POS1: C backbone, derive T
    s2_pos1_T = make_derived(s2_pos1_avg, "C", "T", "S2_POS1 T<-C")
    # S2_POS1: C backbone, derive A (cross-type)
    s2_pos1_A = make_derived(s2_pos1_avg, "C", "A", "S2_POS1 A<-C")
    # S2_POS1: C backbone, derive G (cross-type)
    s2_pos1_G = make_derived(s2_pos1_avg, "C", "G", "S2_POS1 G<-C")
    
    # S2_POS2: G backbone, derive A
    s2_pos2_A = make_derived(s2_pos2_avg, "G", "A", "S2_POS2 A<-G")
    # S2_POS2: G backbone, derive C (cross-type)
    s2_pos2_C = make_derived(s2_pos2_avg, "G", "C", "S2_POS2 C<-G")
    # S2_POS2: G backbone, derive T (cross-type)
    s2_pos2_T = make_derived(s2_pos2_avg, "G", "T", "S2_POS2 T<-G")
    
    # ==========================================================================
    # Verify by reconstructing the original structure
    # ==========================================================================
    print("\n--- Verification: reconstruct original structure ---")
    
    # Reconstruct strand A residue 3 (G, pos1, dinuc_idx=1)
    dinuc_idx = 1
    R_fwd = rot_z(dinuc_idx * measured_twist)
    t_fwd = np.array([0.0, 0.0, dinuc_idx * measured_rise])
    
    orig_A3 = transformed[("A", 3)]
    recon_A3 = {}
    for aname, coords in s1_pos1_avg.items():
        recon_A3[aname] = R_fwd @ coords + t_fwd
    
    max_err = 0.0
    for aname in BACKBONE_NAMES + BASE_ATOM_NAMES["G"]:
        if aname in orig_A3 and aname in recon_A3:
            err = np.linalg.norm(orig_A3[aname] - recon_A3[aname])
            max_err = max(max_err, err)
    print(f"  Reconstruct A:3 (G, pos1): max error = {max_err:.4f} Å")
    
    # Reconstruct strand A residue 4 (C, pos2, dinuc_idx=1)
    orig_A4 = transformed[("A", 4)]
    recon_A4 = {}
    for aname, coords in s1_pos2_avg.items():
        recon_A4[aname] = R_fwd @ coords + t_fwd
    
    max_err = 0.0
    for aname in BACKBONE_NAMES + BASE_ATOM_NAMES["C"]:
        if aname in orig_A4 and aname in recon_A4:
            err = np.linalg.norm(orig_A4[aname] - recon_A4[aname])
            max_err = max(max_err, err)
    print(f"  Reconstruct A:4 (C, pos2): max error = {max_err:.4f} Å")
    
    # ==========================================================================
    # Now output the templates
    # ==========================================================================
    
    # The builder expects templates in a specific coordinate frame.
    # Looking at the existing Model 16 templates, the Z coordinates are negative
    # and the helix is built by applying twist_dinuc=-60° and rise_dinuc=7.250
    # per dinucleotide step.
    
    # In the builder (builder.py), for Z-DNA:
    # - Each dinucleotide step i: rotate by twist_dinuc*i, translate by rise_dinuc*i along Z
    # - But the builder actually handles pos1 and pos2 separately within each dinucleotide
    
    # Let me check how the builder uses these templates by reading builder.py more carefully
    # For now, output the templates as extracted (in the helix frame)
    
    def fmt_template(avg_dict, base_type):
        """Format template as list of tuples."""
        atom_names = BACKBONE_NAMES + BASE_ATOM_NAMES[base_type]
        lines = []
        for aname in atom_names:
            if aname in avg_dict:
                c = avg_dict[aname]
                elem = ELEMENT_MAP.get(aname, aname[0])
                lines.append(
                    f'        ("{aname}",{" " * (5-len(aname))} "{elem}",{" " * 5}'
                    f'{c[0]:8.3f},{" " * 4}{c[1]:8.3f},{" " * 4}{c[2]:8.3f}),'
                )
        return lines
    
    # Write output to file
    output_file = "z_templates_m15_output.py"
    with open(output_file, "w") as f:
        f.write("# Z-DNA Templates extracted from Model 15 (poly d(GC))\n")
        f.write(f"# Helix frame: axis along Z, centroid at origin\n")
        f.write(f"# Measured rise/dinuc = {measured_rise:.3f} Å, twist/dinuc = {measured_twist:.1f}°\n")
        f.write(f"# For builder: rise_dinuc = 7.250, twist_dinuc = -60.0\n\n")
        
        # Z_STRAND1_POS1
        f.write("Z_STRAND1_POS1 = {\n")
        for base, template in [("A", s1_pos1_A), ("G", s1_pos1_avg),
                                ("C", s1_pos1_C), ("T", s1_pos1_T)]:
            f.write(f'    "{base}": [\n')
            for line in fmt_template(template, base):
                f.write(line + "\n")
            f.write("    ],\n")
        f.write("}\n\n")
        
        # Z_STRAND1_POS2
        f.write("Z_STRAND1_POS2 = {\n")
        for base, template in [("C", s1_pos2_avg), ("T", s1_pos2_T),
                                ("A", s1_pos2_A), ("G", s1_pos2_G)]:
            f.write(f'    "{base}": [\n')
            for line in fmt_template(template, base):
                f.write(line + "\n")
            f.write("    ],\n")
        f.write("}\n\n")
        
        # Z_STRAND2_POS1
        f.write("Z_STRAND2_POS1 = {\n")
        for base, template in [("C", s2_pos1_avg), ("T", s2_pos1_T),
                                ("A", s2_pos1_A), ("G", s2_pos1_G)]:
            f.write(f'    "{base}": [\n')
            for line in fmt_template(template, base):
                f.write(line + "\n")
            f.write("    ],\n")
        f.write("}\n\n")
        
        # Z_STRAND2_POS2
        f.write("Z_STRAND2_POS2 = {\n")
        for base, template in [("A", s2_pos2_A), ("G", s2_pos2_avg),
                                ("C", s2_pos2_C), ("T", s2_pos2_T)]:
            f.write(f'    "{base}": [\n')
            for line in fmt_template(template, base):
                f.write(line + "\n")
            f.write("    ],\n")
        f.write("}\n")
    
    print(f"\nTemplates written to {output_file}")
    
    # ==========================================================================
    # Extract internal coordinates
    # ==========================================================================
    print("\n--- Extracting internal coordinates ---")
    
    def extract_ic(keys, base_type, label):
        """Extract internal coordinates from unwound residues."""
        torsions = {k: [] for k in [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
            "O5'-C5'-C4'-O4'", "C5'-C4'-C3'-C2'", "C4'-C3'-C2'-C1'",
            "C5'-O5'-P-O1P", "C5'-O5'-P-O2P",
        ]}
        bl = {k: [] for k in [
            "P-O5'", "O5'-C5'", "C5'-C4'", "C4'-O4'", "C4'-C3'",
            "C3'-O3'", "C3'-C2'", "C2'-C1'", "C1'-O4'", "O3'-P",
            "P-O1P", "P-O2P", "C1'-N9", "C1'-N1",
        ]}
        ba = {k: [] for k in [
            "O3'-P-O5'", "P-O5'-C5'", "O5'-C5'-C4'",
            "C5'-C4'-C3'", "C5'-C4'-O4'", "C3'-C4'-O4'",
            "C4'-C3'-O3'", "C4'-C3'-C2'", "C3'-C2'-C1'",
            "C2'-C1'-O4'", "C4'-O4'-C1'",
            "O3'-P-O1P", "O3'-P-O2P", "O1P-P-O2P",
            "O1P-P-O5'", "O2P-P-O5'",
            "C2'-C1'-N9", "C2'-C1'-N1",
            "O4'-C1'-N9", "O4'-C1'-N1",
            "C3'-O3'-P+",
        ]}
        sp = {"nu2": [], "P_phase": []}
        
        for key in keys:
            chain, resseq = key
            atoms = unwound.get(key, {})
            if not atoms:
                continue
            
            P = atoms.get("P")
            O1P = atoms.get("O1P")
            O2P = atoms.get("O2P")
            O5 = atoms.get("O5'")
            C5 = atoms.get("C5'")
            C4 = atoms.get("C4'")
            O4 = atoms.get("O4'")
            C3 = atoms.get("C3'")
            O3 = atoms.get("O3'")
            C2p = atoms.get("C2'")
            C1 = atoms.get("C1'")
            
            glyc_name = "N9" if base_type in ("A", "G") else "N1"
            chi_ref = "C4" if base_type in ("A", "G") else "C2"
            N_glyc = atoms.get(glyc_name)
            C_chi = atoms.get(chi_ref)
            
            if any(x is None for x in [P, O1P, O2P, O5, C5, C4, O4, C3, O3, C2p, C1]):
                continue
            
            prev_O3 = unwound.get((chain, resseq - 1), {}).get("O3'")
            next_P = unwound.get((chain, resseq + 1), {}).get("P")
            next_O5 = unwound.get((chain, resseq + 1), {}).get("O5'")
            
            if prev_O3 is not None:
                torsions["alpha"].append(dihedral_deg(prev_O3, P, O5, C5))
            torsions["beta"].append(dihedral_deg(P, O5, C5, C4))
            torsions["gamma"].append(dihedral_deg(O5, C5, C4, C3))
            torsions["delta"].append(dihedral_deg(C5, C4, C3, O3))
            if next_P is not None:
                torsions["epsilon"].append(dihedral_deg(C4, C3, O3, next_P))
            if next_P is not None and next_O5 is not None:
                torsions["zeta"].append(dihedral_deg(C3, O3, next_P, next_O5))
            if N_glyc is not None and C_chi is not None:
                torsions["chi"].append(dihedral_deg(O4, C1, N_glyc, C_chi))
            
            torsions["O5'-C5'-C4'-O4'"].append(dihedral_deg(O5, C5, C4, O4))
            torsions["C5'-C4'-C3'-C2'"].append(dihedral_deg(C5, C4, C3, C2p))
            torsions["C4'-C3'-C2'-C1'"].append(dihedral_deg(C4, C3, C2p, C1))
            torsions["C5'-O5'-P-O1P"].append(dihedral_deg(C5, O5, P, O1P))
            torsions["C5'-O5'-P-O2P"].append(dihedral_deg(C5, O5, P, O2P))
            
            bl["P-O5'"].append(distance(P, O5))
            bl["O5'-C5'"].append(distance(O5, C5))
            bl["C5'-C4'"].append(distance(C5, C4))
            bl["C4'-O4'"].append(distance(C4, O4))
            bl["C4'-C3'"].append(distance(C4, C3))
            bl["C3'-O3'"].append(distance(C3, O3))
            bl["C3'-C2'"].append(distance(C3, C2p))
            bl["C2'-C1'"].append(distance(C2p, C1))
            bl["C1'-O4'"].append(distance(C1, O4))
            bl["P-O1P"].append(distance(P, O1P))
            bl["P-O2P"].append(distance(P, O2P))
            if prev_O3 is not None:
                bl["O3'-P"].append(distance(prev_O3, P))
            if N_glyc is not None:
                bl[f"C1'-{glyc_name}"].append(distance(C1, N_glyc))
            
            ba["P-O5'-C5'"].append(angle_deg(P, O5, C5))
            ba["O5'-C5'-C4'"].append(angle_deg(O5, C5, C4))
            ba["C5'-C4'-C3'"].append(angle_deg(C5, C4, C3))
            ba["C5'-C4'-O4'"].append(angle_deg(C5, C4, O4))
            ba["C3'-C4'-O4'"].append(angle_deg(C3, C4, O4))
            ba["C4'-C3'-O3'"].append(angle_deg(C4, C3, O3))
            ba["C4'-C3'-C2'"].append(angle_deg(C4, C3, C2p))
            ba["C3'-C2'-C1'"].append(angle_deg(C3, C2p, C1))
            ba["C2'-C1'-O4'"].append(angle_deg(C2p, C1, O4))
            ba["C4'-O4'-C1'"].append(angle_deg(C4, O4, C1))
            ba["O1P-P-O2P"].append(angle_deg(O1P, P, O2P))
            ba["O1P-P-O5'"].append(angle_deg(O1P, P, O5))
            ba["O2P-P-O5'"].append(angle_deg(O2P, P, O5))
            if prev_O3 is not None:
                ba["O3'-P-O5'"].append(angle_deg(prev_O3, P, O5))
                ba["O3'-P-O1P"].append(angle_deg(prev_O3, P, O1P))
                ba["O3'-P-O2P"].append(angle_deg(prev_O3, P, O2P))
            if N_glyc is not None:
                ba[f"C2'-C1'-{glyc_name}"].append(angle_deg(C2p, C1, N_glyc))
                ba[f"O4'-C1'-{glyc_name}"].append(angle_deg(O4, C1, N_glyc))
            if next_P is not None:
                ba["C3'-O3'-P+"].append(angle_deg(C3, O3, next_P))
            
            nu0 = dihedral_deg(C4, O4, C1, C2p)
            nu1 = dihedral_deg(O4, C1, C2p, C3)
            nu2 = dihedral_deg(C1, C2p, C3, C4)
            nu3 = dihedral_deg(C2p, C3, C4, O4)
            nu4 = dihedral_deg(C3, C4, O4, C1)
            sp["nu2"].append(nu2)
            
            sin36 = np.sin(np.radians(36))
            sin72 = np.sin(np.radians(72))
            A_val = nu2 * (sin36 + sin72)
            B_val = nu4 + nu1 - nu3 - nu0
            if abs(A_val) > 1e-10:
                P_phase = np.degrees(np.arctan2(B_val, 2.0 * A_val))
            else:
                P_phase = 0.0
            sp["P_phase"].append(P_phase)
        
        print(f"\n  {label} ({base_type}):")
        print("    Torsions:")
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi",
                      "O5'-C5'-C4'-O4'", "C5'-C4'-C3'-C2'", "C4'-C3'-C2'-C1'",
                      "C5'-O5'-P-O1P", "C5'-O5'-P-O2P"]:
            if torsions[name]:
                print(f"      {name}: {np.mean(torsions[name]):.1f}° (±{np.std(torsions[name]):.1f}°)")
        print("    Bond lengths:")
        for name in sorted(bl.keys()):
            if bl[name]:
                print(f"      {name}: {np.mean(bl[name]):.3f} Å")
        print("    Bond angles:")
        for name in sorted(ba.keys()):
            if ba[name]:
                print(f"      {name}: {np.mean(ba[name]):.1f}°")
        print("    Sugar pucker:")
        for name in sorted(sp.keys()):
            if sp[name]:
                print(f"      {name}: {np.mean(sp[name]):.1f}°")
        
        return torsions, bl, ba, sp
    
    # Use middle residues for IC extraction
    s1_pos1_mid = [("A", r) for r in [3, 5, 7, 9]]
    s1_pos2_mid = [("A", r) for r in [4, 6, 8, 10]]
    
    pos1_t, pos1_bl, pos1_ba, pos1_sp = extract_ic(s1_pos1_mid, "G", "POS1")
    pos2_t, pos2_bl, pos2_ba, pos2_sp = extract_ic(s1_pos2_mid, "C", "POS2")
    
    # Write IC output
    with open(output_file, "a") as f:
        f.write("\n\n# Internal Coordinates\n")
        
        for pos_label, t, bl_d, ba_d, sp_d, base in [
            ("POS1", pos1_t, pos1_bl, pos1_ba, pos1_sp, "G"),
            ("POS2", pos2_t, pos2_bl, pos2_ba, pos2_sp, "C"),
        ]:
            chi_name = "chi_pur" if base in ("A", "G") else "chi_pyr"
            other_chi = "chi_pyr" if chi_name == "chi_pur" else "chi_pur"
            
            f.write(f"\nZ_DNA_{pos_label}_PARAMS = {{\n")
            f.write('    "torsion_angles": {\n')
            for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
                if t[name]:
                    f.write(f'        "{name}": {np.mean(t[name]):.1f},\n')
            if t["chi"]:
                f.write(f'        "{chi_name}": {np.mean(t["chi"]):.1f},\n')
                f.write(f'        "{other_chi}": {np.mean(t["chi"]):.1f},\n')
            f.write('        # Sugar ring dihedrals\n')
            for name in ["O5'-C5'-C4'-O4'", "C5'-C4'-C3'-C2'", "C4'-C3'-C2'-C1'"]:
                if t[name]:
                    f.write(f'        "{name}": {np.mean(t[name]):.1f},\n')
            f.write('        # Phosphate oxygen dihedrals\n')
            for name in ["C5'-O5'-P-O1P", "C5'-O5'-P-O2P"]:
                if t[name]:
                    f.write(f'        "{name}": {np.mean(t[name]):.1f},\n')
            f.write('    },\n')
            
            f.write('    "bond_lengths": {\n')
            for name in sorted(bl_d.keys()):
                if bl_d[name]:
                    f.write(f'        "{name}": {np.mean(bl_d[name]):.3f},\n')
            f.write('    },\n')
            
            f.write('    "bond_angles": {\n')
            angle_map = {
                "C3'-C2'-C1'": "C1'-C2'-C3'",
                "C4'-C3'-C2'": "C2'-C3'-C4'",
                "C3'-C4'-O4'": "C3'-C4'-O4'",
                "C4'-C3'-O3'": "C4'-C3'-O3'",
                "C4'-O4'-C1'": "C4'-O4'-C1'",
                "C5'-C4'-C3'": "C5'-C4'-C3'",
                "C5'-C4'-O4'": "C5'-C4'-O4'",
                "O1P-P-O2P": "O1P-P-O2P",
                "O1P-P-O5'": "O1P-P-O5'",
                "O2P-P-O5'": "O2P-P-O5'",
                "O3'-P-O1P": "O3'-P-O1P",
                "O3'-P-O2P": "O3'-P-O2P",
                "O3'-P-O5'": "O3'-P-O5'",
                "C2'-C1'-O4'": "O4'-C1'-C2'",
                "O5'-C5'-C4'": "O5'-C5'-C4'",
                "P-O5'-C5'": "P-O5'-C5'",
                "C2'-C1'-N9": "C2'-C1'-N9",
                "C2'-C1'-N1": "C2'-C1'-N1",
                "O4'-C1'-N9": "O4'-C1'-N9",
                "O4'-C1'-N1": "O4'-C1'-N1",
                "C3'-O3'-P+": "C3'-O3'-P+",
            }
            printed = set()
            for ext_name, out_name in sorted(angle_map.items(), key=lambda x: x[1]):
                if ext_name in ba_d and ba_d[ext_name] and out_name not in printed:
                    f.write(f'        "{out_name}": {np.mean(ba_d[ext_name]):.1f},\n')
                    printed.add(out_name)
            f.write('    },\n')
            
            f.write('    "sugar_pucker": {\n')
            if sp_d["P_phase"]:
                f.write(f'        "P": {np.mean(sp_d["P_phase"]):.1f},\n')
            if sp_d["nu2"]:
                f.write(f'        "tau_m": {abs(np.mean(sp_d["nu2"])):.1f},\n')
            f.write('    },\n')
            f.write('}\n')
    
    # Extract cross-strand and strand2 backbone dihedrals
    with open(output_file, "a") as f:
        # Strand 2 backbone dihedrals
        f.write("\n# Strand 2 backbone dihedrals\n")
        f.write("STRAND2_BACKBONE_DIHEDRALS_Z = {\n")
        
        for label, keys, base in [
            ("Z_pos1", [("B", r) for r in [4, 6, 8, 10]], "C"),
            ("Z_pos2", [("B", r) for r in [3, 5, 7, 9]], "G"),
        ]:
            c2_c4_c3_o3 = []
            o4_c3_c4_c5 = []
            for key in keys:
                atoms = unwound.get(key, {})
                C2p = atoms.get("C2'")
                C3p = atoms.get("C3'")
                C4p = atoms.get("C4'")
                O3p = atoms.get("O3'")
                O4p = atoms.get("O4'")
                C5p = atoms.get("C5'")
                if all(x is not None for x in [C2p, C3p, C4p, O3p, O4p, C5p]):
                    c2_c4_c3_o3.append(dihedral_deg(C2p, C4p, C3p, O3p))
                    o4_c3_c4_c5.append(dihedral_deg(O4p, C3p, C4p, C5p))
            
            if c2_c4_c3_o3:
                f.write(f'    "{label}": {{\n')
                f.write(f'        "C2\'-C4\'-C3\'-O3\'": {np.mean(c2_c4_c3_o3):.1f},\n')
                f.write(f'        "O4\'-C3\'-C4\'-C5\'": {np.mean(o4_c3_c4_c5):.1f},\n')
                f.write(f'    }},\n')
        f.write("}\n")
        
        # Cross-strand parameters
        f.write("\n# Cross-strand parameters\n")
        
        s1_pos1_keys = [("A", r) for r in [3, 5, 7, 9]]
        s2_pos1_keys = [("B", 13-r) for r in [3, 5, 7, 9]]
        s1_pos2_keys = [("A", r) for r in [4, 6, 8, 10]]
        s2_pos2_keys = [("B", 13-r) for r in [4, 6, 8, 10]]
        
        for cs_label, s1k, s2k, s1_base, s2_base in [
            ("Z_CROSS_STRAND_POS1", s1_pos1_keys, s2_pos1_keys, "G", "C"),
            ("Z_CROSS_STRAND_POS2", s1_pos2_keys, s2_pos2_keys, "C", "G"),
        ]:
            f.write(f"\n{cs_label} = {{\n")
            
            if s1_base in ("A", "G"):
                ref_atom, angle_ref, dihedral_ref = "N1", "C2", "C6"
            else:
                ref_atom, angle_ref, dihedral_ref = "N3", "C2", "C4"
            
            results = {k: [] for k in [
                "C1'_dist", "C1'_angle", "C1'_dihedral",
                "O4'_angle", "O4'_dihedral", "C2'_angle", "C2'_dihedral",
            ]}
            
            for s1, s2 in zip(s1k, s2k):
                s1a = unwound.get(s1, {})
                s2a = unwound.get(s2, {})
                ref = s1a.get(ref_atom)
                aref = s1a.get(angle_ref)
                dref = s1a.get(dihedral_ref)
                s2_C1 = s2a.get("C1'")
                s2_O4 = s2a.get("O4'")
                s2_C2 = s2a.get("C2'")
                
                if any(x is None for x in [ref, aref, dref, s2_C1, s2_O4, s2_C2]):
                    continue
                
                results["C1'_dist"].append(distance(ref, s2_C1))
                results["C1'_angle"].append(angle_deg(aref, ref, s2_C1))
                results["C1'_dihedral"].append(dihedral_deg(dref, aref, ref, s2_C1))
                results["O4'_angle"].append(angle_deg(ref, s2_C1, s2_O4))
                results["O4'_dihedral"].append(dihedral_deg(aref, ref, s2_C1, s2_O4))
                results["C2'_angle"].append(angle_deg(ref, s2_C1, s2_C2))
                results["C2'_dihedral"].append(dihedral_deg(aref, ref, s2_C1, s2_C2))
            
            pair_key = f"{s1_base}->{s2_base}"
            f.write(f'    "{pair_key}": {{\n')
            f.write(f'        "ref_atom": "{ref_atom}", "angle_ref": "{angle_ref}", "dihedral_ref": "{dihedral_ref}",\n')
            for param in ["C1'_dist", "C1'_angle", "C1'_dihedral",
                           "O4'_angle", "O4'_dihedral", "C2'_angle", "C2'_dihedral"]:
                if results[param]:
                    f.write(f'        "{param}": {np.mean(results[param]):.1f},\n')
            f.write('    },\n')
            f.write("}\n")
        
        # Base templates for internal_coords.py
        f.write("\n# Base templates\n")
        for pos_label, avg_dict, base in [("POS1", s1_pos1_avg, "G"), ("POS2", s1_pos2_avg, "C")]:
            f.write(f'\nZ_BASE_TEMPLATES_{pos_label} = {{\n')
            
            all_bases = {"G": s1_pos1_avg, "A": s1_pos1_A, "C": s1_pos1_C, "T": s1_pos1_T} if pos_label == "POS1" \
                else {"C": s1_pos2_avg, "T": s1_pos2_T, "A": s1_pos2_A, "G": s1_pos2_G}
            
            for bname, btemplate in all_bases.items():
                f.write(f'    "{bname}": {{\n')
                f.write(f'        "atoms": {{\n')
                for aname in BASE_ATOM_NAMES[bname]:
                    if aname in btemplate:
                        c = btemplate[aname]
                        f.write(f'            "{aname}": [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}],\n')
                f.write(f'        }},\n')
                for ref in ["C1'", "O4'", "C2'"]:
                    if ref in avg_dict:
                        c = avg_dict[ref]
                        f.write(f'        "{ref}": [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}],\n')
                f.write(f'    }},\n')
            f.write(f'}}\n')
    
    print(f"\nAll output written to {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
