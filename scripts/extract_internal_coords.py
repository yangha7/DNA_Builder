#!/usr/bin/env python3
"""
Extract internal coordinates (bond lengths, bond angles, dihedral angles)
from Colin's 3DNA structures for A, B, and Z-form DNA.

The XYZ files from 3DNA have a known atom ordering per nucleotide:
  Backbone: P, O1P, O2P, O5', C5', C4', O4', C3', O3', C2', C1'  (11 atoms)
  Base atoms follow (varies by base type):
    A: N9, C8, N7, C5, C6, N6, N1, C2, N3, C4  (10 atoms, total 21)
    T: N1, C2, O2, N3, C4, O4, C5, C5M, C6      (9 atoms, total 20)
    G: N9, C8, N7, C5, C6, O6, N1, C2, N2, N3, C4  (11 atoms, total 22)
    C: N1, C2, O2, N3, C4, N4, C5, C6            (8 atoms, total 19)

After all heavy atoms for both strands, hydrogen atoms follow, then
terminal O atoms at the very end.

This script extracts:
1. Backbone torsion angles (alpha, beta, gamma, delta, epsilon, zeta)
2. Glycosidic angle chi
3. Bond lengths for all backbone bonds
4. Bond angles for all backbone triples
5. Sugar pucker parameters

Output is written as a Python module (internal_coords.py) that can be
imported by the Z-matrix builder.

Usage:
    python extract_internal_coords.py
"""

import numpy as np
import os
import glob
import json
from collections import defaultdict

# =============================================================================
# Atom ordering and counts
# =============================================================================

BACKBONE_NAMES = ["P", "O1P", "O2P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
BACKBONE_COUNT = 11

BASE_ATOM_NAMES = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "T": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
}

HEAVY_ATOM_COUNTS = {
    "A": 21, "T": 20, "G": 22, "C": 19,
}

# Hydrogen counts per nucleotide (for skipping)
H_COUNTS = {
    "A": 13, "T": 13, "G": 13, "C": 12,
}

WC_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
PURINES = {"A", "G"}
PYRIMIDINES = {"T", "C"}


# =============================================================================
# Geometry functions
# =============================================================================

def distance(a, b):
    """Distance between two 3D points."""
    return np.linalg.norm(a - b)


def angle(a, b, c):
    """Angle at b formed by a-b-c, in degrees."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def dihedral(a, b, c, d):
    """Dihedral angle a-b-c-d, in degrees (-180 to 180)."""
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


def pseudorotation_angle(nu0, nu1, nu2, nu3, nu4):
    """
    Calculate sugar pucker pseudorotation angle P and amplitude tau_m.
    
    nu0 = C4'-O4'-C1'-C2'
    nu1 = O4'-C1'-C2'-C3'
    nu2 = C1'-C2'-C3'-C4'
    nu3 = C2'-C3'-C4'-O4'
    nu4 = C3'-C4'-O4'-C1'
    """
    sin_sum = (nu1 + nu3) * np.sin(np.radians(36)) + (nu2) * np.sin(np.radians(72))
    cos_sum = (nu1 + nu3) * np.cos(np.radians(36)) + (nu2) * np.cos(np.radians(72))
    # More accurate: use all 5 torsions
    # tan(P) = [(nu4 + nu1) - (nu3 + nu0)] / [2 * nu2 * (sin(36) + sin(72))]
    numerator = (nu4 + nu1) - (nu3 + nu0)
    denominator = 2.0 * nu2 * (np.sin(np.radians(36.0)) + np.sin(np.radians(72.0)))
    if abs(denominator) < 1e-10:
        P = 0.0
    else:
        P = np.degrees(np.arctan2(numerator, denominator))
    
    if abs(np.cos(np.radians(P))) > 1e-10:
        tau_m = nu2 / np.cos(np.radians(P))
    else:
        tau_m = nu2  # fallback
    
    return P, abs(tau_m)


# =============================================================================
# Parse XYZ file
# =============================================================================

def parse_xyz(filepath):
    """Parse an XYZ file, return list of (element, x, y, z)."""
    atoms = []
    with open(filepath) as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    # comment line is lines[1]
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((elem, np.array([x, y, z])))
    
    return atoms


def assign_atom_names(atoms, sequence):
    """
    Assign atom names to XYZ atoms based on known 3DNA ordering.
    
    The XYZ file contains:
    1. Strand I heavy atoms (nucleotides in 5'->3' order)
    2. Strand II heavy atoms (nucleotides in 5'->3' order of strand II,
       which is the REVERSE COMPLEMENT of the strand I sequence)
    3. Hydrogen atoms (strand I then strand II)
    4. Terminal O atoms (2 atoms at the very end)
    
    Returns dict with 'strand1' and 'strand2', each a list of dicts
    with atom name -> coordinates.
    """
    # Strand 2 in 3DNA XYZ files is the reverse complement, listed 5'->3'
    comp_sequence = "".join(WC_COMPLEMENT[b] for b in sequence)
    rev_comp_sequence = comp_sequence[::-1]  # reverse complement = strand 2 in 5'->3'
    
    result = {"strand1": [], "strand2": []}
    
    # Parse strand I (5'->3')
    idx = 0
    for i, base in enumerate(sequence):
        nuc = {}
        all_names = BACKBONE_NAMES + BASE_ATOM_NAMES[base]
        for name in all_names:
            if idx < len(atoms):
                nuc[name] = atoms[idx][1]
                idx += 1
        result["strand1"].append(nuc)
    
    # Parse strand II (5'->3' of strand 2 = reverse complement)
    for i, base in enumerate(rev_comp_sequence):
        nuc = {}
        all_names = BACKBONE_NAMES + BASE_ATOM_NAMES[base]
        for name in all_names:
            if idx < len(atoms):
                nuc[name] = atoms[idx][1]
                idx += 1
        result["strand2"].append(nuc)
    
    return result, rev_comp_sequence


# =============================================================================
# Extract internal coordinates from a parsed structure
# =============================================================================

def extract_backbone_torsions(nucleotides):
    """
    Extract backbone torsion angles from a list of nucleotide dicts.
    
    The 6 backbone torsion angles:
      alpha:   O3'(i-1) - P(i)   - O5'(i) - C5'(i)
      beta:    P(i)     - O5'(i) - C5'(i)  - C4'(i)
      gamma:   O5'(i)   - C5'(i) - C4'(i)  - C3'(i)
      delta:   C5'(i)   - C4'(i) - C3'(i)  - O3'(i)
      epsilon: C4'(i)   - C3'(i) - O3'(i)  - P(i+1)
      zeta:    C3'(i)   - O3'(i) - P(i+1)  - O5'(i+1)
    """
    torsions = []
    n = len(nucleotides)
    
    for i in range(n):
        nuc = nucleotides[i]
        t = {}
        
        # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
        if i > 0 and "O3'" in nucleotides[i-1] and all(k in nuc for k in ["P", "O5'", "C5'"]):
            t["alpha"] = dihedral(
                nucleotides[i-1]["O3'"], nuc["P"], nuc["O5'"], nuc["C5'"])
        
        # beta: P(i) - O5'(i) - C5'(i) - C4'(i)
        if all(k in nuc for k in ["P", "O5'", "C5'", "C4'"]):
            t["beta"] = dihedral(nuc["P"], nuc["O5'"], nuc["C5'"], nuc["C4'"])
        
        # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i)
        if all(k in nuc for k in ["O5'", "C5'", "C4'", "C3'"]):
            t["gamma"] = dihedral(nuc["O5'"], nuc["C5'"], nuc["C4'"], nuc["C3'"])
        
        # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i)
        if all(k in nuc for k in ["C5'", "C4'", "C3'", "O3'"]):
            t["delta"] = dihedral(nuc["C5'"], nuc["C4'"], nuc["C3'"], nuc["O3'"])
        
        # epsilon: C4'(i) - C3'(i) - O3'(i) - P(i+1)
        if i < n-1 and all(k in nuc for k in ["C4'", "C3'", "O3'"]) and "P" in nucleotides[i+1]:
            t["epsilon"] = dihedral(
                nuc["C4'"], nuc["C3'"], nuc["O3'"], nucleotides[i+1]["P"])
        
        # zeta: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
        if i < n-1 and all(k in nuc for k in ["C3'", "O3'"]) and \
           all(k in nucleotides[i+1] for k in ["P", "O5'"]):
            t["zeta"] = dihedral(
                nuc["C3'"], nuc["O3'"], nucleotides[i+1]["P"], nucleotides[i+1]["O5'"])
        
        torsions.append(t)
    
    return torsions


def extract_glycosidic_chi(nucleotides, sequence):
    """
    Extract glycosidic torsion angle chi.
    
    For purines (A, G):  O4' - C1' - N9 - C4
    For pyrimidines (T, C): O4' - C1' - N1 - C2
    """
    chis = []
    for i, nuc in enumerate(nucleotides):
        base = sequence[i]
        if base in PURINES:
            if all(k in nuc for k in ["O4'", "C1'", "N9", "C4"]):
                chi = dihedral(nuc["O4'"], nuc["C1'"], nuc["N9"], nuc["C4"])
                chis.append(("chi_pur", chi))
            else:
                chis.append(None)
        else:
            if all(k in nuc for k in ["O4'", "C1'", "N1", "C2"]):
                chi = dihedral(nuc["O4'"], nuc["C1'"], nuc["N1"], nuc["C2"])
                chis.append(("chi_pyr", chi))
            else:
                chis.append(None)
    return chis


def extract_bond_lengths(nucleotides, sequence):
    """Extract backbone bond lengths for each nucleotide."""
    bond_defs = [
        ("P-O5'",   "P",   "O5'"),
        ("O5'-C5'", "O5'", "C5'"),
        ("C5'-C4'", "C5'", "C4'"),
        ("C4'-C3'", "C4'", "C3'"),
        ("C3'-O3'", "C3'", "O3'"),
        ("C4'-O4'", "C4'", "O4'"),
        ("C1'-O4'", "C1'", "O4'"),
        ("C2'-C1'", "C2'", "C1'"),
        ("C3'-C2'", "C3'", "C2'"),
        ("P-O1P",   "P",   "O1P"),
        ("P-O2P",   "P",   "O2P"),
    ]
    
    all_bonds = []
    for i, nuc in enumerate(nucleotides):
        bonds = {}
        for name, a1, a2 in bond_defs:
            if a1 in nuc and a2 in nuc:
                bonds[name] = distance(nuc[a1], nuc[a2])
        
        # O3'(i) - P(i+1) inter-nucleotide bond
        if i < len(nucleotides) - 1 and "O3'" in nuc and "P" in nucleotides[i+1]:
            bonds["O3'-P"] = distance(nuc["O3'"], nucleotides[i+1]["P"])
        
        # Glycosidic bond
        base = sequence[i]
        if base in PURINES and "C1'" in nuc and "N9" in nuc:
            bonds["C1'-N9"] = distance(nuc["C1'"], nuc["N9"])
        elif base in PYRIMIDINES and "C1'" in nuc and "N1" in nuc:
            bonds["C1'-N1"] = distance(nuc["C1'"], nuc["N1"])
        
        all_bonds.append(bonds)
    
    return all_bonds


def extract_bond_angles(nucleotides, sequence):
    """Extract backbone bond angles for each nucleotide."""
    # Angles defined as (name, atom1, atom2_center, atom3)
    angle_defs = [
        ("P-O5'-C5'",     "P",   "O5'", "C5'"),
        ("O5'-C5'-C4'",   "O5'", "C5'", "C4'"),
        ("C5'-C4'-C3'",   "C5'", "C4'", "C3'"),
        ("C4'-C3'-O3'",   "C4'", "C3'", "O3'"),
        ("C5'-C4'-O4'",   "C5'", "C4'", "O4'"),
        ("C3'-C4'-O4'",   "C3'", "C4'", "O4'"),
        ("C4'-O4'-C1'",   "C4'", "O4'", "C1'"),
        ("O4'-C1'-C2'",   "O4'", "C1'", "C2'"),
        ("C1'-C2'-C3'",   "C1'", "C2'", "C3'"),
        ("C2'-C3'-C4'",   "C2'", "C3'", "C4'"),
        ("O1P-P-O2P",     "O1P", "P",   "O2P"),
        ("O1P-P-O5'",     "O1P", "P",   "O5'"),
        ("O2P-P-O5'",     "O2P", "P",   "O5'"),
    ]
    
    all_angles = []
    for i, nuc in enumerate(nucleotides):
        angles = {}
        for name, a1, a2, a3 in angle_defs:
            if a1 in nuc and a2 in nuc and a3 in nuc:
                angles[name] = angle(nuc[a1], nuc[a2], nuc[a3])
        
        # Inter-nucleotide angles
        if i > 0 and "O3'" in nucleotides[i-1] and "P" in nuc and "O5'" in nuc:
            angles["O3'-P-O5'"] = angle(nucleotides[i-1]["O3'"], nuc["P"], nuc["O5'"])
        if i > 0 and "O3'" in nucleotides[i-1] and "P" in nuc and "O1P" in nuc:
            angles["O3'-P-O1P"] = angle(nucleotides[i-1]["O3'"], nuc["P"], nuc["O1P"])
        if i > 0 and "O3'" in nucleotides[i-1] and "P" in nuc and "O2P" in nuc:
            angles["O3'-P-O2P"] = angle(nucleotides[i-1]["O3'"], nuc["P"], nuc["O2P"])
        if "C3'" in nuc and "O3'" in nuc and i < len(nucleotides) - 1 and "P" in nucleotides[i+1]:
            angles["C3'-O3'-P+"] = angle(nuc["C3'"], nuc["O3'"], nucleotides[i+1]["P"])
        
        # Glycosidic angle
        base = sequence[i]
        if base in PURINES:
            if all(k in nuc for k in ["O4'", "C1'", "N9"]):
                angles["O4'-C1'-N9"] = angle(nuc["O4'"], nuc["C1'"], nuc["N9"])
            if all(k in nuc for k in ["C2'", "C1'", "N9"]):
                angles["C2'-C1'-N9"] = angle(nuc["C2'"], nuc["C1'"], nuc["N9"])
        else:
            if all(k in nuc for k in ["O4'", "C1'", "N1"]):
                angles["O4'-C1'-N1"] = angle(nuc["O4'"], nuc["C1'"], nuc["N1"])
            if all(k in nuc for k in ["C2'", "C1'", "N1"]):
                angles["C2'-C1'-N1"] = angle(nuc["C2'"], nuc["C1'"], nuc["N1"])
        
        all_angles.append(angles)
    
    return all_angles


def extract_sugar_pucker(nucleotides):
    """Extract sugar pucker parameters (pseudorotation angle P and amplitude)."""
    puckers = []
    for nuc in nucleotides:
        required = ["C4'", "O4'", "C1'", "C2'", "C3'"]
        if all(k in nuc for k in required):
            nu0 = dihedral(nuc["C4'"], nuc["O4'"], nuc["C1'"], nuc["C2'"])
            nu1 = dihedral(nuc["O4'"], nuc["C1'"], nuc["C2'"], nuc["C3'"])
            nu2 = dihedral(nuc["C1'"], nuc["C2'"], nuc["C3'"], nuc["C4'"])
            nu3 = dihedral(nuc["C2'"], nuc["C3'"], nuc["C4'"], nuc["O4'"])
            nu4 = dihedral(nuc["C3'"], nuc["C4'"], nuc["O4'"], nuc["C1'"])
            P_angle, tau_m = pseudorotation_angle(nu0, nu1, nu2, nu3, nu4)
            puckers.append({"P": P_angle, "tau_m": tau_m,
                           "nu0": nu0, "nu1": nu1, "nu2": nu2, "nu3": nu3, "nu4": nu4})
        else:
            puckers.append(None)
    return puckers


def extract_base_geometry(nucleotides, sequence):
    """
    Extract base atom positions relative to C1', for inserting bases.
    Returns the base atom coordinates in a local frame defined by C1', N(glycosidic), O4'.
    """
    base_geoms = []
    for i, nuc in enumerate(nucleotides):
        base = sequence[i]
        base_names = BASE_ATOM_NAMES[base]
        
        # Get base atom positions
        base_atoms = {}
        for name in base_names:
            if name in nuc:
                base_atoms[name] = nuc[name].copy()
        
        # Store C1' and reference atoms
        geom = {
            "base": base,
            "atoms": base_atoms,
        }
        if "C1'" in nuc:
            geom["C1'"] = nuc["C1'"].copy()
        if "O4'" in nuc:
            geom["O4'"] = nuc["O4'"].copy()
        if "C2'" in nuc:
            geom["C2'"] = nuc["C2'"].copy()
        
        base_geoms.append(geom)
    
    return base_geoms


# =============================================================================
# Process all structures for a given form
# =============================================================================

def process_form(form, directory):
    """Process all XYZ files for a given DNA form."""
    pattern = os.path.join(directory, f"{form}_*.xyz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"  No files found matching {pattern}")
        return None
    
    print(f"  Found {len(files)} structures")
    
    all_torsions = defaultdict(list)
    all_bond_lengths = defaultdict(list)
    all_bond_angles = defaultdict(list)
    all_chis = {"chi_pur": [], "chi_pyr": []}
    all_puckers = []
    all_base_geoms = defaultdict(list)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract sequence from filename: B_ATATAT.xyz -> ATATAT
        seq = filename.split("_", 1)[1].replace(".xyz", "")
        
        atoms = parse_xyz(filepath)
        parsed, rev_comp_seq = assign_atom_names(atoms, seq)
        
        # Process strand 1 and strand 2
        s1_nucs = parsed["strand1"]
        s2_nucs = parsed["strand2"]
        
        for strand_nucs, strand_seq, strand_label in [
            (s1_nucs, seq, "s1"),
            (s2_nucs, rev_comp_seq, "s2"),
        ]:
            # Torsion angles
            torsions = extract_backbone_torsions(strand_nucs)
            for t in torsions:
                for name, val in t.items():
                    all_torsions[name].append(val)
            
            # Chi angles
            chis = extract_glycosidic_chi(strand_nucs, strand_seq)
            for chi in chis:
                if chi is not None:
                    all_chis[chi[0]].append(chi[1])
            
            # Bond lengths
            bonds = extract_bond_lengths(strand_nucs, strand_seq)
            for b in bonds:
                for name, val in b.items():
                    all_bond_lengths[name].append(val)
            
            # Bond angles
            angles = extract_bond_angles(strand_nucs, strand_seq)
            for a in angles:
                for name, val in a.items():
                    all_bond_angles[name].append(val)
            
            # Sugar pucker
            puckers = extract_sugar_pucker(strand_nucs)
            for p in puckers:
                if p is not None:
                    all_puckers.append(p)
            
            # Base geometry
            base_geoms = extract_base_geometry(strand_nucs, strand_seq)
            for bg in base_geoms:
                all_base_geoms[bg["base"]].append(bg)
    
    # Compute averages
    result = {
        "torsion_angles": {},
        "bond_lengths": {},
        "bond_angles": {},
        "sugar_pucker": {},
    }
    
    print(f"\n  Backbone torsion angles:")
    for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
        if name in all_torsions and len(all_torsions[name]) > 0:
            vals = np.array(all_torsions[name])
            # Use circular mean for angles
            mean_val = circular_mean(vals)
            std_val = circular_std(vals)
            result["torsion_angles"][name] = round(mean_val, 1)
            print(f"    {name:10s}: {mean_val:8.1f}° ± {std_val:5.1f}° (n={len(vals)})")
    
    print(f"\n  Glycosidic angles:")
    for name in ["chi_pur", "chi_pyr"]:
        if len(all_chis[name]) > 0:
            vals = np.array(all_chis[name])
            mean_val = circular_mean(vals)
            std_val = circular_std(vals)
            result["torsion_angles"][name] = round(mean_val, 1)
            print(f"    {name:10s}: {mean_val:8.1f}° ± {std_val:5.1f}° (n={len(vals)})")
    
    print(f"\n  Bond lengths:")
    for name in sorted(all_bond_lengths.keys()):
        vals = np.array(all_bond_lengths[name])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        result["bond_lengths"][name] = round(mean_val, 3)
        print(f"    {name:12s}: {mean_val:6.3f} Å ± {std_val:5.3f} (n={len(vals)})")
    
    print(f"\n  Bond angles:")
    for name in sorted(all_bond_angles.keys()):
        vals = np.array(all_bond_angles[name])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        result["bond_angles"][name] = round(mean_val, 1)
        print(f"    {name:20s}: {mean_val:7.1f}° ± {std_val:5.1f}° (n={len(vals)})")
    
    print(f"\n  Sugar pucker:")
    if all_puckers:
        P_vals = np.array([p["P"] for p in all_puckers])
        tau_vals = np.array([p["tau_m"] for p in all_puckers])
        mean_P = circular_mean(P_vals)
        mean_tau = np.mean(tau_vals)
        result["sugar_pucker"]["P"] = round(mean_P, 1)
        result["sugar_pucker"]["tau_m"] = round(mean_tau, 1)
        print(f"    P (pseudorotation): {mean_P:7.1f}°")
        print(f"    tau_m (amplitude):  {mean_tau:7.1f}°")
    
    # Extract representative base geometries (from first structure)
    result["base_templates"] = {}
    for base in ["A", "T", "G", "C"]:
        if base in all_base_geoms and len(all_base_geoms[base]) > 0:
            bg = all_base_geoms[base][0]
            result["base_templates"][base] = {
                "atoms": {k: v.tolist() for k, v in bg["atoms"].items()},
                "C1'": bg["C1'"].tolist() if "C1'" in bg else None,
                "O4'": bg["O4'"].tolist() if "O4'" in bg else None,
                "C2'": bg["C2'"].tolist() if "C2'" in bg else None,
            }
    
    return result


def process_z_form(directory):
    """
    Process Z-DNA structures specially.
    
    Z-DNA has a dinucleotide repeat with alternating conformations:
    - Position 1 (even index, 0-based): syn purine
    - Position 2 (odd index, 0-based): anti pyrimidine
    
    We extract separate internal coordinates for each position.
    """
    pattern = os.path.join(directory, "Z_*.xyz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"  No files found matching {pattern}")
        return None
    
    print(f"  Found {len(files)} structures")
    
    # Separate collections for pos1 (syn) and pos2 (anti)
    pos1_torsions = defaultdict(list)
    pos2_torsions = defaultdict(list)
    pos1_bond_lengths = defaultdict(list)
    pos2_bond_lengths = defaultdict(list)
    pos1_bond_angles = defaultdict(list)
    pos2_bond_angles = defaultdict(list)
    pos1_chis = {"chi_pur": [], "chi_pyr": []}
    pos2_chis = {"chi_pur": [], "chi_pyr": []}
    pos1_puckers = []
    pos2_puckers = []
    all_base_geoms = defaultdict(list)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        seq = filename.split("_", 1)[1].replace(".xyz", "")
        
        atoms = parse_xyz(filepath)
        parsed, rev_comp_seq = assign_atom_names(atoms, seq)
        
        for strand_nucs, strand_seq, strand_label in [
            (parsed["strand1"], seq, "s1"),
            (parsed["strand2"], rev_comp_seq, "s2"),
        ]:
            n = len(strand_nucs)
            
            # Extract all torsions
            torsions = extract_backbone_torsions(strand_nucs)
            chis = extract_glycosidic_chi(strand_nucs, strand_seq)
            bonds = extract_bond_lengths(strand_nucs, strand_seq)
            angles = extract_bond_angles(strand_nucs, strand_seq)
            puckers = extract_sugar_pucker(strand_nucs)
            base_geoms = extract_base_geometry(strand_nucs, strand_seq)
            
            for i in range(n):
                is_pos2 = (i % 2 == 1)
                
                if is_pos2:
                    for name, val in torsions[i].items():
                        pos2_torsions[name].append(val)
                    for name, val in bonds[i].items():
                        pos2_bond_lengths[name].append(val)
                    for name, val in angles[i].items():
                        pos2_bond_angles[name].append(val)
                    if chis[i] is not None:
                        pos2_chis[chis[i][0]].append(chis[i][1])
                    if puckers[i] is not None:
                        pos2_puckers.append(puckers[i])
                else:
                    for name, val in torsions[i].items():
                        pos1_torsions[name].append(val)
                    for name, val in bonds[i].items():
                        pos1_bond_lengths[name].append(val)
                    for name, val in angles[i].items():
                        pos1_bond_angles[name].append(val)
                    if chis[i] is not None:
                        pos1_chis[chis[i][0]].append(chis[i][1])
                    if puckers[i] is not None:
                        pos1_puckers.append(puckers[i])
                
                all_base_geoms[base_geoms[i]["base"]].append(base_geoms[i])
    
    result = {"pos1": {}, "pos2": {}}
    
    for pos_label, torsions_dict, bonds_dict, angles_dict, chis_dict, puckers_list in [
        ("pos1", pos1_torsions, pos1_bond_lengths, pos1_bond_angles, pos1_chis, pos1_puckers),
        ("pos2", pos2_torsions, pos2_bond_lengths, pos2_bond_angles, pos2_chis, pos2_puckers),
    ]:
        pos_result = {
            "torsion_angles": {},
            "bond_lengths": {},
            "bond_angles": {},
            "sugar_pucker": {},
        }
        
        print(f"\n  === {pos_label.upper()} ({'syn' if pos_label == 'pos1' else 'anti'}) ===")
        
        print(f"  Backbone torsion angles:")
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
            if name in torsions_dict and len(torsions_dict[name]) > 0:
                vals = np.array(torsions_dict[name])
                mean_val = circular_mean(vals)
                std_val = circular_std(vals)
                pos_result["torsion_angles"][name] = round(mean_val, 1)
                print(f"    {name:10s}: {mean_val:8.1f}° ± {std_val:5.1f}° (n={len(vals)})")
        
        print(f"  Glycosidic angles:")
        for name in ["chi_pur", "chi_pyr"]:
            if len(chis_dict[name]) > 0:
                vals = np.array(chis_dict[name])
                mean_val = circular_mean(vals)
                std_val = circular_std(vals)
                pos_result["torsion_angles"][name] = round(mean_val, 1)
                print(f"    {name:10s}: {mean_val:8.1f}° ± {std_val:5.1f}° (n={len(vals)})")
        
        print(f"  Bond lengths:")
        for name in sorted(bonds_dict.keys()):
            vals = np.array(bonds_dict[name])
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            pos_result["bond_lengths"][name] = round(mean_val, 3)
            print(f"    {name:12s}: {mean_val:6.3f} Å ± {std_val:5.3f} (n={len(vals)})")
        
        print(f"  Bond angles:")
        for name in sorted(angles_dict.keys()):
            vals = np.array(angles_dict[name])
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            pos_result["bond_angles"][name] = round(mean_val, 1)
            print(f"    {name:20s}: {mean_val:7.1f}° ± {std_val:5.1f}° (n={len(vals)})")
        
        print(f"  Sugar pucker:")
        if puckers_list:
            P_vals = np.array([p["P"] for p in puckers_list])
            tau_vals = np.array([p["tau_m"] for p in puckers_list])
            mean_P = circular_mean(P_vals)
            mean_tau = np.mean(tau_vals)
            pos_result["sugar_pucker"]["P"] = round(mean_P, 1)
            pos_result["sugar_pucker"]["tau_m"] = round(mean_tau, 1)
            print(f"    P (pseudorotation): {mean_P:7.1f}°")
            print(f"    tau_m (amplitude):  {mean_tau:7.1f}°")
        
        result[pos_label] = pos_result
    
    # Base templates (shared)
    result["base_templates"] = {}
    for base in ["A", "T", "G", "C"]:
        if base in all_base_geoms and len(all_base_geoms[base]) > 0:
            bg = all_base_geoms[base][0]
            result["base_templates"][base] = {
                "atoms": {k: v.tolist() for k, v in bg["atoms"].items()},
                "C1'": bg["C1'"].tolist() if "C1'" in bg else None,
                "O4'": bg["O4'"].tolist() if "O4'" in bg else None,
                "C2'": bg["C2'"].tolist() if "C2'" in bg else None,
            }
    
    return result


# =============================================================================
# Circular statistics
# =============================================================================

def circular_mean(angles_deg):
    """Compute circular mean of angles in degrees."""
    angles_rad = np.radians(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    return np.degrees(np.arctan2(mean_sin, mean_cos))


def circular_std(angles_deg):
    """Compute circular standard deviation of angles in degrees."""
    angles_rad = np.radians(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    if R > 1.0:
        R = 1.0
    if R < 1e-10:
        return 180.0
    return np.degrees(np.sqrt(-2.0 * np.log(R)))


# =============================================================================
# Generate internal_coords.py module
# =============================================================================

def generate_module(b_result, a_result, z_result):
    """Generate the internal_coords.py module with extracted parameters."""
    
    lines = [
        '"""',
        'Internal coordinate parameters for DNA forms A, B, and Z.',
        '',
        'Extracted from Colin\'s 3DNA fiber diffraction structures.',
        'These parameters define the backbone geometry for building DNA',
        'using the Z-matrix (internal coordinate) method.',
        '',
        'Backbone torsion angles:',
        '  alpha:   O3\'(i-1) - P(i)   - O5\'(i) - C5\'(i)',
        '  beta:    P(i)     - O5\'(i) - C5\'(i)  - C4\'(i)',
        '  gamma:   O5\'(i)   - C5\'(i) - C4\'(i)  - C3\'(i)',
        '  delta:   C5\'(i)   - C4\'(i) - C3\'(i)  - O3\'(i)',
        '  epsilon: C4\'(i)   - C3\'(i) - O3\'(i)  - P(i+1)',
        '  zeta:    C3\'(i)   - O3\'(i) - P(i+1)  - O5\'(i+1)',
        '  chi_pur: O4\'     - C1\'    - N9      - C4  (purines)',
        '  chi_pyr: O4\'     - C1\'    - N1      - C2  (pyrimidines)',
        '',
        'Auto-generated by extract_internal_coords.py',
        '"""',
        '',
    ]
    
    def format_dict(d, indent=8):
        """Format a dict as Python code."""
        prefix = " " * indent
        items = []
        for k, v in sorted(d.items()):
            if isinstance(v, float):
                items.append(f'{prefix}"{k}": {v},')
            else:
                items.append(f'{prefix}"{k}": {v},')
        return "\n".join(items)
    
    # B-DNA
    if b_result:
        lines.append("B_DNA_PARAMS = {")
        lines.append('    "torsion_angles": {')
        lines.append(format_dict(b_result["torsion_angles"]))
        lines.append("    },")
        lines.append('    "bond_lengths": {')
        lines.append(format_dict(b_result["bond_lengths"]))
        lines.append("    },")
        lines.append('    "bond_angles": {')
        lines.append(format_dict(b_result["bond_angles"]))
        lines.append("    },")
        lines.append('    "sugar_pucker": {')
        lines.append(format_dict(b_result["sugar_pucker"]))
        lines.append("    },")
        lines.append("}")
        lines.append("")
    
    # A-DNA
    if a_result:
        lines.append("A_DNA_PARAMS = {")
        lines.append('    "torsion_angles": {')
        lines.append(format_dict(a_result["torsion_angles"]))
        lines.append("    },")
        lines.append('    "bond_lengths": {')
        lines.append(format_dict(a_result["bond_lengths"]))
        lines.append("    },")
        lines.append('    "bond_angles": {')
        lines.append(format_dict(a_result["bond_angles"]))
        lines.append("    },")
        lines.append('    "sugar_pucker": {')
        lines.append(format_dict(a_result["sugar_pucker"]))
        lines.append("    },")
        lines.append("}")
        lines.append("")
    
    # Z-DNA (two positions)
    if z_result:
        lines.append("# Z-DNA has a dinucleotide repeat with alternating conformations")
        lines.append("# pos1 = syn (even index), pos2 = anti (odd index)")
        lines.append("")
        for pos in ["pos1", "pos2"]:
            var_name = f"Z_DNA_{pos.upper()}_PARAMS"
            lines.append(f"{var_name} = {{")
            lines.append('    "torsion_angles": {')
            lines.append(format_dict(z_result[pos]["torsion_angles"]))
            lines.append("    },")
            lines.append('    "bond_lengths": {')
            lines.append(format_dict(z_result[pos]["bond_lengths"]))
            lines.append("    },")
            lines.append('    "bond_angles": {')
            lines.append(format_dict(z_result[pos]["bond_angles"]))
            lines.append("    },")
            lines.append('    "sugar_pucker": {')
            lines.append(format_dict(z_result[pos]["sugar_pucker"]))
            lines.append("    },")
            lines.append("}")
            lines.append("")
    
    # Convenience lookup
    lines.append("INTERNAL_COORDS = {")
    if b_result:
        lines.append('    "B": B_DNA_PARAMS,')
    if a_result:
        lines.append('    "A": A_DNA_PARAMS,')
    if z_result:
        lines.append('    "Z": {"pos1": Z_DNA_POS1_PARAMS, "pos2": Z_DNA_POS2_PARAMS},')
    lines.append("}")
    lines.append("")
    
    # Base template geometries - store for each form
    lines.append("# Base atom templates (relative coordinates from first structure)")
    lines.append("# Used to place base atoms relative to the sugar")
    
    def _format_ref_atom(bt, key):
        """Format a reference atom line for the base template."""
        c = bt[key]
        return '        "{}": [{:.3f}, {:.3f}, {:.3f}],'.format(key, c[0], c[1], c[2])

    for form_name, form_result in [("B", b_result), ("A", a_result)]:
        if form_result and "base_templates" in form_result:
            lines.append("")
            lines.append("{}_BASE_TEMPLATES = {{".format(form_name))
            for base in ["A", "T", "G", "C"]:
                if base in form_result["base_templates"]:
                    bt = form_result["base_templates"][base]
                    lines.append('    "{}": {{'.format(base))
                    lines.append('        "atoms": {')
                    for atom_name, coords in sorted(bt["atoms"].items()):
                        lines.append('            "{}": [{:.3f}, {:.3f}, {:.3f}],'.format(
                            atom_name, coords[0], coords[1], coords[2]))
                    lines.append('        },')
                    if bt.get("C1'"):
                        lines.append(_format_ref_atom(bt, "C1'"))
                    if bt.get("O4'"):
                        lines.append(_format_ref_atom(bt, "O4'"))
                    if bt.get("C2'"):
                        lines.append(_format_ref_atom(bt, "C2'"))
                    lines.append('    },')
            lines.append("}")
    
    if z_result and "base_templates" in z_result:
        lines.append("")
        lines.append("Z_BASE_TEMPLATES = {")
        for base in ["A", "T", "G", "C"]:
            if base in z_result["base_templates"]:
                bt = z_result["base_templates"][base]
                lines.append('    "{}": {{'.format(base))
                lines.append('        "atoms": {')
                for atom_name, coords in sorted(bt["atoms"].items()):
                    lines.append('            "{}": [{:.3f}, {:.3f}, {:.3f}],'.format(
                        atom_name, coords[0], coords[1], coords[2]))
                lines.append('        },')
                if bt.get("C1'"):
                    lines.append(_format_ref_atom(bt, "C1'"))
                if bt.get("O4'"):
                    lines.append(_format_ref_atom(bt, "O4'"))
                if bt.get("C2'"):
                    lines.append(_format_ref_atom(bt, "C2'"))
                lines.append('    },')
        lines.append("}")
    
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    base_dir = "Colin_structures"
    
    print("=" * 60)
    print("Extracting internal coordinates from Colin's 3DNA structures")
    print("=" * 60)
    
    # B-DNA
    print(f"\n{'='*60}")
    print("B-DNA")
    print(f"{'='*60}")
    b_result = process_form("B", os.path.join(base_dir, "B_form", "xyz"))
    
    # A-DNA
    print(f"\n{'='*60}")
    print("A-DNA")
    print(f"{'='*60}")
    a_result = process_form("A", os.path.join(base_dir, "A_form", "xyz"))
    
    # Z-DNA (special handling for dinucleotide repeat)
    print(f"\n{'='*60}")
    print("Z-DNA")
    print(f"{'='*60}")
    z_result = process_z_form(os.path.join(base_dir, "Z_form", "xyz"))
    
    # Generate module
    print(f"\n{'='*60}")
    print("Generating dna_builder/internal_coords.py")
    print(f"{'='*60}")
    
    module_code = generate_module(b_result, a_result, z_result)
    
    output_path = os.path.join("dna_builder", "internal_coords.py")
    with open(output_path, "w") as f:
        f.write(module_code)
    
    print(f"Written to {output_path}")
    
    # Also save raw data as JSON for reference
    raw_data = {
        "B": b_result,
        "A": a_result,
        "Z": z_result,
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    json_path = "internal_coords_raw.json"
    with open(json_path, "w") as f:
        json.dump(convert_for_json(raw_data), f, indent=2)
    print(f"Raw data written to {json_path}")


if __name__ == "__main__":
    main()
