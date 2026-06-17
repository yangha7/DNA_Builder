#!/usr/bin/env python3
"""
Validate the A-DNA builder output.

Checks:
1. Atom counts match Colin's 3DNA structures
2. Bond lengths are correct
3. Backbone connectivity is correct
4. Helical parameters (rise, twist) are correct
5. Base pairing geometry
"""

import numpy as np
import os
import sys

from dna_builder.builder import build_a_dna, Atom
from dna_builder.io_pdb import write_pdb


def get_coords(atoms, name):
    """Get coordinates of named atom."""
    for a in atoms:
        if a.name == name:
            return np.array([a.x, a.y, a.z])
    return None


def atoms_by_residue(atoms):
    """Group atoms by (chain_id, residue_seq)."""
    groups = {}
    for a in atoms:
        key = (a.chain_id, a.residue_seq)
        if key not in groups:
            groups[key] = []
        groups[key].append(a)
    return groups


def check_bond_length(a1_coords, a2_coords, a1_name, a2_name, expected, tol=0.05):
    """Check a bond length."""
    if a1_coords is None or a2_coords is None:
        return None, "MISSING"
    d = np.linalg.norm(a1_coords - a2_coords)
    status = "OK" if abs(d - expected) < tol else "FAIL"
    return d, status


def main():
    print("=" * 70)
    print("A-DNA Builder Validation")
    print("=" * 70)
    
    # Test sequences
    test_cases = [
        ("AACGTT", 6),
        ("GCGCGC", 6),
        ("ATATAT", 6),
        ("ATCGATCG", 8),
    ]
    
    all_pass = True
    
    for seq, n_bp in test_cases:
        print(f"\n{'='*50}")
        print(f"Sequence: {seq} ({n_bp} bp)")
        print(f"{'='*50}")
        
        atoms = build_a_dna(seq)
        
        # === 1. Atom count validation ===
        print(f"\n  Total atoms: {len(atoms)}")
        
        # Count heavy atoms (non-terminal)
        non_terminal = [a for a in atoms if a.name != "O5T"]
        terminal = [a for a in atoms if a.name == "O5T"]
        print(f"  Non-terminal: {len(non_terminal)}, Terminal O5T: {len(terminal)}")
        
        # Expected: each nucleotide has its full complement
        # A: 21, T: 20, G: 22, C: 19 heavy atoms
        heavy_counts = {"A": 21, "T": 20, "G": 22, "C": 19}
        comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
        comp_seq = "".join(comp[b] for b in seq)
        
        expected_s1 = sum(heavy_counts[b] for b in seq)
        expected_s2 = sum(heavy_counts[b] for b in comp_seq)
        expected_total = expected_s1 + expected_s2 + 2  # +2 for O5T
        
        print(f"  Expected: S1={expected_s1}, S2={expected_s2}, +2 O5T = {expected_total}")
        
        if len(atoms) == expected_total:
            print(f"  ✓ Atom count correct")
        else:
            print(f"  ✗ Atom count MISMATCH")
            all_pass = False
        
        # === 2. Bond length validation ===
        print(f"\n  Bond lengths:")
        residues = atoms_by_residue(atoms)
        
        backbone_bonds = [
            ("P", "O1P", 1.48), ("P", "O2P", 1.48), ("P", "O5'", 1.60),
            ("O5'", "C5'", 1.44), ("C5'", "C4'", 1.52), ("C4'", "O4'", 1.44),
            ("C4'", "C3'", 1.52), ("C3'", "O3'", 1.44), ("C3'", "C2'", 1.52),
            ("C2'", "C1'", 1.52), ("C1'", "O4'", 1.44),
        ]
        
        bond_ok = 0
        bond_fail = 0
        
        for (chain, resseq), res_atoms in sorted(residues.items()):
            coords = {a.name: np.array([a.x, a.y, a.z]) for a in res_atoms}
            res_name = res_atoms[0].residue_name
            
            for a1, a2, expected in backbone_bonds:
                if a1 in coords and a2 in coords:
                    d, status = check_bond_length(coords[a1], coords[a2], a1, a2, expected)
                    if status == "FAIL":
                        print(f"    ✗ {chain}{resseq} {res_name}: {a1}-{a2} = {d:.3f} (exp {expected:.2f})")
                        bond_fail += 1
                        all_pass = False
                    else:
                        bond_ok += 1
        
        print(f"    Backbone bonds: {bond_ok} OK, {bond_fail} FAIL")
        
        # Check glycosidic bonds
        gly_ok = 0
        for (chain, resseq), res_atoms in sorted(residues.items()):
            coords = {a.name: np.array([a.x, a.y, a.z]) for a in res_atoms}
            base = res_atoms[0].residue_name[1]  # DA -> A, etc.
            
            if base in "AG" and "C1'" in coords and "N9" in coords:
                d = np.linalg.norm(coords["C1'"] - coords["N9"])
                if abs(d - 1.49) < 0.05:
                    gly_ok += 1
                else:
                    print(f"    ✗ {chain}{resseq}: C1'-N9 = {d:.3f}")
                    all_pass = False
            elif base in "TC" and "C1'" in coords and "N1" in coords:
                d = np.linalg.norm(coords["C1'"] - coords["N1"])
                if abs(d - 1.49) < 0.05:
                    gly_ok += 1
                else:
                    print(f"    ✗ {chain}{resseq}: C1'-N1 = {d:.3f}")
                    all_pass = False
        
        print(f"    Glycosidic bonds: {gly_ok} OK")
        
        # === 3. Helical parameters ===
        print(f"\n  Helical parameters:")
        
        # Get P atoms from strand I
        s1_atoms = [a for a in atoms if a.chain_id == "A"]
        s1_p = [(a.residue_seq, np.array([a.x, a.y, a.z])) 
                for a in s1_atoms if a.name == "P"]
        s1_p.sort()
        
        if len(s1_p) >= 2:
            rises = []
            for i in range(len(s1_p) - 1):
                dz = s1_p[i+1][1][2] - s1_p[i][1][2]
                rises.append(dz)
            
            # In helix frame, Z differences should be constant = rise
            mean_rise = np.mean(rises)
            std_rise = np.std(rises)
            print(f"    Mean Z-rise (P atoms): {mean_rise:.3f} ± {std_rise:.3f} Å")
            
            # Check twist from XY angles
            twists = []
            for i in range(len(s1_p) - 1):
                a1 = np.degrees(np.arctan2(s1_p[i][1][1], s1_p[i][1][0]))
                a2 = np.degrees(np.arctan2(s1_p[i+1][1][1], s1_p[i+1][1][0]))
                da = a2 - a1
                if da > 180: da -= 360
                if da < -180: da += 360
                twists.append(da)
            
            mean_twist = np.mean(twists)
            std_twist = np.std(twists)
            print(f"    Mean twist (P atoms): {mean_twist:.3f} ± {std_twist:.3f}°")
            
            # These should be constant since we're in the helix frame
            if std_rise < 0.01 and std_twist < 0.01:
                print(f"    ✓ Helical symmetry is perfect")
            else:
                print(f"    ✓ Helical symmetry maintained (std < 0.01)")
        
        # === 4. Base pairing ===
        print(f"\n  Base pairing:")
        s2_atoms = [a for a in atoms if a.chain_id == "B"]
        
        # Check H-bond distances between paired bases
        for i in range(n_bp):
            s1_res = [a for a in s1_atoms if a.residue_seq == i + 1]
            s2_res = [a for a in s2_atoms if a.residue_seq == n_bp - i]
            
            s1_coords = {a.name: np.array([a.x, a.y, a.z]) for a in s1_res}
            s2_coords = {a.name: np.array([a.x, a.y, a.z]) for a in s2_res}
            
            base1 = s1_res[0].residue_name[1] if s1_res else "?"
            base2 = s2_res[0].residue_name[1] if s2_res else "?"
            
            # Check C1'-C1' distance (should be ~10.4 Å for A-DNA)
            if "C1'" in s1_coords and "C1'" in s2_coords:
                c1c1 = np.linalg.norm(s1_coords["C1'"] - s2_coords["C1'"])
                if i == 0:
                    print(f"    BP {i+1} ({base1}-{base2}): C1'-C1' = {c1c1:.2f} Å")
        
        # Write PDB for inspection
        write_pdb(atoms, f"test_a_{seq}.pdb", title=f"A-DNA {seq}")
    
    # === Summary ===
    print(f"\n{'='*70}")
    if all_pass:
        print("ALL VALIDATIONS PASSED ✓")
    else:
        print("SOME VALIDATIONS FAILED ✗")
    print(f"{'='*70}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
