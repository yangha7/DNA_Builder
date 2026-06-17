#!/usr/bin/env python3
"""
Compare A-DNA builder output against Colin's 3DNA structures.

Since the builder outputs in the helix frame and Colin's structures
are in a different frame, we use Kabsch RMSD (optimal superposition).
"""

import numpy as np
import os
import sys
from dna_builder.builder import build_a_dna

# Atom definitions
A_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("N6","N"),("N1","N"),("C2","C"),("N3","N"),("C4","C")]
T_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("O4","O"),("C5","C"),("C5M","C"),("C6","C")]
G_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("O6","O"),("N1","N"),("C2","C"),("N2","N"),("N3","N"),("C4","C")]
C_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("N4","N"),("C5","C"),("C6","C")]
BASE_ATOMS = {"A": A_ATOMS, "T": T_ATOMS, "G": G_ATOMS, "C": C_ATOMS}
WC = {"A": "T", "T": "A", "G": "C", "C": "G"}


def parse_xyz(fp):
    atoms = []
    with open(fp) as f:
        int(f.readline()); f.readline()
        for l in f:
            p = l.split()
            if len(p) >= 4:
                atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def get_colin_heavy_coords(fp, seq):
    """Get heavy atom coordinates from Colin's XYZ file in strand I + strand II order."""
    heavy = [(e,x,y,z) for e,x,y,z in parse_xyz(fp) if e != "H"]
    n = len(seq)
    comp = "".join(WC[b] for b in seq)
    s2fs = comp[::-1]
    
    coords = []
    idx = 0
    # Strand I
    for b in seq:
        for nm, el in BASE_ATOMS[b]:
            coords.append([heavy[idx][1], heavy[idx][2], heavy[idx][3]])
            idx += 1
    # Strand II (file order = reversed complement)
    for b in s2fs:
        for nm, el in BASE_ATOMS[b]:
            coords.append([heavy[idx][1], heavy[idx][2], heavy[idx][3]])
            idx += 1
    
    return np.array(coords)


def get_builder_heavy_coords(seq):
    """Get heavy atom coordinates from builder output, matching Colin's ordering."""
    atoms = build_a_dna(seq)
    
    # Remove O5T terminal atoms
    atoms = [a for a in atoms if a.name != "O5T"]
    
    n = len(seq)
    comp = "".join(WC[b] for b in seq)
    s2fs = comp[::-1]  # strand II file order
    
    # Strand I atoms are in order (chain A, residues 1..n)
    s1_atoms = [a for a in atoms if a.chain_id == "A"]
    
    # Strand II atoms are in paired order (chain B, residues n..1)
    # But Colin's file has them in reversed order (5'->3' of strand II)
    # So we need to reorder: Colin's file position k = paired position (n-1-k)
    s2_atoms = [a for a in atoms if a.chain_id == "B"]
    
    # Group strand II by residue
    s2_by_res = {}
    for a in s2_atoms:
        if a.residue_seq not in s2_by_res:
            s2_by_res[a.residue_seq] = []
        s2_by_res[a.residue_seq].append(a)
    
    # Reorder strand II to match Colin's file order (5'->3' = reversed paired)
    # In builder: residue n is at paired position 0, residue 1 is at paired position n-1
    # Colin file order: position 0 = paired position n-1 = residue 1
    # So Colin file order is: residue 1, 2, ..., n (ascending)
    
    coords = []
    # Strand I
    for a in s1_atoms:
        coords.append([a.x, a.y, a.z])
    
    # Strand II in Colin's file order
    # Colin outputs strand II in 5'->3' direction
    # In our builder, strand II residue numbering goes n, n-1, ..., 1 (paired order)
    # Colin's file order is the reverse: residues 1, 2, ..., n
    for resseq in range(1, n + 1):
        if resseq in s2_by_res:
            for a in s2_by_res[resseq]:
                coords.append([a.x, a.y, a.z])
    
    return np.array(coords)


def kabsch_rmsd(P, Q):
    """Compute RMSD after optimal superposition using Kabsch algorithm."""
    assert P.shape == Q.shape
    n = P.shape[0]
    
    # Center
    centP = np.mean(P, axis=0)
    centQ = np.mean(Q, axis=0)
    Pc = P - centP
    Qc = Q - centQ
    
    # Kabsch
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    
    # Apply rotation
    Pc_rot = (R @ Pc.T).T
    
    # RMSD
    diff = Pc_rot - Qc
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd


def main():
    d = "Colin_structures/A_form/xyz"
    
    print("=" * 70)
    print("RMSD Comparison: A-DNA Builder vs Colin's 3DNA Structures")
    print("=" * 70)
    print("(Using Kabsch optimal superposition)")
    print()
    
    results = []
    
    for seq in ["ATATAT", "GCGCGC", "AACGTT", "ACATGT", "CGATCG", 
                "ATCGAT", "AAATTT", "GCCGGC", "TATATA", "AGCGCT"]:
        fp = os.path.join(d, f"A_{seq}.xyz")
        if not os.path.exists(fp):
            continue
        
        colin = get_colin_heavy_coords(fp, seq)
        builder = get_builder_heavy_coords(seq)
        
        if colin.shape[0] != builder.shape[0]:
            print(f"  {seq}: Shape mismatch: Colin {colin.shape[0]} vs Builder {builder.shape[0]}")
            continue
        
        rmsd = kabsch_rmsd(builder, colin)
        results.append((seq, rmsd))
        print(f"  {seq}: RMSD = {rmsd:.4f} Å ({colin.shape[0]} atoms)")
    
    if results:
        rmsds = [r for _, r in results]
        print(f"\n  Mean RMSD: {np.mean(rmsds):.4f} Å")
        print(f"  Min RMSD:  {np.min(rmsds):.4f} Å")
        print(f"  Max RMSD:  {np.max(rmsds):.4f} Å")
    
    # Also try comparing against 440D PDB if available
    pdb_440d = "reference_structures/440D.pdb"
    if os.path.exists(pdb_440d):
        print(f"\n\nComparison with 440D crystal structure:")
        # Would need PDB parser - skip for now
        print("  (PDB comparison requires downloading 440D)")
    else:
        print(f"\n  Note: 440D.pdb not found in reference_structures/")
        print(f"  To compare with crystal structure, download from RCSB PDB")


if __name__ == "__main__":
    main()
