#!/usr/bin/env python3
"""
Extract A-DNA nucleotide templates from Colin's 3DNA XYZ structures.

Strategy:
1. Fit helix axis on ATATAT (proven RMSD 0.0009 Å) -> get A, T templates
2. For G/C: superimpose GCGCGC backbone onto ATATAT backbone using Kabsch
3. Extract G/C base atoms in the ATATAT helix frame
4. All templates share the same backbone geometry (3DNA fiber property)
"""

import numpy as np
from scipy.optimize import minimize
import os
from collections import defaultdict

A_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("N6","N"),("N1","N"),("C2","C"),("N3","N"),("C4","C")]
T_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("O4","O"),("C5","C"),("C5M","C"),("C6","C")]
G_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("O6","O"),("N1","N"),("C2","C"),("N2","N"),("N3","N"),("C4","C")]
C_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("N4","N"),("C5","C"),("C6","C")]

BASE_ATOMS = {"A": A_ATOMS, "T": T_ATOMS, "G": G_ATOMS, "C": C_ATOMS}
WC = {"A": "T", "T": "A", "G": "C", "C": "G"}
N_BACKBONE = 11  # First 11 atoms are backbone (same for all bases)

def parse_xyz(fp):
    atoms = []
    with open(fp) as f:
        int(f.readline()); f.readline()
        for l in f:
            p = l.split()
            if len(p) >= 4: atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms

def rot_z(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def parse_nucs(fp):
    seq = os.path.splitext(os.path.basename(fp))[0].split("_")[1]
    heavy = [(e,x,y,z) for e,x,y,z in parse_xyz(fp) if e != "H"]
    n = len(seq); comp = "".join(WC[b] for b in seq); s2fs = comp[::-1]
    s1, s2f, idx = [], [], 0
    for i, b in enumerate(seq):
        nuc = []
        for nm, el in BASE_ATOMS[b]:
            e,x,y,z = heavy[idx]; assert e == el; nuc.append((nm,el,x,y,z)); idx += 1
        s1.append((b, i, nuc))
    for i, b in enumerate(s2fs):
        nuc = []
        for nm, el in BASE_ATOMS[b]:
            e,x,y,z = heavy[idx]; assert e == el; nuc.append((nm,el,x,y,z)); idx += 1
        s2f.append((b, i, nuc))
    s2p = [(s2f[n-1-p][0], p, s2f[n-1-p][2]) for p in range(n)]
    return seq, s1, s2p

def fit_axis_full(s1_nucs):
    """Full 8-param fit. Initial guess: axis along Z."""
    bb = []
    for _, _, nuc in s1_nucs:
        d = {nm: np.array([x,y,z]) for nm,_,x,y,z in nuc}
        bb.append(np.array([d["P"], d["C4'"], d["C1'"]]))
    ns = len(bb)
    def rmsd(params):
        ax,ay,az,nx,ny,nz,theta,d = params
        n = np.array([nx,ny,nz]); nn = np.linalg.norm(n)
        if nn < 1e-10: return 1e10
        n /= nn; p = np.array([ax,ay,az])
        ct,st = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        tot, cnt = 0.0, 0
        for i in range(ns-1):
            sh = bb[i] - p
            rot = np.array([v*ct + np.cross(n,v)*st + n*np.dot(n,v)*(1-ct) for v in sh])
            tot += np.sum((rot + p + d*n - bb[i+1])**2); cnt += len(bb[i])
        return np.sqrt(tot/cnt)
    x0 = [0, 0, 0, 0, 0, 1, 32.727, 2.548]
    res = minimize(rmsd, x0, method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})
    ax,ay,az,nx,ny,nz,theta,d = res.x
    n = np.array([nx,ny,nz]); n /= np.linalg.norm(n)
    if n[2] < 0: n = -n; theta = -theta; d = -d
    return np.array([ax,ay,az]), n, theta, d, res.fun

def to_hf(coords, apt, adir):
    z = np.array([0,0,1.0])
    if np.allclose(np.abs(np.dot(adir,z)), 1.0):
        R = np.eye(3) if np.dot(adir,z) > 0 else np.diag([1,-1,-1])
    else:
        v = np.cross(adir,z); s = np.linalg.norm(v); c = np.dot(adir,z)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx*(1-c)/(s*s)
    return (R @ (coords - apt).T).T, R

def kabsch(P, Q):
    """Find rotation R and translation t that minimizes |R@P + t - Q|.
    Returns R, t such that Q ≈ R @ P + t."""
    centP = np.mean(P, axis=0)
    centQ = np.mean(Q, axis=0)
    Pc = P - centP
    Qc = Q - centQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = centQ - R @ centP
    return R, t

def extract_templates(s1, s2, apt, adir, tw, ri):
    """Transform to helix frame and unwind all nucleotides."""
    c1 = np.array([[x,y,z] for _,_,nuc in s1 for _,_,x,y,z in nuc])
    c2 = np.array([[x,y,z] for _,_,nuc in s2 for _,_,x,y,z in nuc])
    h1, _ = to_hf(c1, apt, adir); h2, _ = to_hf(c2, apt, adir)
    
    s1t, s2t = defaultdict(list), defaultdict(list)
    idx = 0
    for b, step, nuc in s1:
        Ri = rot_z(-tw*step); tv = np.array([0,0,ri*step])
        uw = [(nm,el,*(Ri@(h1[idx+j]-tv))) for j,(nm,el,_,_,_) in enumerate(nuc)]
        s1t[b].append(uw); idx += len(nuc)
    idx = 0
    for b, step, nuc in s2:
        Ri = rot_z(-tw*step); tv = np.array([0,0,ri*step])
        uw = [(nm,el,*(Ri@(h2[idx+j]-tv))) for j,(nm,el,_,_,_) in enumerate(nuc)]
        s2t[b].append(uw); idx += len(nuc)
    return s1t, s2t

def avg(insts):
    return [(insts[0][j][0], insts[0][j][1],
             np.mean([i[j][2] for i in insts]), np.mean([i[j][3] for i in insts]),
             np.mean([i[j][4] for i in insts])) for j in range(len(insts[0]))]

def get_nuc_coords(nuc):
    """Get coordinates array from nucleotide atom list."""
    return np.array([[x,y,z] for _,_,x,y,z in nuc])

def get_backbone_coords(nuc):
    """Get backbone coordinates (first 11 atoms)."""
    return np.array([[x,y,z] for _,_,x,y,z in nuc[:N_BACKBONE]])

def get_base_coords(nuc):
    """Get base coordinates (atoms 11+)."""
    return np.array([[x,y,z] for _,_,x,y,z in nuc[N_BACKBONE:]])

def superimpose_and_extract_base(src_nuc, ref_backbone):
    """
    Superimpose src_nuc's backbone onto ref_backbone using Kabsch,
    then return the transformed base atoms.
    """
    src_bb = get_backbone_coords(src_nuc)
    R, t = kabsch(src_bb, ref_backbone)
    
    # Transform all atoms
    src_all = get_nuc_coords(src_nuc)
    transformed = (R @ src_all.T).T + t
    
    # Return base atoms (11+) with names
    result = []
    for j in range(len(src_nuc)):
        nm, el = src_nuc[j][0], src_nuc[j][1]
        x, y, z = transformed[j]
        result.append((nm, el, x, y, z))
    return result

def fmt(name, t):
    lines = [f"{name} = {{"]
    for b in "ATGC":
        if b not in t: continue
        lines.append(f'    "{b}": [')
        for an,el,x,y,z in t[b]:
            lines.append(f'        ("{an}",{" "*(5-len(an))}"{el}",{" "*(3-len(el))}{x:>10.3f}, {y:>10.3f}, {z:>10.3f}),')
        lines.append("    ],")
    lines.append("}"); return "\n".join(lines)

def main():
    d = "Colin_structures/A_form/xyz"
    
    # Step 1: Get A/T templates from ATATAT
    print("=== Step 1: ATATAT (full fit) ===")
    _, s1_at, s2_at = parse_nucs(os.path.join(d, "A_ATATAT.xyz"))
    apt, adir, tw, ri, frmsd = fit_axis_full(s1_at)
    print(f"  Axis: ({adir[0]:.4f}, {adir[1]:.4f}, {adir[2]:.4f})")
    print(f"  Twist: {tw:.4f}°, Rise: {ri:.4f} Å, Fit: {frmsd:.6f} Å")
    
    s1t_at, s2t_at = extract_templates(s1_at, s2_at, apt, adir, tw, ri)
    
    # Average A and T templates
    s1_A = avg(s1t_at["A"])
    s1_T = avg(s1t_at["T"])
    s2_A = avg(s2t_at["A"])
    s2_T = avg(s2t_at["T"])
    
    print(f"  S1 A backbone: {get_backbone_coords(s1_A).shape}")
    print(f"  S1 T backbone: {get_backbone_coords(s1_T).shape}")
    
    # Step 2: Get G/C base atoms from GCGCGC by superimposing onto A/T backbone
    print("\n=== Step 2: GCGCGC (Kabsch superposition) ===")
    _, s1_gc, s2_gc = parse_nucs(os.path.join(d, "A_GCGCGC.xyz"))
    
    # For strand I: G nucleotides at positions 0,2,4; C at 1,3,5
    # Superimpose each G/C nucleotide's backbone onto the A backbone (position 0)
    # Use the A backbone as reference since purines (A,G) share N9 connectivity
    # and pyrimidines (T,C) share N1 connectivity
    
    ref_purine_bb = get_backbone_coords(s1_A)   # A backbone for G
    ref_pyrimidine_bb = get_backbone_coords(s1_T)  # T backbone for C
    ref_purine_bb_s2 = get_backbone_coords(s2_A)
    ref_pyrimidine_bb_s2 = get_backbone_coords(s2_T)
    
    s1_G_list = []
    s1_C_list = []
    s2_G_list = []
    s2_C_list = []
    
    for base, step, nuc in s1_gc:
        # Get original coords
        orig_coords = get_nuc_coords(nuc)
        orig_bb = orig_coords[:N_BACKBONE]
        
        if base == "G":
            ref_bb = ref_purine_bb
        else:  # C
            ref_bb = ref_pyrimidine_bb
        
        # Kabsch: find R,t such that R@orig_bb + t ≈ ref_bb
        R, t = kabsch(orig_bb, ref_bb)
        transformed = (R @ orig_coords.T).T + t
        
        result = [(nuc[j][0], nuc[j][1], transformed[j][0], transformed[j][1], transformed[j][2])
                  for j in range(len(nuc))]
        
        if base == "G":
            s1_G_list.append(result)
        else:
            s1_C_list.append(result)
    
    for base, step, nuc in s2_gc:
        orig_coords = get_nuc_coords(nuc)
        orig_bb = orig_coords[:N_BACKBONE]
        
        if base == "G":
            ref_bb = ref_purine_bb_s2
        else:
            ref_bb = ref_pyrimidine_bb_s2
        
        R, t = kabsch(orig_bb, ref_bb)
        transformed = (R @ orig_coords.T).T + t
        
        result = [(nuc[j][0], nuc[j][1], transformed[j][0], transformed[j][1], transformed[j][2])
                  for j in range(len(nuc))]
        
        if base == "G":
            s2_G_list.append(result)
        else:
            s2_C_list.append(result)
    
    s1_G = avg(s1_G_list)
    s1_C = avg(s1_C_list)
    s2_G = avg(s2_G_list)
    s2_C = avg(s2_C_list)
    
    # Check backbone RMSD after superposition
    for label, tmpl, ref in [("S1 G vs A bb", s1_G, s1_A), ("S1 C vs T bb", s1_C, s1_T),
                              ("S2 G vs A bb", s2_G, s2_A), ("S2 C vs T bb", s2_C, s2_T)]:
        bb1 = get_backbone_coords(tmpl)
        bb2 = get_backbone_coords(ref)
        rmsd = np.sqrt(np.mean(np.sum((bb1-bb2)**2, axis=1)))
        print(f"  {label}: RMSD = {rmsd:.4f} Å")
    
    # Combine
    fs1 = {"A": s1_A, "T": s1_T, "G": s1_G, "C": s1_C}
    fs2 = {"A": s2_A, "T": s2_T, "G": s2_G, "C": s2_C}
    
    # Bond lengths
    print("\n=== Bond Lengths ===")
    bonds = [("P","O1P",1.48),("P","O2P",1.48),("P","O5'",1.60),("O5'","C5'",1.44),
             ("C5'","C4'",1.52),("C4'","O4'",1.44),("C4'","C3'",1.52),("C3'","O3'",1.44),
             ("C3'","C2'",1.52),("C2'","C1'",1.52),("C1'","O4'",1.44)]
    for sn, t in [("S1",fs1),("S2",fs2)]:
        for b in "ATGC":
            cd = {a[0]: np.array([a[2],a[3],a[4]]) for a in t[b]}
            bl = [f"{a1}-{a2}={np.linalg.norm(cd[a1]-cd[a2]):.3f}" for a1,a2,_ in bonds]
            print(f"  {sn} {b}: {', '.join(bl)}")
    
    # Check glycosidic bond (C1'-N9 for purines, C1'-N1 for pyrimidines)
    print("\n=== Glycosidic Bond ===")
    for sn, t in [("S1",fs1),("S2",fs2)]:
        for b in "ATGC":
            cd = {a[0]: np.array([a[2],a[3],a[4]]) for a in t[b]}
            if b in "AG":
                gly_d = np.linalg.norm(cd["C1'"] - cd["N9"])
                print(f"  {sn} {b}: C1'-N9 = {gly_d:.3f} Å")
            else:
                gly_d = np.linalg.norm(cd["C1'"] - cd["N1"])
                print(f"  {sn} {b}: C1'-N1 = {gly_d:.3f} Å")
    
    # Rebuild verification
    print("\n=== Rebuild ===")
    for test_seq in ["ATATAT","GCGCGC","ACATGT","AACGTT","CGATCG"]:
        fp = os.path.join(d, f"A_{test_seq}.xyz")
        if not os.path.exists(fp): continue
        _, s1n, s2n = parse_nucs(fp)
        
        # Use fixed axis params from ATATAT, optimize only axis point
        bbone = []
        for _, _, nuc in s1n:
            dd = {nm: np.array([x,y,z]) for nm,_,x,y,z in nuc}
            bbone.append(np.array([dd["P"], dd["C4'"], dd["C1'"]]))
        
        def rmsd_pt(params):
            p = np.array(params)
            ct,st = np.cos(np.radians(tw)), np.sin(np.radians(tw))
            tot, cnt = 0.0, 0
            for i in range(len(bbone)-1):
                sh = bbone[i] - p
                rot = np.array([v*ct + np.cross(adir,v)*st + adir*np.dot(adir,v)*(1-ct) for v in sh])
                tot += np.sum((rot + p + ri*adir - bbone[i+1])**2); cnt += len(bbone[i])
            return np.sqrt(tot/cnt)
        
        res = minimize(rmsd_pt, [0,0,0], method='Nelder-Mead',
                      options={'maxiter':10000,'xatol':1e-10,'fatol':1e-12})
        apt_r = np.array(res.x)
        
        orig = np.array([[x,y,z] for _,_,nuc in s1n for _,_,x,y,z in nuc] +
                        [[x,y,z] for _,_,nuc in s2n for _,_,x,y,z in nuc])
        ohf, _ = to_hf(orig, apt_r, adir)
        
        rb = []
        for b,step,nuc in s1n:
            c = np.array([[a[2],a[3],a[4]] for a in fs1[b]])
            rb.extend((rot_z(tw*step)@c.T).T + [0,0,ri*step])
        for b,step,nuc in s2n:
            c = np.array([[a[2],a[3],a[4]] for a in fs2[b]])
            rb.extend((rot_z(tw*step)@c.T).T + [0,0,ri*step])
        rb = np.array(rb)
        rmsd_val = np.sqrt(np.mean(np.sum((rb-ohf)**2,1)))
        print(f"  {test_seq}: RMSD = {rmsd_val:.4f} Å (axis_pt fit: {res.fun:.4f})")
    
    # Output
    print("\n\n" + "="*70)
    print(fmt("A_STRAND1", fs1)); print(); print(fmt("A_STRAND2", fs2))

if __name__ == "__main__":
    main()
