#!/usr/bin/env python3
"""
Find the helix axis in Colin's 3DNA A-form structures and transform
coordinates to the helix frame (axis along Z).

The 3DNA fiber structures have helical screw symmetry, but the helix
axis is not necessarily along Z. We need to:
1. Find the helix axis direction and position
2. Transform all coordinates to the helix frame
3. Extract templates at position 0
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import os
from collections import defaultdict

# Atom definitions (same as before)
A_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("N6","N"),("N1","N"),("C2","C"),("N3","N"),("C4","C")]
T_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("O4","O"),("C5","C"),("C5M","C"),("C6","C")]
G_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("O6","O"),("N1","N"),("C2","C"),("N2","N"),("N3","N"),("C4","C")]
C_ATOMS = [("P","P"),("O1P","O"),("O2P","O"),("O5'","O"),("C5'","C"),("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),("C1'","C"),("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("N4","N"),("C5","C"),("C6","C")]

BASE_ATOMS = {"A": A_ATOMS, "T": T_ATOMS, "G": G_ATOMS, "C": C_ATOMS}
HEAVY_COUNTS = {b: len(a) for b, a in BASE_ATOMS.items()}
WC_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

A_RISE = 2.548
A_TWIST = 32.727


def parse_xyz(filepath):
    atoms = []
    with open(filepath) as f:
        n = int(f.readline().strip())
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def rot_z(angle_deg):
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def get_sequence(filename):
    return os.path.splitext(os.path.basename(filename))[0].split("_")[1]


def parse_nucleotides(filepath):
    """Parse XYZ file into nucleotides with atom names."""
    seq = get_sequence(filepath)
    atoms = parse_xyz(filepath)
    heavy = [(e, x, y, z) for e, x, y, z in atoms if e != "H"]
    
    n_bp = len(seq)
    comp = "".join(WC_COMPLEMENT[b] for b in seq)
    s2_file_seq = comp[::-1]
    
    s1_nucs = []
    idx = 0
    for i, base in enumerate(seq):
        template = BASE_ATOMS[base]
        nuc = []
        for name, elem in template:
            e, x, y, z = heavy[idx]
            if e != elem:
                return None, None, None
            nuc.append((name, elem, x, y, z))
            idx += 1
        s1_nucs.append((base, i, nuc))
    
    s2_nucs_file = []
    for i, base in enumerate(s2_file_seq):
        template = BASE_ATOMS[base]
        nuc = []
        for name, elem in template:
            e, x, y, z = heavy[idx]
            if e != elem:
                return None, None, None
            nuc.append((name, elem, x, y, z))
            idx += 1
        s2_nucs_file.append((base, i, nuc))
    
    # Remap to paired order
    s2_nucs_paired = []
    for p in range(n_bp):
        file_k = n_bp - 1 - p
        base, _, nuc = s2_nucs_file[file_k]
        s2_nucs_paired.append((base, p, nuc))
    
    return seq, s1_nucs, s2_nucs_paired


def get_backbone_coords(nuc_atoms):
    """Get backbone atom coordinates (P, C4', C1') for helix fitting."""
    coords = {}
    for name, elem, x, y, z in nuc_atoms:
        coords[name] = np.array([x, y, z])
    # Use P, C4', C1' as reference points
    result = []
    for name in ["P", "C4'", "C1'"]:
        if name in coords:
            result.append(coords[name])
    return np.array(result)


def fit_helical_screw(coords_list, n_ref_atoms=3):
    """
    Find the helical screw axis and parameters that best map
    consecutive nucleotides onto each other.
    
    coords_list: list of (n_atoms, 3) arrays for consecutive nucleotides
                 (must have same number of atoms - use backbone only)
    
    Returns: axis_point, axis_direction, twist, rise, rmsd
    """
    n_steps = len(coords_list)
    
    # Use pairs of consecutive nucleotides
    # For each pair, find the rigid body transformation
    # Then find the common screw axis
    
    # Method: Use the midpoint approach
    # For a helical screw, all backbone atoms trace helices
    # The helix axis can be found from 3+ points on the same helix
    
    # Use P atoms from all steps
    p_coords = np.array([cl[0] for cl in coords_list])  # P atom from each step
    
    # Fit a helix to the P atom trajectory
    # A helix: r(t) = center + R*cos(ωt+φ)*e1 + R*sin(ωt+φ)*e2 + h*t*axis
    # where axis, e1, e2 form an orthonormal basis
    
    # Step 1: Find the helix axis direction
    # The axis direction is the eigenvector of the covariance matrix
    # corresponding to the direction of maximum extent
    
    # Actually, let's use a simpler approach:
    # Find the best-fit screw transformation between step 0 and step 1
    
    def screw_transform(params, coords):
        """Apply screw transformation: rotate around axis + translate along axis."""
        # params: ax, ay, az (axis point), dx, dy, dz (axis direction), theta (angle), d (displacement)
        ax, ay, az, nx, ny, nz, theta, d = params
        
        # Normalize axis direction
        n = np.array([nx, ny, nz])
        n = n / np.linalg.norm(n)
        
        # Point on axis
        p = np.array([ax, ay, az])
        
        # Rotation around axis through point p
        # Translate to origin, rotate, translate back
        shifted = coords - p
        
        # Rodrigues' rotation formula
        ct = np.cos(np.radians(theta))
        st = np.sin(np.radians(theta))
        
        rotated = np.zeros_like(shifted)
        for i in range(len(shifted)):
            v = shifted[i]
            rotated[i] = v * ct + np.cross(n, v) * st + n * np.dot(n, v) * (1 - ct)
        
        # Translate back and add displacement along axis
        result = rotated + p + d * n
        return result
    
    def total_rmsd(params):
        """Total RMSD across all consecutive pairs."""
        total = 0.0
        count = 0
        for i in range(n_steps - 1):
            pred = screw_transform(params, coords_list[i])
            diff = pred - coords_list[i + 1]
            total += np.sum(diff ** 2)
            count += len(coords_list[i])
        return np.sqrt(total / count)
    
    # Initial guess: axis along Z through origin
    x0 = [0, 0, 0, 0, 0, 1, A_TWIST, A_RISE]
    
    result = minimize(total_rmsd, x0, method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    ax, ay, az, nx, ny, nz, theta, d = result.x
    n = np.array([nx, ny, nz])
    n = n / np.linalg.norm(n)
    
    return np.array([ax, ay, az]), n, theta, d, result.fun


def transform_to_helix_frame(coords, axis_point, axis_dir):
    """
    Transform coordinates so the helix axis is along Z through the origin.
    
    axis_point: a point on the helix axis
    axis_dir: unit vector along the helix axis
    """
    # Build rotation matrix to align axis_dir with Z
    z = np.array([0, 0, 1.0])
    
    if np.allclose(axis_dir, z) or np.allclose(axis_dir, -z):
        R = np.eye(3) if np.dot(axis_dir, z) > 0 else np.diag([1, -1, -1])
    else:
        v = np.cross(axis_dir, z)
        s = np.linalg.norm(v)
        c = np.dot(axis_dir, z)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    
    # Translate so axis_point is at origin, then rotate
    shifted = coords - axis_point
    rotated = (R @ shifted.T).T
    
    return rotated, R


def main():
    xyz_dir = "Colin_structures/A_form/xyz"
    
    # Use a structure with repeated bases for easier analysis
    # ATATAT has A at positions 0,2,4 and T at positions 1,3,5
    filepath = os.path.join(xyz_dir, "A_ATATAT.xyz")
    seq, s1_nucs, s2_nucs = parse_nucleotides(filepath)
    
    print(f"Analyzing {seq}")
    print(f"Strand I: {len(s1_nucs)} nucleotides")
    
    # Get backbone coordinates for each nucleotide
    bb_coords = []
    for base, step, nuc in s1_nucs:
        bb = get_backbone_coords(nuc)
        bb_coords.append(bb)
    
    # Fit helical screw
    print("\nFitting helical screw to strand I backbone...")
    axis_pt, axis_dir, twist, rise, rmsd = fit_helical_screw(bb_coords)
    
    print(f"  Axis point: ({axis_pt[0]:.4f}, {axis_pt[1]:.4f}, {axis_pt[2]:.4f})")
    print(f"  Axis direction: ({axis_dir[0]:.4f}, {axis_dir[1]:.4f}, {axis_dir[2]:.4f})")
    print(f"  Twist: {twist:.4f}°")
    print(f"  Rise: {rise:.4f} Å")
    print(f"  RMSD: {rmsd:.6f} Å")
    
    # Transform all coordinates to helix frame
    print("\nTransforming to helix frame...")
    
    # Get all heavy atom coordinates
    all_s1_coords = []
    all_s1_names = []
    for base, step, nuc in s1_nucs:
        for name, elem, x, y, z in nuc:
            all_s1_coords.append([x, y, z])
            all_s1_names.append((name, elem, base, step))
    all_s1_coords = np.array(all_s1_coords)
    
    transformed, R = transform_to_helix_frame(all_s1_coords, axis_pt, axis_dir)
    
    # Check: P atoms should now have constant radius and linear Z
    print("\nP atoms in helix frame:")
    p_indices = [i for i, (n, e, b, s) in enumerate(all_s1_names) if n == "P"]
    for pi in p_indices:
        x, y, z = transformed[pi]
        r = np.sqrt(x**2 + y**2)
        angle = np.degrees(np.arctan2(y, x))
        name, elem, base, step = all_s1_names[pi]
        print(f"  Step {step} ({base}): ({x:8.3f}, {y:8.3f}, {z:8.3f})  r={r:.3f}  angle={angle:.3f}°")
    
    # Check Z differences
    print("\nZ differences between consecutive P atoms:")
    for i in range(len(p_indices) - 1):
        dz = transformed[p_indices[i+1]][2] - transformed[p_indices[i]][2]
        print(f"  Step {i} -> {i+1}: dz = {dz:.4f}")
    
    # Check angle differences
    print("\nAngle differences between consecutive P atoms:")
    for i in range(len(p_indices) - 1):
        a1 = np.degrees(np.arctan2(transformed[p_indices[i]][1], transformed[p_indices[i]][0]))
        a2 = np.degrees(np.arctan2(transformed[p_indices[i+1]][1], transformed[p_indices[i+1]][0]))
        da = a2 - a1
        if da > 180: da -= 360
        if da < -180: da += 360
        print(f"  Step {i} -> {i+1}: dangle = {da:.4f}°")
    
    # Now unwind each nucleotide to position 0
    print("\n\nUnwinding nucleotides to position 0...")
    
    # Reconstruct nucleotides in helix frame
    idx = 0
    s1_helix = []
    for base, step, nuc in s1_nucs:
        nuc_transformed = []
        for name, elem, _, _, _ in nuc:
            x, y, z = transformed[idx]
            nuc_transformed.append((name, elem, x, y, z))
            idx += 1
        s1_helix.append((base, step, nuc_transformed))
    
    # Unwind using fitted parameters
    s1_unwound = []
    for base, step, nuc in s1_helix:
        R_inv = rot_z(-twist * step)
        t_vec = np.array([0, 0, rise * step])
        unwound = []
        for name, elem, x, y, z in nuc:
            coord = R_inv @ (np.array([x, y, z]) - t_vec)
            unwound.append((name, elem, coord[0], coord[1], coord[2]))
        s1_unwound.append((base, step, unwound))
    
    # Check consistency: all A nucleotides should be identical after unwinding
    print("\nConsistency check (A nucleotides after unwinding):")
    a_nucs = [(b, s, n) for b, s, n in s1_unwound if b == "A"]
    if len(a_nucs) >= 2:
        ref = np.array([[a[2], a[3], a[4]] for a in a_nucs[0][2]])
        for i in range(1, len(a_nucs)):
            other = np.array([[a[2], a[3], a[4]] for a in a_nucs[i][2]])
            rmsd_check = np.sqrt(np.mean(np.sum((ref - other)**2, axis=1)))
            print(f"  A at step {a_nucs[0][1]} vs step {a_nucs[i][1]}: RMSD = {rmsd_check:.6f} Å")
    
    print("\nConsistency check (T nucleotides after unwinding):")
    t_nucs = [(b, s, n) for b, s, n in s1_unwound if b == "T"]
    if len(t_nucs) >= 2:
        ref = np.array([[a[2], a[3], a[4]] for a in t_nucs[0][2]])
        for i in range(1, len(t_nucs)):
            other = np.array([[a[2], a[3], a[4]] for a in t_nucs[i][2]])
            rmsd_check = np.sqrt(np.mean(np.sum((ref - other)**2, axis=1)))
            print(f"  T at step {t_nucs[0][1]} vs step {t_nucs[i][1]}: RMSD = {rmsd_check:.6f} Å")
    
    # Print position-0 templates
    print("\n\nPosition-0 A template (helix frame):")
    for name, elem, x, y, z in s1_unwound[0][2]:
        print(f"  {name:5s} {elem}: ({x:10.3f}, {y:10.3f}, {z:10.3f})")
    
    # Now do the same for strand II
    print("\n\n=== Strand II ===")
    all_s2_coords = []
    all_s2_names = []
    for base, step, nuc in s2_nucs:
        for name, elem, x, y, z in nuc:
            all_s2_coords.append([x, y, z])
            all_s2_names.append((name, elem, base, step))
    all_s2_coords = np.array(all_s2_coords)
    
    transformed_s2, _ = transform_to_helix_frame(all_s2_coords, axis_pt, axis_dir)
    
    print("Strand II P atoms in helix frame:")
    p_indices_s2 = [i for i, (n, e, b, s) in enumerate(all_s2_names) if n == "P"]
    for pi in p_indices_s2:
        x, y, z = transformed_s2[pi]
        r = np.sqrt(x**2 + y**2)
        angle = np.degrees(np.arctan2(y, x))
        name, elem, base, step = all_s2_names[pi]
        print(f"  Paired step {step} ({base}): ({x:8.3f}, {y:8.3f}, {z:8.3f})  r={r:.3f}  angle={angle:.3f}°")
    
    # Unwind strand II
    idx = 0
    s2_helix = []
    for base, step, nuc in s2_nucs:
        nuc_transformed = []
        for name, elem, _, _, _ in nuc:
            x, y, z = transformed_s2[idx]
            nuc_transformed.append((name, elem, x, y, z))
            idx += 1
        s2_helix.append((base, step, nuc_transformed))
    
    s2_unwound = []
    for base, step, nuc in s2_helix:
        R_inv = rot_z(-twist * step)
        t_vec = np.array([0, 0, rise * step])
        unwound = []
        for name, elem, x, y, z in nuc:
            coord = R_inv @ (np.array([x, y, z]) - t_vec)
            unwound.append((name, elem, coord[0], coord[1], coord[2]))
        s2_unwound.append((base, step, unwound))
    
    print("\nConsistency check (strand II T nucleotides after unwinding):")
    t_nucs_s2 = [(b, s, n) for b, s, n in s2_unwound if b == "T"]
    if len(t_nucs_s2) >= 2:
        ref = np.array([[a[2], a[3], a[4]] for a in t_nucs_s2[0][2]])
        for i in range(1, len(t_nucs_s2)):
            other = np.array([[a[2], a[3], a[4]] for a in t_nucs_s2[i][2]])
            rmsd_check = np.sqrt(np.mean(np.sum((ref - other)**2, axis=1)))
            print(f"  T at paired step {t_nucs_s2[0][1]} vs {t_nucs_s2[i][1]}: RMSD = {rmsd_check:.6f} Å")
    
    # === Now rebuild and check RMSD ===
    print("\n\n=== Rebuild verification ===")
    
    # Get templates from position 0
    s1_template_A = s1_unwound[0][2]  # A at step 0
    s1_template_T = s1_unwound[1][2]  # T at step 1 (unwound to 0)
    s2_template_T = s2_unwound[0][2]  # T at paired step 0
    s2_template_A = s2_unwound[1][2]  # A at paired step 1 (unwound to 0)
    
    templates_s1 = {"A": s1_template_A, "T": s1_template_T}
    templates_s2 = {"A": s2_template_A, "T": s2_template_T}
    
    # Rebuild in helix frame
    rebuilt_s1 = []
    for base, step, _ in s1_helix:
        template = templates_s1[base]
        coords = np.array([[a[2], a[3], a[4]] for a in template])
        R_step = rot_z(twist * step)
        t_step = np.array([0, 0, rise * step])
        transformed_step = (R_step @ coords.T).T + t_step
        for j in range(len(template)):
            rebuilt_s1.append(transformed_step[j])
    
    rebuilt_s1 = np.array(rebuilt_s1)
    ref_s1 = transformed[:len(rebuilt_s1)]
    
    rmsd_s1 = np.sqrt(np.mean(np.sum((rebuilt_s1 - ref_s1)**2, axis=1)))
    print(f"Strand I rebuild RMSD (helix frame): {rmsd_s1:.6f} Å")
    
    # Check max deviation
    devs = np.sqrt(np.sum((rebuilt_s1 - ref_s1)**2, axis=1))
    print(f"  Max deviation: {np.max(devs):.6f} Å")
    print(f"  Mean deviation: {np.mean(devs):.6f} Å")


if __name__ == "__main__":
    main()
