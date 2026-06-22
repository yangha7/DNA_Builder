#!/usr/bin/env python3
"""
Compare DNA Builder V1 (template) and V2 (zmatrix) against real PDB crystal
structures for A-, B-, and Z-form DNA.

Downloads crystal structures from RCSB PDB, extracts middle residues to avoid
end effects, builds the same sequences with both methods, and computes RMSD
after Kabsch superposition.

Crystal structures used:
  B-DNA: 1BNA (CGCGAATTCGCG, 12-mer) — middle 6 bp (residues 4–9)
  A-DNA: 440D (AGGGGCCCCT, 10-mer) — middle 6 bp (residues 3–8)
  Z-DNA: 3P4J (CGCGCG, 6-mer)  — middle 4 bp (residues 2–5)
         fallback: 1DCG (CGCGCG, 6-mer) — middle 4 bp (residues 2–5)
"""

import os
import sys
import urllib.request
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dna_builder.builder import build_dna, Atom
from dna_builder.zmatrix_builder import build_dna_v2


# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------

CRYSTAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "crystal_structures")


def download_pdb(pdb_id: str) -> str:
    """Download a PDB file from RCSB if not already cached."""
    os.makedirs(CRYSTAL_DIR, exist_ok=True)
    filepath = os.path.join(CRYSTAL_DIR, f"{pdb_id.upper()}.pdb")
    if os.path.exists(filepath):
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return ""
    return filepath


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

ATOM_NAME_MAP = {"OP1": "O1P", "OP2": "O2P", "OP3": "O3P"}

RESNAME_MAP = {
    "DA": "DA", "DT": "DT", "DC": "DC", "DG": "DG",
    "A": "DA", "T": "DT", "C": "DC", "G": "DG",
}

RESNAME_TO_BASE = {"DA": "A", "DT": "T", "DC": "C", "DG": "G"}


class CrystalAtom:
    """Parsed atom from a PDB crystal structure."""
    __slots__ = ("name", "element", "x", "y", "z",
                 "residue_name", "residue_seq", "chain_id")

    def __init__(self, name, element, x, y, z,
                 residue_name, residue_seq, chain_id):
        self.name = name
        self.element = element
        self.x = x
        self.y = y
        self.z = z
        self.residue_name = residue_name
        self.residue_seq = residue_seq
        self.chain_id = chain_id


def parse_pdb(filepath: str) -> List[CrystalAtom]:
    """
    Parse a PDB file, returning heavy atoms only.
    Handles alternate conformations (keeps 'A' or first), skips hydrogens,
    normalizes atom/residue names.
    """
    atoms = []
    seen_alt = set()

    with open(filepath) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            alt_loc = line[16]
            res_name = line[17:20].strip()
            chain_id = line[21]
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                element = atom_name[0]

            # Skip hydrogens
            if element in ("H", "D"):
                continue

            # Normalize atom name
            if atom_name in ATOM_NAME_MAP:
                atom_name = ATOM_NAME_MAP[atom_name]

            # Normalize residue name
            if res_name not in RESNAME_MAP:
                continue  # skip non-DNA
            res_name = RESNAME_MAP[res_name]

            # Handle alternate conformations: keep ' ' or 'A' only
            if alt_loc not in (" ", "", "A"):
                continue
            key = (chain_id, res_seq, atom_name)
            if key in seen_alt:
                continue
            seen_alt.add(key)

            atoms.append(CrystalAtom(
                name=atom_name, element=element,
                x=x, y=y, z=z,
                residue_name=res_name, residue_seq=res_seq,
                chain_id=chain_id,
            ))

    return atoms


# ---------------------------------------------------------------------------
# Kabsch RMSD
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute RMSD after optimal superposition (Kabsch algorithm).
    P, Q: Nx3 arrays of corresponding atom coordinates.
    """
    assert P.shape == Q.shape and P.shape[0] > 0
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


# ---------------------------------------------------------------------------
# Atom matching
# ---------------------------------------------------------------------------

BACKBONE_ATOMS = {"P", "O5'", "C5'", "C4'", "C3'", "O3'"}
SKIP_ATOMS = {"O5T", "O3T", "HO5'", "HO3'", "H5T", "H3T"}


def normalize_name(name: str) -> str:
    name = name.strip()
    return ATOM_NAME_MAP.get(name, name)


def _build_residue_atom_dict(atoms, chain, resseq_list):
    """
    Build dict: resseq -> {atom_name: (x,y,z)}
    Only for atoms in the given chain and resseq_list.
    """
    d = defaultdict(dict)
    for a in atoms:
        ch = a.chain_id if hasattr(a, 'chain_id') else ""
        rs = a.residue_seq if hasattr(a, 'residue_seq') else 0
        if ch != chain or rs not in resseq_list:
            continue
        aname = normalize_name(a.name)
        if aname in SKIP_ATOMS:
            continue
        d[rs][aname] = np.array([a.x, a.y, a.z])
    return dict(d)


def match_paired_residues(
    crystal_atoms, crystal_chain, crystal_resseqs,
    builder_atoms, builder_chain, builder_resseqs,
    backbone_only=False,
):
    """
    Match atoms between crystal and builder for a list of paired residues.

    crystal_resseqs[i] and builder_resseqs[i] correspond to the same
    base-pair position.

    Returns (crystal_coords, builder_coords, n_matched).
    """
    assert len(crystal_resseqs) == len(builder_resseqs), \
        f"Residue list length mismatch: {len(crystal_resseqs)} vs {len(builder_resseqs)}"

    c_dict = _build_residue_atom_dict(crystal_atoms, crystal_chain,
                                       set(crystal_resseqs))
    b_dict = _build_residue_atom_dict(builder_atoms, builder_chain,
                                       set(builder_resseqs))

    c_coords = []
    b_coords = []

    for c_rs, b_rs in zip(crystal_resseqs, builder_resseqs):
        c_atoms = c_dict.get(c_rs, {})
        b_atoms = b_dict.get(b_rs, {})

        common = set(c_atoms.keys()) & set(b_atoms.keys())
        if backbone_only:
            common = common & BACKBONE_ATOMS

        for aname in sorted(common):
            c_coords.append(c_atoms[aname])
            b_coords.append(b_atoms[aname])

    if not c_coords:
        return np.zeros((0, 3)), np.zeros((0, 3)), 0

    return np.array(c_coords), np.array(b_coords), len(c_coords)


# ---------------------------------------------------------------------------
# Crystal structure definitions
# ---------------------------------------------------------------------------

class CrystalCase:
    """
    Defines one comparison case: a crystal structure, the middle portion
    to compare, and how residues map between crystal and builder.
    """
    def __init__(self, pdb_id, form, full_sequence, description,
                 crystal_chain_a, crystal_a_all,
                 crystal_chain_b, crystal_b_all,
                 middle_a_crystal, middle_b_crystal):
        """
        Parameters
        ----------
        pdb_id : str
        form : str  ("A", "B", or "Z")
        full_sequence : str  (strand I, 5'→3')
        description : str
        crystal_chain_a, crystal_chain_b : str
            Chain IDs in the crystal structure.
        crystal_a_all, crystal_b_all : list of int
            All residue sequence numbers for chains A and B in the crystal,
            listed in 5'→3' order for each strand.
        middle_a_crystal : list of int
            Crystal strand-A residue numbers for the middle portion (5'→3').
        middle_b_crystal : list of int
            Crystal strand-B residue numbers for the middle portion,
            listed in *pairing order* with middle_a_crystal.
            i.e. middle_b_crystal[i] is the complement of middle_a_crystal[i].
        """
        self.pdb_id = pdb_id
        self.form = form
        self.full_sequence = full_sequence
        self.description = description
        self.crystal_chain_a = crystal_chain_a
        self.crystal_a_all = crystal_a_all
        self.crystal_chain_b = crystal_chain_b
        self.crystal_b_all = crystal_b_all
        self.middle_a_crystal = middle_a_crystal
        self.middle_b_crystal = middle_b_crystal

    @property
    def n_middle(self):
        return len(self.middle_a_crystal)

    @property
    def middle_seq(self):
        """Sequence of the middle portion (strand A, 5'→3')."""
        n = len(self.full_sequence)
        a_all = self.crystal_a_all
        indices = [a_all.index(r) for r in self.middle_a_crystal]
        return "".join(self.full_sequence[i] for i in indices)

    def builder_middle_a(self):
        """Builder strand-A residue numbers for the middle portion."""
        # Builder always numbers strand A as 1..N
        a_all = self.crystal_a_all
        # Offset of middle within full chain
        start_idx = a_all.index(self.middle_a_crystal[0])
        return list(range(start_idx + 1, start_idx + 1 + self.n_middle))

    def builder_middle_b(self):
        """
        Builder strand-B residue numbers for the middle portion,
        in pairing order with builder_middle_a.

        Builder strand B: residue (N-i) pairs with strand A residue (i+1),
        where i is 0-based and N = len(full_sequence).
        """
        N = len(self.full_sequence)
        a_all = self.crystal_a_all
        result = []
        for c_a_rs in self.middle_a_crystal:
            # 0-based index of this residue in the full chain
            idx = a_all.index(c_a_rs)
            # Builder strand A residue number: idx + 1
            # Paired builder strand B residue: N - idx
            result.append(N - idx)
        return result


# ---- B-DNA: 1BNA ----
# Chain A: 1-12 (CGCGAATTCGCG, 5'→3')
# Chain B: 13-24 (CGCGAATTCGCG, 5'→3' of strand B)
# Pairing: A:1↔B:24, A:2↔B:23, ..., A:12↔B:13
# Middle 6 bp: A:4-9 (GAATTC)
# Paired B: 21,20,19,18,17,16
CASE_1BNA = CrystalCase(
    pdb_id="1BNA", form="B",
    full_sequence="CGCGAATTCGCG",
    description="Dickerson dodecamer B-DNA (12-mer)",
    crystal_chain_a="A", crystal_a_all=list(range(1, 13)),
    crystal_chain_b="B", crystal_b_all=list(range(13, 25)),
    middle_a_crystal=[4, 5, 6, 7, 8, 9],
    middle_b_crystal=[21, 20, 19, 18, 17, 16],
)

# ---- A-DNA: 440D ----
# Chain A: 1-10 (AGGGGCCCCT, 5'→3')
# Chain B: 11-20 (AGGGGCCCCT, 5'→3' of strand B)
# Pairing: A:1↔B:20, A:2↔B:19, ..., A:10↔B:11
# Middle 6 bp: A:3-8 (GGGCCC)
# Paired B: 18,17,16,15,14,13
CASE_440D = CrystalCase(
    pdb_id="440D", form="A",
    full_sequence="AGGGGCCCCT",
    description="A-DNA decamer (10-mer)",
    crystal_chain_a="A", crystal_a_all=list(range(1, 11)),
    crystal_chain_b="B", crystal_b_all=list(range(11, 21)),
    middle_a_crystal=[3, 4, 5, 6, 7, 8],
    middle_b_crystal=[18, 17, 16, 15, 14, 13],
)

# ---- Z-DNA: 3P4J ----
# Crystal: Chain A: 1-6 (CGCGCG, 5'→3'), Chain B: 7-12 (CGCGCG, 5'→3')
# Pairing: A:1↔B:12, A:2↔B:11, ..., A:6↔B:7
# Builder builds GCGCGC (canonical pur-pyr alternation for Z-DNA).
# Crystal A:2(G)↔builder A:1(G), crystal A:3(C)↔builder A:2(C), etc.
# Middle 4 bp: crystal A:3,4,5,6 ↔ builder A:2,3,4,5
# Paired B: crystal B:10,9,8,7 ↔ builder B:5,4,3,2
CASE_3P4J = CrystalCase(
    pdb_id="3P4J", form="Z",
    full_sequence="GCGCGC",
    description="Z-DNA hexamer (6-mer)",
    crystal_chain_a="A", crystal_a_all=[2, 3, 4, 5, 6, 1],
    crystal_chain_b="B", crystal_b_all=[11, 10, 9, 8, 7, 12],
    middle_a_crystal=[3, 4, 5, 6],
    middle_b_crystal=[10, 9, 8, 7],
)

# ---- Z-DNA fallback: 1DCG ----
# Same layout as 3P4J
CASE_1DCG = CrystalCase(
    pdb_id="1DCG", form="Z",
    full_sequence="GCGCGC",
    description="Z-DNA hexamer (6-mer, 1DCG)",
    crystal_chain_a="A", crystal_a_all=[2, 3, 4, 5, 6, 1],
    crystal_chain_b="B", crystal_b_all=[11, 10, 9, 8, 7, 12],
    middle_a_crystal=[3, 4, 5, 6],
    middle_b_crystal=[10, 9, 8, 7],
)


# ---------------------------------------------------------------------------
# Inspect crystal structure
# ---------------------------------------------------------------------------

def inspect_crystal(atoms: List[CrystalAtom], case: CrystalCase):
    """Print crystal structure layout for verification."""
    residues = defaultdict(list)
    for a in atoms:
        residues[(a.chain_id, a.residue_seq)].append(a)

    chains = defaultdict(list)
    for (ch, rs) in sorted(residues.keys()):
        chains[ch].append(rs)

    print(f"\n  Crystal structure {case.pdb_id}:")
    for ch in sorted(chains.keys()):
        res_list = sorted(chains[ch])
        seq = []
        for r in res_list:
            rn = residues[(ch, r)][0].residue_name
            seq.append(RESNAME_TO_BASE.get(rn, "?"))
        print(f"    Chain {ch}: residues {res_list[0]}-{res_list[-1]} "
              f"({len(res_list)} nt), seq: {''.join(seq)}")

    # Verify middle residues exist
    for rs in case.middle_a_crystal:
        if (case.crystal_chain_a, rs) not in residues:
            print(f"    WARNING: missing A:{rs}")
    for rs in case.middle_b_crystal:
        if (case.crystal_chain_b, rs) not in residues:
            print(f"    WARNING: missing B:{rs}")


# ---------------------------------------------------------------------------
# Run one comparison
# ---------------------------------------------------------------------------

def compare_one(case: CrystalCase) -> Optional[Dict]:
    """Run comparison for one crystal structure. Returns dict of results."""
    print(f"\n{'='*70}")
    print(f"  {case.pdb_id}: {case.description}")
    print(f"  Form: {case.form}-DNA | Full sequence: {case.full_sequence}")
    print(f"{'='*70}")

    filepath = download_pdb(case.pdb_id)
    if not filepath:
        print("  SKIPPED: download failed")
        return None

    crystal_atoms = parse_pdb(filepath)
    if not crystal_atoms:
        print("  SKIPPED: no DNA atoms parsed")
        return None

    inspect_crystal(crystal_atoms, case)

    mid_seq = case.middle_seq
    print(f"\n  Middle {case.n_middle} bp: {mid_seq}")
    print(f"    Crystal strand A: {case.middle_a_crystal}")
    print(f"    Crystal strand B (pairing order): {case.middle_b_crystal}")

    bld_a = case.builder_middle_a()
    bld_b = case.builder_middle_b()
    print(f"    Builder strand A: {bld_a}")
    print(f"    Builder strand B (pairing order): {bld_b}")

    # Build full sequence (suppress expected Z-DNA warnings)
    import warnings
    print(f"\n  Building {case.full_sequence} ({case.form}-form)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            v1_atoms = build_dna(case.full_sequence, case.form)
            print(f"    V1 (template): {len(v1_atoms)} atoms")
        except Exception as e:
            print(f"    V1 ERROR: {e}")
            v1_atoms = None

        try:
            v2_atoms = build_dna_v2(case.full_sequence, case.form)
            print(f"    V2 (zmatrix):  {len(v2_atoms)} atoms")
        except Exception as e:
            print(f"    V2 ERROR: {e}")
            v2_atoms = None

    results = {}

    for label, builder_atoms in [("V1", v1_atoms), ("V2", v2_atoms)]:
        if builder_atoms is None:
            continue

        for bb_only in [False, True]:
            scope_tag = "backbone" if bb_only else "all-atom"

            # ---- Strand A only ----
            c_c, b_c, n = match_paired_residues(
                crystal_atoms, case.crystal_chain_a, case.middle_a_crystal,
                builder_atoms, "A", bld_a,
                backbone_only=bb_only,
            )
            if n > 0:
                rmsd = kabsch_rmsd(c_c, b_c)
                results[f"{label}_strandA_{scope_tag}"] = (rmsd, n)
            else:
                results[f"{label}_strandA_{scope_tag}"] = (float('nan'), 0)

            # ---- Both strands ----
            # Strand A
            c_a, b_a, n_a = match_paired_residues(
                crystal_atoms, case.crystal_chain_a, case.middle_a_crystal,
                builder_atoms, "A", bld_a,
                backbone_only=bb_only,
            )
            # Strand B
            c_b, b_b, n_b = match_paired_residues(
                crystal_atoms, case.crystal_chain_b, case.middle_b_crystal,
                builder_atoms, "B", bld_b,
                backbone_only=bb_only,
            )

            n_total = n_a + n_b
            if n_total > 0:
                parts_c = [x for x in [c_a, c_b] if x.shape[0] > 0]
                parts_b = [x for x in [b_a, b_b] if x.shape[0] > 0]
                c_all = np.vstack(parts_c)
                b_all = np.vstack(parts_b)
                rmsd = kabsch_rmsd(c_all, b_all)
                results[f"{label}_both_{scope_tag}"] = (rmsd, n_total)
            else:
                results[f"{label}_both_{scope_tag}"] = (float('nan'), 0)

    # Print per-structure results
    print(f"\n  Results for {case.pdb_id}:")
    print(f"  {'Method':<6} {'Scope':<12} {'Atoms':>6}  "
          f"{'All-atom RMSD':>14}  {'Backbone RMSD':>14}")
    print(f"  {'-'*58}")
    for method in ["V1", "V2"]:
        for scope_label, scope_key in [("Strand A", "strandA"),
                                        ("Both", "both")]:
            aa = results.get(f"{method}_{scope_key}_all-atom", (float('nan'), 0))
            bb = results.get(f"{method}_{scope_key}_backbone", (float('nan'), 0))
            aa_s = f"{aa[0]:.3f} Å" if not np.isnan(aa[0]) else "N/A"
            bb_s = f"{bb[0]:.3f} Å" if not np.isnan(bb[0]) else "N/A"
            print(f"  {method:<6} {scope_label:<12} {aa[1]:>6}  "
                  f"{aa_s:>14}  {bb_s:>14}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: Dict[str, Tuple[CrystalCase, Dict]]):
    """Print a compact summary table."""
    print("\n")
    print("=" * 90)
    print("  SUMMARY: DNA Builder vs PDB Crystal Structures")
    print("  (RMSD in Å after Kabsch superposition, middle residues only)")
    print("=" * 90)

    # Detailed table
    print(f"\n  {'PDB':<6} {'Form':<6} {'Mid':<6} {'Method':<7} "
          f"{'Strand A':>10} {'Both':>10} "
          f"{'BB Str.A':>10} {'BB Both':>10}")
    print(f"  {'-'*72}")

    for pdb_id, (case, results) in all_results.items():
        if results is None:
            print(f"  {pdb_id:<6} {case.form:<6} {'—':<6} {'SKIP':<7}")
            continue
        for method in ["V1", "V2"]:
            sa_aa = results.get(f"{method}_strandA_all-atom", (float('nan'), 0))[0]
            bo_aa = results.get(f"{method}_both_all-atom", (float('nan'), 0))[0]
            sa_bb = results.get(f"{method}_strandA_backbone", (float('nan'), 0))[0]
            bo_bb = results.get(f"{method}_both_backbone", (float('nan'), 0))[0]

            def fmt(v):
                return f"{v:.3f}" if not np.isnan(v) else "N/A"

            print(f"  {pdb_id:<6} {case.form:<6} {case.n_middle:>2} bp "
                  f" {method:<7} "
                  f"{fmt(sa_aa):>10} {fmt(bo_aa):>10} "
                  f"{fmt(sa_bb):>10} {fmt(bo_bb):>10}")

    print(f"  {'-'*72}")

    # Compact comparison
    print(f"\n  {'Form':<6} {'PDB':<6} "
          f"{'V1 (all)':>10} {'V2 (all)':>10} "
          f"{'V1 (BB)':>10} {'V2 (BB)':>10}  (both strands)")
    print(f"  {'-'*56}")
    for pdb_id, (case, results) in all_results.items():
        if results is None:
            continue
        v1 = results.get("V1_both_all-atom", (float('nan'), 0))[0]
        v2 = results.get("V2_both_all-atom", (float('nan'), 0))[0]
        v1b = results.get("V1_both_backbone", (float('nan'), 0))[0]
        v2b = results.get("V2_both_backbone", (float('nan'), 0))[0]

        def fmt(v):
            return f"{v:.3f}" if not np.isnan(v) else "N/A"

        print(f"  {case.form:<6} {pdb_id:<6} "
              f"{fmt(v1):>10} {fmt(v2):>10} "
              f"{fmt(v1b):>10} {fmt(v2b):>10}")
    print(f"  {'-'*56}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  DNA Builder: Comparison with PDB Crystal Structures")
    print("  V1 (template) vs V2 (zmatrix) vs Experiment")
    print("=" * 70)

    all_results = {}

    # B-DNA
    r = compare_one(CASE_1BNA)
    all_results["1BNA"] = (CASE_1BNA, r)

    # A-DNA
    r = compare_one(CASE_440D)
    all_results["440D"] = (CASE_440D, r)

    # Z-DNA: compare against all available structures
    for z_case in [CASE_3P4J, CASE_1DCG]:
        fp = download_pdb(z_case.pdb_id)
        if not fp:
            continue
        atoms = parse_pdb(fp)
        if len(atoms) < 10:
            continue
        chains = set(a.chain_id for a in atoms)
        if z_case.crystal_chain_b not in chains:
            print(f"  {z_case.pdb_id}: strand B chain not found, skipping")
            continue
        r = compare_one(z_case)
        if r is not None:
            all_results[z_case.pdb_id] = (z_case, r)

    print_summary(all_results)


if __name__ == "__main__":
    main()
