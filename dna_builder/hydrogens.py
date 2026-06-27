"""
Add explicit hydrogen atoms to DNA structures.

Hydrogen positions are computed purely from the heavy-atom coordinates using
standard local bond geometry:

  sp3 C with 3 heavy bonds → 1 H  (tetrahedral opposite the three bonds)
  sp3 C with 2 heavy bonds → 2 H  (tetrahedral CH2)
  sp2 ring C/N with 2 bonds → 1 H  (in-plane, opposite bisector)
  ring N-H with 2 bonds   → 1 H  (in-plane, opposite bisector)
  amino N with 1 C bond   → 2 H  (planar, C-N-H = 120°)
  sp3 CH3 (1 C bond)      → 3 H  (tetrahedral, staggered)

Bond lengths used (Å): C-H = 1.09, N-H = 1.01, O-H = 0.98
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict

from .builder import Atom

# Bond lengths (Å)
_CH = 1.09
_NH = 1.01
_OH = 0.98


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _n(v: np.ndarray) -> np.ndarray:
    """Safe unit-vector normalization."""
    m = float(np.linalg.norm(v))
    return v / m if m > 1e-10 else v


def _sp3_1h(center: np.ndarray, n1, n2, n3, bl: float = _CH) -> np.ndarray:
    """One H at sp3 center with 3 known heavy-atom neighbours."""
    v = _n(n1 - center) + _n(n2 - center) + _n(n3 - center)
    return center + bl * _n(-v)


def _sp3_2h(center: np.ndarray, n1, n2, bl: float = _CH) -> Tuple[np.ndarray, np.ndarray]:
    """Two H at sp3 CH2 centre with 2 known heavy-atom bonds."""
    v1, v2 = _n(n1 - center), _n(n2 - center)
    neg_bis = _n(-(v1 + v2))
    cross   = np.cross(v1, v2)
    if np.linalg.norm(cross) < 1e-6:
        # Degenerate (collinear) – use an arbitrary perpendicular
        perp = np.array([0., 0., 1.])
        if abs(np.dot(v1, perp)) > 0.9:
            perp = np.array([0., 1., 0.])
        cross = np.cross(v1, perp)
    perp = _n(cross)
    # For exact tetrahedral: a = 1/√3, b = √(2/3)
    a = 1.0 / np.sqrt(3.0)
    b = np.sqrt(2.0 / 3.0)
    return (center + bl * _n(a * neg_bis + b * perp),
            center + bl * _n(a * neg_bis - b * perp))


def _sp2_1h(center: np.ndarray, n1, n2, bl: float = _CH) -> np.ndarray:
    """One H at sp2 centre (ring C-H or ring N-H): in-plane, opposite bisector."""
    v1, v2 = _n(n1 - center), _n(n2 - center)
    return center + bl * _n(-(v1 + v2))


def _amino_2h(n_center: np.ndarray, c_bonded, ring_ref,
              bl: float = _NH) -> Tuple[np.ndarray, np.ndarray]:
    """Two H at planar amino N (C-N-H = 120°).

    Parameters
    ----------
    n_center  : amino nitrogen position
    c_bonded  : the sp2 carbon bonded to this N
    ring_ref  : any second atom that defines the base plane
    """
    vc   = _n(c_bonded - n_center)          # N → C
    rdir = _n(ring_ref  - n_center)
    normal = _n(np.cross(vc, rdir))          # out-of-plane normal
    perp   = _n(np.cross(normal, vc))        # in-plane, ⊥ to N-C

    # H-N-C = 120°  ⟹  H = -½vc ± (√3/2)·perp
    h1 = n_center + bl * _n(-0.5 * vc + (np.sqrt(3.0) / 2.0) * perp)
    h2 = n_center + bl * _n(-0.5 * vc - (np.sqrt(3.0) / 2.0) * perp)
    return h1, h2


def _methyl_3h(center: np.ndarray, c_bonded,
               bl: float = _CH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three H at methyl carbon (1 heavy-atom bond to c_bonded)."""
    axis = _n(c_bonded - center)        # bond axis pointing toward parent C

    # Build an arbitrary perpendicular
    perp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, perp)) > 0.9:
        perp = np.array([0.0, 1.0, 0.0])
    perp = _n(perp - np.dot(perp, axis) * axis)

    # Tetrahedral angle from bond axis: cos θ = −1/3
    cos_t, sin_t = -1.0 / 3.0, np.sqrt(8.0 / 9.0)
    hs = []
    for k in range(3):
        phi     = k * 2.0 * np.pi / 3.0
        h_dir   = cos_t * axis + sin_t * (np.cos(phi) * perp +
                                           np.sin(phi) * np.cross(axis, perp))
        hs.append(center + bl * _n(h_dir))
    return tuple(hs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _place_terminal_oh(o_pos: np.ndarray, c_bonded, c_ref,
                       bl: float = _OH) -> np.ndarray:
    """Place H on a terminal hydroxyl O with one heavy-atom bond (e.g. 3'-OH).

    H-O-C angle = tetrahedral (109.47°), placed *trans* to c_ref about the
    C-O bond axis (H-O-C-Cref dihedral = 180°).
    """
    vc    = _n(c_bonded - o_pos)                         # O→C
    v4    = _n(c_ref - c_bonded)                         # C→Cref
    v4_perp = _n(v4 - np.dot(v4, vc) * vc)              # perp to O-C axis
    # H·vc = cos(109.47°) = -1/3; trans → opposite to v4_perp
    h_dir = _n((-1.0 / 3.0) * vc - np.sqrt(8.0 / 9.0) * v4_perp)
    return o_pos + bl * h_dir


def add_hydrogens(atoms: List[Atom]) -> List[Atom]:
    """Return a new atom list with explicit H atoms appended.

    Charge model: phosphate oxygens (O1P, O2P, O5T) are left without H,
    preserving −1 formal charge per nucleotide.  Only the 3′-terminal O3′
    receives HO3′ (neutral free hydroxyl, no charge impact).
    """
    # ── group residues ────────────────────────────────────────────────────
    residues: dict = defaultdict(dict)   # (chain, resseq) → {name: coords}
    res_meta: dict = {}                  # (chain, resseq) → (resname, chain_id, resseq)
    for a in atoms:
        key = (a.chain_id, a.residue_seq)
        residues[key][a.name] = np.array([a.x, a.y, a.z], dtype=float)
        res_meta[key] = (a.residue_name, a.chain_id, a.residue_seq)

    # ── find terminal residues ────────────────────────────────────────────
    # Strand A: 1→n (5'→3'); 3′ terminal = max resseq, 5′ terminal = min resseq.
    # Strand B: n→1 (5'→3'); 3′ terminal = min resseq, 5′ terminal = max resseq.
    chain_resseqs: dict = defaultdict(list)
    for (ch, rs) in residues:
        chain_resseqs[ch].append(rs)
    terminal_3prime = set()
    terminal_5prime = set()
    for ch, rss in chain_resseqs.items():
        # Both chains: residue 1 = 5' terminal, residue n = 3' terminal.
        # Chain B runs antiparallel in 3D but its residue numbers still increase 5'→3'.
        terminal_3prime.add((ch, max(rss)))
        terminal_5prime.add((ch, min(rss)))

    # ── helper: make an Atom ──────────────────────────────────────────────
    def H(name: str, xyz, resname: str, chain: str, resseq: int) -> Atom:
        return Atom(name=name, element='H',
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    residue_name=resname, residue_seq=resseq, chain_id=chain)

    new_h: List[Atom] = []

    for key in sorted(residues, key=lambda k: (k[0], k[1])):
        r = residues[key]          # atom_name → coords
        resname, chain, resseq = res_meta[key]

        def c(name):
            """Coordinates of named atom in this residue, or None."""
            return r.get(name)

        hs: List[Atom] = []
        mk = lambda name, xyz: H(name, xyz, resname, chain, resseq)  # noqa: E731

        # ── Sugar ────────────────────────────────────────────────────────

        # C1'–H1'  (sp3, bonds to C2', O4', N9 or N1)
        if c("C1'") is not None:
            n_gly = c("N9") if "N9" in r else c("N1")
            if c("C2'") is not None and c("O4'") is not None and n_gly is not None:
                hs.append(mk("H1'", _sp3_1h(c("C1'"), c("C2'"), c("O4'"), n_gly)))

        # C2'–H2', H2''  (sp3 CH2, bonds to C1' and C3')
        if c("C2'") is not None and c("C1'") is not None and c("C3'") is not None:
            h1, h2 = _sp3_2h(c("C2'"), c("C1'"), c("C3'"))
            hs.append(mk("H2'",  h1))
            hs.append(mk("H2''", h2))

        # C3'–H3'  (sp3, bonds to C2', C4', O3')
        if c("C3'") is not None and c("C2'") is not None and \
                c("C4'") is not None and c("O3'") is not None:
            hs.append(mk("H3'", _sp3_1h(c("C3'"), c("C2'"), c("C4'"), c("O3'"))))

        # C4'–H4'  (sp3, bonds to C3', C5', O4')
        if c("C4'") is not None and c("C3'") is not None and \
                c("C5'") is not None and c("O4'") is not None:
            hs.append(mk("H4'", _sp3_1h(c("C4'"), c("C3'"), c("C5'"), c("O4'"))))

        # C5'–H5', H5''  (sp3 CH2, bonds to C4' and O5')
        if c("C5'") is not None and c("C4'") is not None and c("O5'") is not None:
            h1, h2 = _sp3_2h(c("C5'"), c("C4'"), c("O5'"))
            hs.append(mk("H5'",  h1))
            hs.append(mk("H5''", h2))

        # O3'–HO3'  (3′-terminal free hydroxyl only — NOT for internal residues,
        # whose O3' bridges to the next nucleotide's phosphorus)
        if key in terminal_3prime:
            if c("O3'") is not None and c("C3'") is not None and c("C4'") is not None:
                hs.append(mk("HO3'", _place_terminal_oh(c("O3'"), c("C3'"), c("C4'"))))

        # 5′-terminal phosphate proton: one of the non-bridging oxygens is
        # protonated to keep the terminal phosphate at −1 (same as internal).
        # Convention (matching Colin/3DNA): chain A → H on O1P; chain B → H on O2P.
        if key in terminal_5prime:
            if chain == 'A':
                if c("O1P") is not None and c("P") is not None and c("O2P") is not None:
                    hs.append(mk("HO1P", _place_terminal_oh(c("O1P"), c("P"), c("O2P"))))
            else:
                if c("O2P") is not None and c("P") is not None and c("O1P") is not None:
                    hs.append(mk("HO2P", _place_terminal_oh(c("O2P"), c("P"), c("O1P"))))

        # ── Base ─────────────────────────────────────────────────────────

        if resname == 'DA':
            # C2–H2  (sp2, between N1 and N3)
            if c("C2") is not None and c("N1") is not None and c("N3") is not None:
                hs.append(mk("H2", _sp2_1h(c("C2"), c("N1"), c("N3"))))
            # C8–H8  (sp2, between N7 and N9)
            if c("C8") is not None and c("N7") is not None and c("N9") is not None:
                hs.append(mk("H8", _sp2_1h(c("C8"), c("N7"), c("N9"))))
            # N6–H61, H62  (amino; ring ref = N1 to orient the plane)
            if c("N6") is not None and c("C6") is not None and c("N1") is not None:
                h1, h2 = _amino_2h(c("N6"), c("C6"), c("N1"))
                hs.append(mk("H61", h1))
                hs.append(mk("H62", h2))

        elif resname == 'DT':
            # N3–H3  (ring NH, between C2 and C4)
            if c("N3") is not None and c("C2") is not None and c("C4") is not None:
                hs.append(mk("H3", _sp2_1h(c("N3"), c("C2"), c("C4"), )))
            # C6–H6  (sp2, between C5 and N1)
            if c("C6") is not None and c("C5") is not None and c("N1") is not None:
                hs.append(mk("H6", _sp2_1h(c("C6"), c("C5"), c("N1"))))
            # C5M–H51, H52, H53  (methyl, bonded to C5)
            if c("C5M") is not None and c("C5") is not None:
                h1, h2, h3 = _methyl_3h(c("C5M"), c("C5"))
                hs.append(mk("H51", h1))
                hs.append(mk("H52", h2))
                hs.append(mk("H53", h3))

        elif resname == 'DG':
            # N1–H1  (ring NH, between C2 and C6)
            if c("N1") is not None and c("C2") is not None and c("C6") is not None:
                hs.append(mk("H1", _sp2_1h(c("N1"), c("C2"), c("C6"), )))
            # C8–H8  (sp2, between N7 and N9)
            if c("C8") is not None and c("N7") is not None and c("N9") is not None:
                hs.append(mk("H8", _sp2_1h(c("C8"), c("N7"), c("N9"))))
            # N2–H21, H22  (amino; ring ref = N1)
            if c("N2") is not None and c("C2") is not None and c("N1") is not None:
                h1, h2 = _amino_2h(c("N2"), c("C2"), c("N1"))
                hs.append(mk("H21", h1))
                hs.append(mk("H22", h2))

        elif resname == 'DC':
            # N4–H41, H42  (amino; ring ref = N3)
            if c("N4") is not None and c("C4") is not None and c("N3") is not None:
                h1, h2 = _amino_2h(c("N4"), c("C4"), c("N3"))
                hs.append(mk("H41", h1))
                hs.append(mk("H42", h2))
            # C5–H5  (sp2, between C4 and C6)
            if c("C5") is not None and c("C4") is not None and c("C6") is not None:
                hs.append(mk("H5", _sp2_1h(c("C5"), c("C4"), c("C6"))))
            # C6–H6  (sp2, between C5 and N1)
            if c("C6") is not None and c("C5") is not None and c("N1") is not None:
                hs.append(mk("H6", _sp2_1h(c("C6"), c("C5"), c("N1"))))

        new_h.extend(hs)

    return list(atoms) + new_h
