"""Flask backend for DNA Builder GUI."""

import io
import os
import sys
import tempfile
from collections import defaultdict
from typing import Optional

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_builder.builder import build_dna
from dna_builder.classifier import classify_structure
from dna_builder.io_pdb import write_pdb, write_xyz, write_mmcif
from dna_builder.io_parser import parse_structure, detect_format
from dna_builder.zmatrix_builder import build_dna_v2

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/build", methods=["POST"])
def api_build():
    data = request.get_json()
    sequence = data.get("sequence", "").upper().strip()
    form = data.get("form", "B")
    method = data.get("method", "zmatrix")

    hydrogens = data.get("hydrogens", True)

    if not sequence:
        return jsonify({"error": "No sequence provided"}), 400
    bad = set(sequence) - set("ATGC")
    if bad:
        return jsonify({"error": f"Invalid characters: {', '.join(sorted(bad))}"}), 400
    if len(sequence) < 2:
        return jsonify({"error": "Sequence must be at least 2 bases"}), 400
    if form == "Z" and len(sequence) % 2 != 0:
        return jsonify({"error": "Z-DNA requires an even-length sequence"}), 400
    if form == "Z":
        comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
        rev_comp = "".join(comp[b] for b in reversed(sequence))
        if rev_comp != sequence:
            return jsonify({"error": f"Z-DNA requires a palindromic sequence. '{sequence}' has reverse complement '{rev_comp}'."}), 400
        purines = set("AG")
        is_canonical = all((sequence[i] in purines) == (i % 2 == 0) for i in range(len(sequence)))

    try:
        atoms = build_dna_v2(sequence, form) if method == "zmatrix" else build_dna(sequence, form, method="template")
        if hydrogens:
            from dna_builder.hydrogens import add_hydrogens
            atoms = add_hydrogens(atoms)

        buf = io.StringIO()
        write_pdb(atoms, buf, f"{form}-DNA 5'-{sequence}-3'")
        pdb_str = buf.getvalue()

        fingerprint = _compute_fingerprint(atoms, sequence, form)
        warn = None
        if form == "Z" and not is_canonical:
            warn = f"'{sequence}' does not alternate purine-pyrimidine — structure quality may be reduced."
        return jsonify({"pdb": pdb_str, "fingerprint": fingerprint, "ok": True, "warning": warn})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/convert", methods=["POST"])
def api_convert():
    """Convert an uploaded file (CIF, XYZ, PDB) to PDB string."""
    data = request.get_json()
    content = data.get("content", "")
    filename = data.get("filename", "structure.pdb")
    if not content:
        return jsonify({"error": "No file content provided"}), 400

    ext = os.path.splitext(filename)[1].lower() or ".pdb"
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as fh:
        fh.write(content)
        tmp_path = fh.name
    try:
        atoms = parse_structure(tmp_path)
        buf = io.StringIO()
        write_pdb(atoms, buf, filename)
        return jsonify({"pdb": buf.getvalue(), "ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        os.unlink(tmp_path)


@app.route("/api/export", methods=["POST"])
def api_export():
    """Export current PDB content to a requested format."""
    data = request.get_json()
    pdb_content = data.get("pdb", "")
    fmt = data.get("format", "pdb").lower().strip(".")
    title = data.get("title", "DNA structure")
    if not pdb_content:
        return jsonify({"error": "No structure provided"}), 400

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(pdb_content)
        tmp_path = fh.name
    try:
        from dna_builder.io_parser import parse_pdb
        atoms = parse_pdb(tmp_path)
        buf = io.StringIO()
        if fmt == "xyz":
            write_xyz(atoms, buf, title)
            ext = "xyz"
        elif fmt in ("cif", "mmcif"):
            write_mmcif(atoms, buf, title)
            ext = "cif"
        else:
            write_pdb(atoms, buf, title)
            ext = "pdb"
        return jsonify({"content": buf.getvalue(), "ext": ext, "ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        os.unlink(tmp_path)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json()
    pdb_content = data.get("pdb", "")
    if not pdb_content:
        return jsonify({"error": "No structure loaded"}), 400

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(pdb_content)
        tmp_path = fh.name
    try:
        result = classify_structure(tmp_path)
        return jsonify(_jsonify_dict(result))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Helical parameter visualization
# ---------------------------------------------------------------------------

from dna_builder.fiber_data import BASE_RING_ATOMS as _BASE_RING_ATOMS


def _fit_bp_plane(atoms, ch_a: str, res_a: int,
                  ch_b: Optional[str], res_b: Optional[int]):
    """
    Fit a plane through the aromatic ring atoms of a Watson–Crick base pair.

    Returns (normal, centroid, pts) where pts is the (N,3) array of ring
    atom coordinates and normal points in the approximate helix-axis direction.
    Returns (None, None, empty_array) if fewer than 3 ring atoms are found.
    """
    pts = []
    for a in atoms:
        match_a = (a.chain_id == ch_a and a.residue_seq == res_a)
        match_b = (ch_b is not None and a.chain_id == ch_b and a.residue_seq == res_b)
        if match_a or match_b:
            if a.name in _BASE_RING_ATOMS.get(a.residue_name, []):
                pts.append(np.array([a.x, a.y, a.z]))
    if len(pts) < 3:
        return None, None, np.empty((0, 3))
    pts_arr = np.array(pts)
    centroid = pts_arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_arr - centroid)
    normal = Vt[-1]; normal /= np.linalg.norm(normal)
    return normal, centroid, pts_arr


def _plane_mesh(normal: np.ndarray, centroid: np.ndarray,
                pts: np.ndarray, pad: float = 1.8):
    """
    Build a double-sided triangle mesh for a base-pair plane surface.

    Projects ring atom positions onto the plane, extends each point `pad` Å
    outward from the centroid, sorts by angle, and fan-triangulates from
    the centroid.  Returns (vertices_flat, faces_flat) suitable for
    viewer.addCustom().
    """
    tmp = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 1., 0.])
    pu = np.cross(normal, tmp); pu /= np.linalg.norm(pu)
    pv = np.cross(normal, pu);  pv /= np.linalg.norm(pv)

    # Project pts onto the plane, then push each point `pad` Å outward
    rel = pts - centroid
    in_plane = rel - np.outer(rel @ normal, normal)
    norms = np.linalg.norm(in_plane, axis=1, keepdims=True)
    norms_safe = np.where(norms < 0.1, 1.0, norms)
    extended = centroid + in_plane + (in_plane / norms_safe) * pad

    # Sort by angle in the plane for a convex polygon
    x2d = (extended - centroid) @ pu
    y2d = (extended - centroid) @ pv
    order = np.argsort(np.arctan2(y2d, x2d))
    ordered = extended[order]

    # Flat vertex list: centroid at index 0, ring points at 1..n
    verts = centroid.tolist() + [c for pt in ordered for c in pt.tolist()]
    n = len(ordered)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [0, i + 1, j + 1,   0, j + 1, i + 1]  # double-sided
    return verts, faces


def _helical_viz_data(atoms, parameter: str) -> dict:
    """Return 3Dmol drawing primitives for visualizing a helical parameter."""

    def _tors(a, b, c, d):
        b1, b2, b3 = b - a, c - b, d - c
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
        return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))

    chain_ids = sorted(set(a.chain_id for a in atoms))
    ch_a = chain_ids[0] if chain_ids else 'A'
    ch_b = chain_ids[1] if len(chain_ids) >= 2 else None

    def get_c1(ch):
        return sorted([(a.residue_seq, np.array([a.x, a.y, a.z]))
                       for a in atoms if a.chain_id == ch and a.name == "C1'"],
                      key=lambda x: x[0])

    c1_A = get_c1(ch_a)
    c1_B = get_c1(ch_b) if ch_b else []
    n_A = len(c1_A)
    if n_A < 2:
        return {"primitives": [], "description": "Not enough residues for visualization."}

    # ── Helix axis and base-pair midpoints ────────────────────────────────────
    n_bp = min(n_A, len(c1_B)) if c1_B else n_A
    if c1_B and n_bp >= 2:
        mids = np.array([(c1_A[i][1] + c1_B[n_bp - 1 - i][1]) / 2.0 for i in range(n_bp)])
    else:
        mids = np.array([c[1] for c in c1_A[:n_bp]])

    centroid = mids.mean(axis=0)

    # Axis: average base-pair plane normals (robust for short A-form helices;
    # SVD of midpoints fails when C1'–C1' midpoints are off-axis due to
    # X-displacement).
    if c1_B and n_bp >= 4:
        _bp_normals: list = []
        for _i in range(n_bp):
            _res_a = c1_A[_i][0]
            _res_b = c1_B[n_bp - 1 - _i][0]
            _pts = [np.array([a.x, a.y, a.z]) for a in atoms
                    if ((a.chain_id == ch_a and a.residue_seq == _res_a) or
                        (a.chain_id == ch_b and a.residue_seq == _res_b))
                    and a.name in _BASE_RING_ATOMS.get(a.residue_name, [])]
            if len(_pts) >= 3:
                _arr = np.array(_pts); _cen = _arr.mean(0)
                _, _, _Vt = np.linalg.svd(_arr - _cen)
                _bp_normals.append(_Vt[-1])
        if len(_bp_normals) >= 3:
            _ref = _bp_normals[0]
            _aligned = [n if np.dot(n, _ref) >= 0 else -n for n in _bp_normals]
            axis = np.mean(_aligned, axis=0); axis /= np.linalg.norm(axis)
        else:
            _, _, _Vt = np.linalg.svd(mids - centroid)
            axis = _Vt[0]
    else:
        _, _, _Vt = np.linalg.svd(mids - centroid)
        axis = _Vt[0]
    if np.dot(axis, mids[-1] - mids[0]) < 0:
        axis = -axis

    v_tmp = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    u1 = np.cross(axis, v_tmp); u1 /= np.linalg.norm(u1)
    u2 = np.cross(axis, u1);   u2 /= np.linalg.norm(u2)

    def p3(v): return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    prims = []

    def add_circle(ctr, r, color, lw=2, n=24):
        pts = [ctr + r * (np.cos(2*np.pi*k/n)*u1 + np.sin(2*np.pi*k/n)*u2) for k in range(n + 1)]
        for k in range(n):
            prims.append({"type": "line", "start": p3(pts[k]), "end": p3(pts[k+1]),
                          "color": color, "linewidth": lw})

    def add_arc(ctr, r, v0, v1, color, lw=3, n=24):
        cos_a = float(np.clip(np.dot(v0, v1), -1, 1))
        ang = float(np.arccos(cos_a))
        if np.dot(np.cross(v0, v1), axis) < 0:
            ang = -ang
        pts = []
        for k in range(n + 1):
            t = k / n
            v = (np.sin((1 - t)*ang)*v0 + np.sin(t*ang)*v1) / np.sin(ang) if abs(ang) > 1e-6 else v0.copy()
            pts.append(ctr + r * v)
        for k in range(n):
            prims.append({"type": "line", "start": p3(pts[k]), "end": p3(pts[k+1]),
                          "color": color, "linewidth": lw})

    # Axis extents
    projs = [float(np.dot(m - centroid, axis)) for m in mids]
    ax0 = centroid + (min(projs) - 2.0) * axis
    ax1 = centroid + (max(projs) + 2.0) * axis

    # Middle base-pair step gives most representative geometry
    i0 = max(0, n_bp // 2 - 1)
    i1 = min(i0 + 1, n_bp - 1)
    m0, m1 = mids[i0], mids[i1]

    # Precompute average twist over all steps (used by pitch and twist viz)
    all_perp = np.array([
        c - (centroid + float(np.dot(c - centroid, axis)) * axis)
        for c in [c[1] for c in c1_A]
    ])
    _all_t: list = []
    for _ii in range(n_A - 1):
        _va, _vb = all_perp[_ii], all_perp[_ii + 1]
        _na, _nb = float(np.linalg.norm(_va)), float(np.linalg.norm(_vb))
        if _na > 0.01 and _nb > 0.01:
            _ct = float(np.clip(np.dot(_va, _vb) / (_na * _nb), -1, 1))
            _at = float(np.degrees(np.arccos(_ct)))
            if np.dot(np.cross(_va, _vb), axis) < 0: _at = -_at
            _all_t.append(_at)
    avg_twist_all = float(np.mean(_all_t)) if _all_t else 36.0

    # ── Rise ─────────────────────────────────────────────────────────────────
    if parameter == "rise":
        # Average rise over all steps (matches what the classifier reports)
        all_projs = [float(np.dot(m - centroid, axis)) for m in mids]
        avg_rise = float(np.mean(np.abs(np.diff(all_projs))))

        # Try to get actual base-pair plane surfaces from ring atoms
        def _get_bp(ia, ib):
            res_a = c1_A[ia][0]
            res_b = c1_B[ib][0] if c1_B else None
            return _fit_bp_plane(atoms, ch_a, res_a, ch_b, res_b)

        n0, c_bp0, pts0 = _get_bp(i0, n_bp - 1 - i0)
        n1, c_bp1, pts1 = _get_bp(i1, n_bp - 1 - i1)

        # Helix axis
        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.1, "color": "#555555", "opacity": 0.5,
                      "fromCap": True, "toCap": True})

        # Draw plane 0 — prefer mesh surface; fall back to circle
        if n0 is not None and len(pts0) >= 3:
            if np.dot(n0, axis) < 0: n0 = -n0
            verts0, faces0 = _plane_mesh(n0, c_bp0, pts0)
            prims.append({"type": "custom", "vertices": verts0, "faces": faces0,
                          "color": "#4da6ff", "opacity": 0.35})
            prims.append({"type": "sphere", "center": p3(c_bp0), "radius": 0.3,
                          "color": "#4da6ff", "opacity": 0.9})
            foot0 = centroid + float(np.dot(c_bp0 - centroid, axis)) * axis
        else:
            add_circle(m0, 4.5, "#4da6ff", lw=2)
            foot0 = centroid + float(np.dot(m0 - centroid, axis)) * axis

        # Draw plane 1
        if n1 is not None and len(pts1) >= 3:
            if np.dot(n1, axis) < 0: n1 = -n1
            verts1, faces1 = _plane_mesh(n1, c_bp1, pts1)
            prims.append({"type": "custom", "vertices": verts1, "faces": faces1,
                          "color": "#ff7f7f", "opacity": 0.35})
            prims.append({"type": "sphere", "center": p3(c_bp1), "radius": 0.3,
                          "color": "#ff7f7f", "opacity": 0.9})
            foot1 = centroid + float(np.dot(c_bp1 - centroid, axis)) * axis
        else:
            add_circle(m1, 4.5, "#ff7f7f", lw=2)
            foot1 = centroid + float(np.dot(m1 - centroid, axis)) * axis

        # Arrow purely along the helix axis between the two feet —
        # its length = axial distance = what we label
        step_rise = float(np.linalg.norm(foot1 - foot0))
        prims.append({"type": "arrow", "start": p3(foot0), "end": p3(foot1),
                      "radius": 0.13, "color": "#ffd700", "opacity": 0.95})
        prims.append({"type": "label", "text": f"avg {avg_rise:.2f} Å",
                      "position": p3((foot0 + foot1) / 2 + 1.8 * u1)})
        desc = (f"Rise (avg {avg_rise:.2f} Å/bp, this step {step_rise:.2f} Å): "
                "axial distance between consecutive base-pair planes (surfaces). "
                "Arrow is purely along the helix axis.")

    # ── Twist / bp-per-turn / handedness ─────────────────────────────────────
    elif parameter in ("twist", "bpturn", "handedness"):
        c0 = c1_A[i0][1]; c1x = c1_A[i1][1]

        perp_A_all = all_perp  # precomputed above

        def foot_and_perp(pt):
            t = float(np.dot(pt - centroid, axis))
            foot = centroid + t * axis
            return foot, pt - foot

        f0, dv0 = foot_and_perp(c0)
        f1, dv1 = foot_and_perp(c1x)
        n0, n1 = np.linalg.norm(dv0), np.linalg.norm(dv1)
        if n0 < 0.1 or n1 < 0.1:
            return {"primitives": [], "description": "C1’ too close to axis."}

        dv0u = dv0 / n0; dv1u = dv1 / n1

        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.1, "color": "#555555", "opacity": 0.5,
                      "fromCap": True, "toCap": True})
        add_circle(mids[i0], 4.5, "#4da6ff", lw=1)
        add_circle(mids[i1], 4.5, "#ff7f7f", lw=1)
        prims += [
            {"type": "sphere", "center": p3(c0),  "radius": 0.45, "color": "#4da6ff", "opacity": 0.95},
            {"type": "sphere", "center": p3(c1x), "radius": 0.45, "color": "#ff7f7f", "opacity": 0.95},
            {"type": "sphere", "center": p3(f0),  "radius": 0.18, "color": "#aaaaaa", "opacity": 0.7},
            {"type": "sphere", "center": p3(f1),  "radius": 0.18, "color": "#aaaaaa", "opacity": 0.7},
            {"type": "cylinder", "start": p3(f0), "end": p3(c0),  "radius": 0.08,
             "color": "#4da6ff", "opacity": 0.8, "fromCap": True, "toCap": True},
            {"type": "cylinder", "start": p3(f1), "end": p3(c1x), "radius": 0.08,
             "color": "#ff7f7f", "opacity": 0.8, "fromCap": True, "toCap": True},
        ]
        arc_ctr = (f0 + f1) / 2
        arc_r = (n0 + n1) / 2 * 0.65
        add_arc(arc_ctr, arc_r, dv0u, dv1u, "#ffd700")

        cos_t = float(np.clip(np.dot(dv0u, dv1u), -1, 1))
        t_ang = float(np.degrees(np.arccos(cos_t)))
        if np.dot(np.cross(dv0u, dv1u), axis) < 0:
            t_ang = -t_ang

        mid_dir = dv0u + dv1u
        nd = np.linalg.norm(mid_dir)
        mid_dir = mid_dir / nd if nd > 0.01 else u1
        # Also compute average twist over all steps (matches classifier sidebar)
        all_twists = []
        for ii in range(n_A - 1):
            va, vb = perp_A_all[ii], perp_A_all[ii + 1]
            na2, nb2 = np.linalg.norm(va), np.linalg.norm(vb)
            if na2 > 0.01 and nb2 > 0.01:
                ct = float(np.clip(np.dot(va, vb) / (na2 * nb2), -1, 1))
                at = float(np.degrees(np.arccos(ct)))
                if np.dot(np.cross(va, vb), axis) < 0: at = -at
                all_twists.append(at)
        avg_t = float(np.mean(all_twists)) if all_twists else t_ang
        prims.append({"type": "label", "text": f"step {t_ang:+.1f}° / avg {avg_t:+.1f}°",
                      "position": p3(arc_ctr + arc_r * 1.5 * mid_dir)})
        hand = "right-handed" if avg_t > 0 else "left-handed"
        desc = (f"Twist (avg {avg_t:+.1f}°/bp, this step {t_ang:+.1f}°): rotation between "
                f"consecutive C1’ radii (blue/red) around the helix axis (grey). {hand} helix.")

    # ── Pitch ─────────────────────────────────────────────────────────────────
    elif parameter == "pitch":
        # Use the average twist (same source as sidebar) to compute bp_per_turn.
        # The middle-step method fails for Z-DNA: a ~−6° CpG step gives 60 bp/turn,
        # so n_show gets capped to n_bp and the "pitch" shows the structure height
        # instead of one full turn.
        bp_per_turn = abs(360.0 / avg_twist_all) if abs(avg_twist_all) > 0.5 else 10.0
        n_show = min(n_bp, max(3, round(bp_per_turn)))

        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.1, "color": "#555555", "opacity": 0.5,
                      "fromCap": True, "toCap": True})
        add_circle(mids[0], 4.5, "#4da6ff", lw=2)
        add_circle(mids[n_show - 1], 4.5, "#ff7f7f", lw=2)
        for i in range(1, n_show - 1):
            add_circle(mids[i], 4.5, "#666666", lw=1)
        prims += [
            {"type": "sphere", "center": p3(mids[0]),          "radius": 0.35, "color": "#4da6ff", "opacity": 0.9},
            {"type": "sphere", "center": p3(mids[n_show - 1]), "radius": 0.35, "color": "#ff7f7f", "opacity": 0.9},
            {"type": "arrow", "start": p3(mids[0]), "end": p3(mids[n_show - 1]),
             "radius": 0.13, "color": "#ffd700", "opacity": 0.95},
        ]
        all_projs_p = [float(np.dot(m - centroid, axis)) for m in mids]
        avg_rise_p = float(abs(np.mean(np.diff(all_projs_p))))
        pitch_computed = avg_rise_p * bp_per_turn
        pitch_shown = abs(float(np.dot(mids[n_show - 1] - mids[0], axis)))
        partial = n_show >= n_bp  # structure doesn't cover a full turn
        label_txt = f"~{pitch_computed:.1f} Å" + (" (extrap.)" if partial else "")
        prims.append({"type": "label", "text": label_txt,
                      "position": p3((mids[0] + mids[n_show - 1]) / 2 + 1.8 * u1)})
        desc = (f"Pitch (~{pitch_computed:.1f} Å): height of one full helical turn "
                f"= rise ({avg_rise_p:.2f} Å) × bp/turn ({bp_per_turn:.1f})"
                + (f". Arrow shows {n_show} of {round(bp_per_turn):.0f} bp/turn "
                   f"(structure shorter than one turn)." if partial else "."))

    # ── ν₂ sugar pucker ───────────────────────────────────────────────────────
    elif parameter == "nu2":
        by_res: dict = {}
        for a in atoms:
            if a.chain_id == ch_a:
                by_res.setdefault(a.residue_seq, {})[a.name] = np.array([a.x, a.y, a.z])
        res_keys = sorted(by_res.keys())
        ri = res_keys[len(res_keys) // 2]
        res = by_res[ri]
        needed = ["C2'", "C3'", "C4'", "O4'"]
        if not all(k in res for k in needed):
            return {"primitives": [], "description": "Required sugar atoms not found."}
        colors = {"C2'": "#88ff88", "C3'": "#ffdd44", "C4'": "#ff8844", "O4'": "#ff5555"}
        for nm in needed:
            prims.append({"type": "sphere", "center": p3(res[nm]),
                          "radius": 0.5, "color": colors[nm], "opacity": 0.92})
        for i in range(len(needed) - 1):
            prims.append({"type": "cylinder",
                          "start": p3(res[needed[i]]), "end": p3(res[needed[i + 1]]),
                          "radius": 0.12, "color": "#ffd700", "opacity": 0.85,
                          "fromCap": True, "toCap": True})
        prims.append({"type": "cylinder", "start": p3(res["C2'"]), "end": p3(res["O4'"]),
                      "radius": 0.05, "color": "#888888", "opacity": 0.4,
                      "fromCap": True, "toCap": True})
        nu2 = _tors(res["C2'"], res["C3'"], res["C4'"], res["O4'"])
        prims.append({"type": "label", "text": f"ν₂ = {nu2:+.1f}°",
                      "position": p3((res["C2'"] + res["O4'"]) / 2 + u1 * 2.0)})
        puck = "C3’-endo (A-form)" if nu2 > 0 else "C2’-endo (B-form)"
        desc = (f"ν₂ = {nu2:+.1f}°: torsion C2’–C3’–C4’–O4’. {puck}.")

    # ── Phosphate radius ──────────────────────────────────────────────────────
    elif parameter == "prad":
        p_list = sorted([(a.residue_seq, np.array([a.x, a.y, a.z]))
                         for a in atoms if a.chain_id == ch_a and a.name == 'P'],
                        key=lambda x: x[0])
        if not p_list:
            return {"primitives": [], "description": "No phosphorus atoms found."}
        _, p_coord = p_list[len(p_list) // 2]
        t = float(np.dot(p_coord - centroid, axis))
        foot = centroid + t * axis
        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.1, "color": "#555555", "opacity": 0.5,
                      "fromCap": True, "toCap": True})
        r_val = float(np.linalg.norm(p_coord - foot))
        prims += [
            {"type": "sphere", "center": p3(p_coord), "radius": 0.55, "color": "#ff9900", "opacity": 0.95},
            {"type": "sphere", "center": p3(foot),    "radius": 0.20, "color": "#aaaaaa", "opacity": 0.8},
            {"type": "cylinder", "start": p3(foot), "end": p3(p_coord),
             "radius": 0.10, "color": "#ffd700", "opacity": 0.9, "fromCap": True, "toCap": True},
        ]
        prims.append({"type": "label", "text": f"r = {r_val:.1f} Å",
                      "position": p3((foot + p_coord) / 2 + u2 * 1.2)})
        desc = (f"P-axis radius ({r_val:.1f} Å): perpendicular distance from "
                "P atom (orange) to the helix axis (grey).")

    # ── Inclination: angle between base-pair plane and helix axis ────────────
    elif parameter == "inclination":
        if not c1_B:
            return {"primitives": [], "description": "Inclination requires both strands."}

        # Use the middle base pair
        res_a = c1_A[i0][0]
        res_b = c1_B[n_bp - 1 - i0][0]
        bp_n, bp_cen, bp_pts = _fit_bp_plane(atoms, ch_a, res_a, ch_b, res_b)
        if bp_n is None:
            return {"primitives": [], "description": "Base ring atoms not found."}
        if np.dot(bp_n, axis) < 0:
            bp_n = -bp_n

        cos_incl = float(np.clip(np.dot(bp_n, axis), -1.0, 1.0))
        incl_deg = float(np.degrees(np.arccos(cos_incl)))

        # Helix axis
        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.1, "color": "#555555", "opacity": 0.5,
                      "fromCap": True, "toCap": True})

        # Base-pair plane surface
        v_mesh, f_mesh = _plane_mesh(bp_n, bp_cen, bp_pts)
        prims.append({"type": "custom", "vertices": v_mesh, "faces": f_mesh,
                      "color": "#4da6ff", "opacity": 0.4})

        # Normal arrow (blue — points perpendicular to the base pair)
        n_end = bp_cen + 4.5 * bp_n
        prims.append({"type": "arrow", "start": p3(bp_cen), "end": p3(n_end),
                      "radius": 0.12, "color": "#4da6ff", "opacity": 0.9})

        # Helix-axis arrow at bp centroid (red)
        t_bp = float(np.dot(bp_cen - centroid, axis))
        ax_at_bp = centroid + t_bp * axis
        ax_end = ax_at_bp + 4.5 * axis
        prims.append({"type": "arrow", "start": p3(ax_at_bp), "end": p3(ax_end),
                      "radius": 0.12, "color": "#ff7f7f", "opacity": 0.9})

        # Arc between bp_n and axis
        add_arc(bp_cen, 3.0, bp_n, axis, "#ffd700")

        # Label near midpoint of arc
        mid_vec = bp_n + axis
        mid_len = np.linalg.norm(mid_vec)
        mid_dir = mid_vec / mid_len if mid_len > 0.01 else u1
        prims.append({"type": "label", "text": f"{incl_deg:.1f}°",
                      "position": p3(bp_cen + 3.5 * mid_dir + 0.5 * u2)})

        form_note = "B-DNA ~6°, A-DNA ~20°, Z-DNA ~9°"
        desc = (f"Inclination ({incl_deg:.1f}°): angle between base-pair plane normal "
                f"(blue arrow) and helix axis (red arrow). {form_note}.")

    # ── Propeller twist: dihedral between individual base plane normals ──────────
    elif parameter == "propeller":
        if not c1_B:
            return {"primitives": [], "description": "Propeller twist requires both strands."}

        res_a = c1_A[i0][0]
        res_b = c1_B[n_bp - 1 - i0][0]

        def _fit_base(ch, res):
            pts = [np.array([a.x, a.y, a.z]) for a in atoms
                   if a.chain_id == ch and a.residue_seq == res
                   and a.name in _BASE_RING_ATOMS.get(a.residue_name, [])]
            if len(pts) < 3: return None, None, np.empty((0, 3))
            arr = np.array(pts); cen = arr.mean(0)
            _, _, Vt = np.linalg.svd(arr - cen)
            n = Vt[-1]; n /= np.linalg.norm(n)
            return n, cen, arr

        n_a, cen_a, pts_a = _fit_base(ch_a, res_a)
        n_b, cen_b, pts_b = _fit_base(ch_b, res_b)
        if n_a is None or n_b is None:
            return {"primitives": [], "description": "Base ring atoms not found."}

        # Orient both normals consistently (toward helix axis direction)
        if np.dot(n_a, axis) < 0: n_a = -n_a
        if np.dot(n_b, axis) < 0: n_b = -n_b

        # C1'–C1' axis
        c1a = c1_A[i0][1]; c1b = c1_B[n_bp - 1 - i0][1]
        c1_vec = c1b - c1a
        c1_len = float(np.linalg.norm(c1_vec))
        c1_hat = c1_vec / c1_len if c1_len > 0.1 else axis.copy()

        # Propeller angle
        n_a_p = n_a - np.dot(n_a, c1_hat) * c1_hat
        n_b_p = n_b - np.dot(n_b, c1_hat) * c1_hat
        la, lb = float(np.linalg.norm(n_a_p)), float(np.linalg.norm(n_b_p))
        if la < 0.01 or lb < 0.01:
            return {"primitives": [], "description": "Base normals parallel to C1'–C1'; cannot compute propeller."}
        n_a_p /= la; n_b_p /= lb
        cos_p = float(np.clip(np.dot(n_a_p, n_b_p), -1.0, 1.0))
        prop_ang = float(np.degrees(np.arccos(cos_p)))
        if np.dot(np.cross(n_a_p, n_b_p), c1_hat) < 0: prop_ang = -prop_ang

        # Helix axis (background context)
        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.08, "color": "#555555", "opacity": 0.35,
                      "fromCap": True, "toCap": True})

        # Base plane surfaces
        if len(pts_a) >= 3:
            va, fa = _plane_mesh(n_a, cen_a, pts_a, pad=1.5)
            prims.append({"type": "custom", "vertices": va, "faces": fa,
                          "color": "#4da6ff", "opacity": 0.45})
        if len(pts_b) >= 3:
            vb, fb = _plane_mesh(n_b, cen_b, pts_b, pad=1.5)
            prims.append({"type": "custom", "vertices": vb, "faces": fb,
                          "color": "#ff9966", "opacity": 0.45})

        # C1'–C1' axis arrow
        prims.append({"type": "arrow", "start": p3(c1a), "end": p3(c1b),
                      "radius": 0.10, "color": "#aaaaaa", "opacity": 0.7})

        # Normal arrows for each base
        n_end_a = cen_a + 3.5 * n_a
        n_end_b = cen_b + 3.5 * n_b
        prims.append({"type": "arrow", "start": p3(cen_a), "end": p3(n_end_a),
                      "radius": 0.10, "color": "#4da6ff", "opacity": 0.85})
        prims.append({"type": "arrow", "start": p3(cen_b), "end": p3(n_end_b),
                      "radius": 0.10, "color": "#ff9966", "opacity": 0.85})

        # Arc between the two projected normals (in plane perp to C1'–C1')
        arc_ctr_p = (cen_a + cen_b) / 2
        add_arc(arc_ctr_p, 2.5, n_a_p, n_b_p, "#ffd700")

        # Label
        mid_vec = n_a_p + n_b_p; nd = np.linalg.norm(mid_vec)
        mid_dir = (mid_vec / nd) if nd > 0.01 else u1
        prims.append({"type": "label", "text": f"{prop_ang:+.1f}°",
                      "position": p3(arc_ctr_p + 3.2 * mid_dir)})

        desc = (f"Propeller twist ({prop_ang:+.1f}°): dihedral between base A "
                f"(blue) and base B (orange) plane normals around the C1'–C1' "
                f"axis (grey). B-DNA ≈ −11°.")

    # ── Helix axis only (standalone, for persistent overlay) ────────────────────
    elif parameter == "axis":
        prims.append({"type": "cylinder", "start": p3(ax0), "end": p3(ax1),
                      "radius": 0.15, "color": "#888888", "opacity": 0.7,
                      "fromCap": True, "toCap": True})
        # Tick marks at each base-pair level
        for m in mids:
            t_val = float(np.dot(m - centroid, axis))
            tick_cen = centroid + t_val * axis
            r_tick = 0.6
            prims.append({"type": "sphere", "center": p3(tick_cen),
                          "radius": 0.18, "color": "#aaaaaa", "opacity": 0.6})
        desc = f"Helix axis ({n_bp} base pairs shown as grey dots)."

    else:
        return {"primitives": [], "description": f"Unknown parameter: {parameter}"}

    return {"primitives": prims, "description": desc}


@app.route("/api/helical_viz", methods=["POST"])
def api_helical_viz():
    """Return 3Dmol drawing primitives for visualizing a helical parameter."""
    data = request.get_json()
    pdb_content = data.get("pdb", "")
    parameter = data.get("parameter", "rise")
    if not pdb_content:
        return jsonify({"error": "No structure provided"}), 400

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(pdb_content)
        tmp_path = fh.name
    try:
        from dna_builder.io_parser import parse_pdb
        atoms = parse_pdb(tmp_path)
        result = _helical_viz_data(atoms, parameter)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonify_dict(obj):
    """Recursively convert numpy scalars to Python types for JSON."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonify_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_dict(x) for x in obj]
    return obj


def _compute_fingerprint(atoms, sequence: str, form: str) -> dict:
    """Compute structural fingerprint parameters from the atom list."""
    residues: dict = defaultdict(dict)
    for a in atoms:
        residues[(a.chain_id, a.residue_seq)][a.name] = np.array([a.x, a.y, a.z])

    a_keys = sorted([k for k in residues if k[0] == "A"], key=lambda x: x[1])
    b_keys = sorted([k for k in residues if k[0] == "B"], key=lambda x: x[1])
    n = len(a_keys)

    # P–P intra-strand consecutive distances (strand A)
    pp = []
    for i in range(1, n):
        ra, rb = residues[a_keys[i]], residues[a_keys[i - 1]]
        if "P" in ra and "P" in rb:
            pp.append(round(float(np.linalg.norm(ra["P"] - rb["P"])), 3))

    # C1'–C1' cross-strand (antiparallel pairing)
    c1c1 = []
    for i, ak in enumerate(a_keys):
        j = n - 1 - i
        if j < len(b_keys):
            bk = b_keys[j]
            if "C1'" in residues[ak] and "C1'" in residues[bk]:
                c1c1.append(round(float(np.linalg.norm(residues[ak]["C1'"] - residues[bk]["C1'"])), 3))

    # Glycosidic N cross-strand
    gly = []
    for i, ak in enumerate(a_keys):
        j = n - 1 - i
        if j < len(b_keys):
            bk = b_keys[j]
            ra_res, rb_res = residues[ak], residues[bk]
            na = ra_res.get("N9") if "N9" in ra_res else ra_res.get("N1")
            nb = rb_res.get("N1") if "N1" in rb_res else rb_res.get("N9")
            if na is not None and nb is not None:
                gly.append(round(float(np.linalg.norm(na - nb)), 3))

    def _stats(vals):
        if not vals:
            return {"values": [], "mean": 0.0, "std": 0.0}
        return {
            "values": vals,
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
        }

    return {
        "form": form,
        "sequence": sequence,
        "n_bp": n,
        "pp_distances": _stats(pp),
        "c1c1_distances": _stats(c1c1),
        "glycosidic_distances": _stats(gly),
    }


if __name__ == "__main__":
    import signal
    import subprocess

    port = 5052

    # Kill any process already listening on this port
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                       stderr=subprocess.DEVNULL).decode().split()
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            import time; time.sleep(0.4)
            print(f"Stopped previous server (PID {', '.join(pids)})")
    except subprocess.CalledProcessError:
        pass  # nothing was listening

    print(f"\nDNA Builder GUI  →  http://localhost:{port}\n")
    app.run(debug=True, port=port, use_reloader=False)
