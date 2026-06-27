"""Flask backend for DNA Builder GUI."""

import io
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_builder.builder import build_dna
from dna_builder.classifier import classify_structure
from dna_builder.io_pdb import write_pdb
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
        if not is_canonical:
            return jsonify({"error": f"Z-DNA sequence '{sequence}' does not alternate purine-pyrimidine. Use a canonical sequence such as GCGCGCGC."}), 400

    try:
        atoms = build_dna_v2(sequence, form) if method == "zmatrix" else build_dna(sequence, form, method="template")
        if hydrogens:
            from dna_builder.hydrogens import add_hydrogens
            atoms = add_hydrogens(atoms)

        buf = io.StringIO()
        write_pdb(atoms, buf, f"{form}-DNA 5'-{sequence}-3'")
        pdb_str = buf.getvalue()

        fingerprint = _compute_fingerprint(atoms, sequence, form)
        return jsonify({"pdb": pdb_str, "fingerprint": fingerprint, "ok": True})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
