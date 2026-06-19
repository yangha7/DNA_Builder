"""
Command-line interface for DNA Builder.

Usage:
    python -m dna_builder SEQUENCE [--form {A,B,Z}] [--output FILE] [--format {pdb,xyz,cml}]
    python -m dna_builder --classify INPUT_FILE [--sequence SEQ]

Examples:
    python -m dna_builder ATCGATCG
    python -m dna_builder ATCGATCG --form A --output a_dna.pdb
    python -m dna_builder GCGCGCGC --form Z --output z_dna.pdb
    python -m dna_builder ATCG --format xyz --output dna.xyz
    python -m dna_builder --classify fold_atatat_model_0.cif
    python -m dna_builder --classify some_structure.pdb
    python -m dna_builder --classify structure.xyz --sequence ATATAT
"""

import argparse
import sys
from .builder import build_dna
from .io_pdb import write_pdb, write_xyz, write_cml


def main():
    parser = argparse.ArgumentParser(
        prog="dna_builder",
        description="Build accurate A, B, and Z-form DNA structures from sequence, "
                    "or classify existing structures.",
        epilog=(
            "Examples:\n"
            "  python -m dna_builder ATCGATCG\n"
            "  python -m dna_builder ATCGATCG --form A --output a_dna.pdb\n"
            "  python -m dna_builder GCGCGCGC --form Z --output z_dna.pdb\n"
            "  python -m dna_builder ATCG --format xyz --output dna.xyz\n"
            "  python -m dna_builder --classify fold_atatat_model_0.cif\n"
            "  python -m dna_builder --classify structure.xyz --sequence ATATAT\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "sequence",
        type=str,
        nargs='?',
        default=None,
        help="DNA sequence for strand I (5' to 3'). Only A, T, G, C allowed.",
    )
    parser.add_argument(
        "--form", "-f",
        type=str,
        choices=["A", "B", "Z", "a", "b", "z"],
        default="B",
        help="DNA form: A, B (default), or Z.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename. If not specified, writes to stdout.",
    )
    parser.add_argument(
        "--format", "-F",
        type=str,
        choices=["pdb", "xyz", "cml"],
        default="pdb",
        help="Output format: pdb (default), xyz, or cml.",
    )
    parser.add_argument(
        "--classify", "-c",
        type=str,
        default=None,
        metavar="INPUT_FILE",
        help="Classify a DNA structure file (PDB, CIF, or XYZ) as A/B/Z-DNA.",
    )
    parser.add_argument(
        "--sequence-hint", "-s",
        type=str,
        default=None,
        metavar="SEQ",
        dest="sequence_hint",
        help="DNA sequence hint for classification (required for XYZ files "
             "without recognizable sequence in filename).",
    )

    args = parser.parse_args()

    # --- Classify mode ---
    if args.classify:
        _run_classify(args.classify, args.sequence_hint)
        return

    # --- Build mode ---
    if args.sequence is None:
        parser.error("the following arguments are required: sequence "
                     "(or use --classify INPUT_FILE)")

    sequence = args.sequence.upper().strip()
    form = args.form.upper()
    fmt = args.format.lower()

    # Validate sequence
    invalid = set(sequence) - set("ATGC")
    if invalid:
        print(f"Error: Invalid bases in sequence: {invalid}", file=sys.stderr)
        sys.exit(1)

    if len(sequence) < 2:
        print("Error: Sequence must be at least 2 bases long.", file=sys.stderr)
        sys.exit(1)

    # Build DNA
    print(f"Building {form}-form DNA: 5'-{sequence}-3'", file=sys.stderr)
    print(f"Complement:          3'-{''.join({'A':'T','T':'A','G':'C','C':'G'}[b] for b in sequence)}-5'",
          file=sys.stderr)
    print(f"Length: {len(sequence)} base pairs", file=sys.stderr)

    try:
        atoms = build_dna(sequence, form)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated {len(atoms)} atoms", file=sys.stderr)

    # Determine output
    title = f"{form}-DNA 5'-{sequence}-3'"

    if args.output:
        if fmt == "pdb":
            write_pdb(atoms, args.output, title=title)
        elif fmt == "xyz":
            write_xyz(atoms, args.output, comment=title)
        elif fmt == "cml":
            write_cml(atoms, args.output, title=title)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        if fmt == "pdb":
            write_pdb(atoms, sys.stdout, title=title)
        elif fmt == "xyz":
            write_xyz(atoms, sys.stdout, comment=title)
        elif fmt == "cml":
            write_cml(atoms, sys.stdout, title=title)


def _run_classify(filepath: str, sequence: str = None) -> None:
    """Run the DNA conformation classifier."""
    import os

    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        from .classifier import classify_structure
        result = classify_structure(filepath, verbose=True, sequence=sequence)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
