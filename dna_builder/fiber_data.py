"""
DNA structure data: standard base geometries, backbone templates, and helical parameters.

Uses the base pair reference frame approach (Olson et al., 2001; Lu & Olson, 2003)
with standard base geometries (Clowney et al., 1996; Parkinson et al., 1996).

The construction algorithm:
1. Build a Watson-Crick base pair in the base pair reference frame
2. Position the base pair relative to the helix axis (x-displacement, inclination)
3. Attach sugar-phosphate backbone with correct sugar pucker
4. Apply helical screw (rise + twist) to generate successive base pairs
5. Both strands are built simultaneously from the base pair

The pseudo-dyad operation for Watson-Crick base pairs:
   (x, y, z) -> (-x, -y, -z)
This maps strand I to strand II within a base pair, giving C1'-C1' = 10.4 Å.

References:
- Olson WK et al. (2001) J Mol Biol 313:229 (base pair parameters)
- Lu XJ, Olson WK (2003) Nucleic Acids Res 31:5108 (3DNA)
- Clowney L et al. (1996) JACS 118:509 (standard base geometries)
- Saenger W (1984) Principles of Nucleic Acid Structure
"""

import numpy as np

# =============================================================================
# Helical parameters for each DNA form
# =============================================================================

HELICAL_PARAMS = {
    "A": {
        "rise": 2.548,        # Å per base pair
        "twist": 32.727,      # degrees per base pair
        "x_disp": -4.41,      # Å, displacement of bp from helix axis
        "y_disp": 0.0,
        "inclination": 19.1,  # degrees, tilt of bp plane relative to helix axis
        "tip": 0.0,
        "propeller": -11.8,   # degrees
        "sugar_pucker": "C3_endo",
        "glycosidic": "anti",
    },
    "B": {
        "rise": 3.375,
        "twist": 36.0,
        "x_disp": -0.71,
        "y_disp": 0.0,
        "inclination": -5.9,
        "tip": 0.0,
        "propeller": -11.4,
        "sugar_pucker": "C2_endo",
        "glycosidic": "anti",
    },
    "Z": {
        # Z-DNA uses dinucleotide repeat
        "rise_1": 3.530,      # pyr->pur step
        "rise_2": 3.892,      # pur->pyr step
        "twist_1": -9.0,      # pyr->pur step
        "twist_2": -51.0,     # pur->pyr step
        "x_disp": 3.0,
        "y_disp": 0.0,
        "inclination": -6.2,
        "tip": 0.0,
        "propeller": -1.3,
        "sugar_pucker_pur": "C3_endo",  # purines
        "sugar_pucker_pyr": "C2_endo",  # pyrimidines
        "glycosidic_pur": "syn",        # purines in syn
        "glycosidic_pyr": "anti",       # pyrimidines in anti
    },
}

# =============================================================================
# Standard base atom coordinates in the base pair reference frame
# =============================================================================
# The base pair reference frame has:
# - Origin at the center of the base pair (midpoint of C1'-C1')
# - y-axis along the long axis (from strand II C1' to strand I C1')
# - x-axis toward the major groove
# - z-axis perpendicular to base pair plane (along helix axis)
#
# The glycosidic nitrogen (N9 for purines, N1 for pyrimidines) connects
# to C1' of the sugar at y ≈ 5.2 Å from the bp center.
#
# Coordinates from Clowney et al. (1996) and Parkinson et al. (1996),
# positioned in the base pair reference frame.

# Adenine (purine) - N9 connects to sugar
BASE_A = [
    ("N9",  "N",  -1.291,  4.498,  0.000),
    ("C8",  "C",   0.024,  4.897,  0.000),
    ("N7",  "N",   0.877,  3.902,  0.000),
    ("C5",  "C",   0.071,  2.771,  0.000),
    ("C6",  "C",   0.369,  1.398,  0.000),
    ("N6",  "N",   1.611,  0.909,  0.000),
    ("N1",  "N",  -0.668,  0.532,  0.000),
    ("C2",  "C",  -1.912,  1.023,  0.000),
    ("N3",  "N",  -2.320,  2.290,  0.000),
    ("C4",  "C",  -1.267,  3.124,  0.000),
]

# Thymine (pyrimidine) - N1 connects to sugar
BASE_T = [
    ("N1",  "N",  -1.284,  4.500,  0.000),
    ("C2",  "C",  -1.462,  3.135,  0.000),
    ("O2",  "O",  -2.562,  2.608,  0.000),
    ("N3",  "N",  -0.298,  2.407,  0.000),
    ("C4",  "C",   0.994,  2.897,  0.000),
    ("O4",  "O",   1.944,  2.119,  0.000),
    ("C5",  "C",   1.106,  4.338,  0.000),
    ("C5M", "C",   2.466,  4.961,  0.000),
    ("C6",  "C",  -0.024,  5.057,  0.000),
]

# Guanine (purine) - N9 connects to sugar
BASE_G = [
    ("N9",  "N",  -1.291,  4.498,  0.000),
    ("C8",  "C",   0.024,  4.897,  0.000),
    ("N7",  "N",   0.877,  3.902,  0.000),
    ("C5",  "C",   0.071,  2.771,  0.000),
    ("C6",  "C",   0.424,  1.380,  0.000),
    ("O6",  "O",   1.554,  0.856,  0.000),
    ("N1",  "N",  -0.700,  0.583,  0.000),
    ("N2",  "N",  -2.949,  0.139,  0.000),
    ("C2",  "C",  -1.999,  1.087,  0.000),
    ("N3",  "N",  -2.342,  2.364,  0.000),
    ("C4",  "C",  -1.267,  3.124,  0.000),
]

# Cytosine (pyrimidine) - N1 connects to sugar
BASE_C = [
    ("N1",  "N",  -1.284,  4.500,  0.000),
    ("C2",  "C",  -1.462,  3.135,  0.000),
    ("O2",  "O",  -2.562,  2.608,  0.000),
    ("N3",  "N",  -0.298,  2.407,  0.000),
    ("C4",  "C",   0.994,  2.897,  0.000),
    ("N4",  "N",   2.093,  2.120,  0.000),
    ("C5",  "C",   1.106,  4.338,  0.000),
    ("C6",  "C",  -0.024,  5.057,  0.000),
]

# =============================================================================
# Sugar-phosphate backbone templates
# =============================================================================
# Backbone atom coordinates in the base pair reference frame.
# C1' is positioned at (0.012, 5.200, 0.000) to give:
#   - N9/N1 to C1' bond length = 1.48 Å
#   - C1'-C1' distance across base pair = 10.4 Å (via pseudo-dyad inversion)
#
# The backbone runs: ...—P—O5'—C5'—C4'—C3'—O3'—P—...
#                              |         |
#                             C4'       C2'
#                              |         |
#                             O4'———C1'
#                                    |
#                                   N9/N1 (base)

# C2'-endo backbone (B-DNA, anti glycosidic angle chi ~ -117°)
BACKBONE_C2ENDO_ANTI = [
    ("C1'", "C",   0.012,  5.200,  0.000),
    ("C2'", "C",   0.902,  6.046,  0.924),
    ("C3'", "C",   0.415,  7.482,  0.714),
    ("O3'", "O",   1.351,  8.432,  1.200),
    ("C4'", "C",   0.200,  7.550, -0.800),
    ("O4'", "O",  -0.336,  6.244, -1.100),
    ("C5'", "C",  -0.791,  8.624, -1.180),
    ("O5'", "O",  -0.191,  9.854, -0.720),
    ("P",   "P",  -1.001, 11.194, -0.480),
    ("OP1", "O",  -0.131, 12.214,  0.180),
    ("OP2", "O",  -1.891, 10.714,  0.620),
]

# C3'-endo backbone (A-DNA, anti glycosidic angle chi ~ -154°)
BACKBONE_C3ENDO_ANTI = [
    ("C1'", "C",   0.012,  5.200,  0.000),
    ("C2'", "C",   1.119,  6.194,  0.200),
    ("C3'", "C",   0.539,  7.414, -0.530),
    ("O3'", "O",   1.509,  8.434, -0.600),
    ("C4'", "C",   0.139,  6.914, -1.930),
    ("O4'", "O",  -0.461,  5.614, -1.300),
    ("C5'", "C",  -0.911,  7.814, -2.550),
    ("O5'", "O",  -0.311,  9.094, -2.700),
    ("P",   "P",  -1.111, 10.114, -3.560),
    ("OP1", "O",  -0.211, 11.274, -3.560),
    ("OP2", "O",  -1.611,  9.514, -4.830),
]

# C3'-endo backbone for syn glycosidic angle (Z-DNA purines, chi ~ 60°)
# Offset: C1' moved from (-0.160, 5.890) to (0.012, 5.200)
BACKBONE_C3ENDO_SYN = [
    ("C1'", "C",   0.012,  5.200,  0.000),
    ("C2'", "C",  -0.928,  6.110,  0.800),
    ("C3'", "C",  -0.328,  7.490,  0.500),
    ("O3'", "O",  -1.228,  8.510,  0.300),
    ("C4'", "C",   0.272,  7.210, -0.880),
    ("O4'", "O",   0.872,  5.890, -0.700),
    ("C5'", "C",  -0.728,  7.310, -2.000),
    ("O5'", "O",  -0.128,  8.110, -3.050),
    ("P",   "P",  -0.928,  8.510, -4.350),
    ("OP1", "O",  -0.028,  9.410, -5.100),
    ("OP2", "O",  -2.028,  9.210, -3.700),
]

# =============================================================================
# Watson-Crick base pair geometry
# =============================================================================
# The WC partner base is related to the reference base by the pseudo-dyad:
#   (x, y, z) -> (-x, -y, -z)
# This gives C1'-C1' distance = 2 * sqrt(0.012² + 5.200²) = 10.400 Å

# Lookup tables
BASE_COORDS = {"A": BASE_A, "T": BASE_T, "G": BASE_G, "C": BASE_C}
WC_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
RESIDUE_NAMES = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
PURINE_BASES = {"A", "G"}
PYRIMIDINE_BASES = {"T", "C"}
