"""Default element colours and covalent radii (Cordero 2008).

Colours are a muted, publication-friendly palette.  Conventional
associations are preserved (red for oxygen, blue for nitrogen, etc.)
but with desaturated tones that reproduce well in print.
"""

from __future__ import annotations

from hofmann.model import AtomStyle, BondSpec, Colour

# Muted element colours as normalised RGB tuples.
# Common elements use hand-picked colours; less common elements
# use desaturated tones grouped by periodic table region.
ELEMENT_COLOURS: dict[str, tuple[float, float, float]] = {
    # Period 1
    "H":  (0.965, 0.914, 0.808),  # off-white cream
    "He": (0.710, 0.824, 0.871),  # pale steel blue
    # Period 2
    "Li": (0.992, 0.294, 0.098),  # vermilion
    "Be": (0.886, 0.796, 0.478),  # muted gold
    "B":  (0.957, 0.718, 0.506),  # tan
    "C":  (0.388, 0.373, 0.357),  # warm dark grey
    "N":  (0.059, 0.200, 0.420),  # dark blue
    "O":  (0.631, 0.000, 0.000),  # deep red
    "F":  (0.384, 0.753, 0.631),  # seafoam green
    "Ne": (0.690, 0.824, 0.831),  # pale blue-grey
    # Period 3
    "Na": (0.373, 0.616, 0.682),  # soft blue
    "Mg": (0.549, 0.529, 0.314),  # olive-grey
    "Al": (0.675, 0.675, 0.675),  # silver grey
    "Si": (0.855, 0.482, 0.133),  # orange-brown
    "P":  (1.000, 0.471, 0.196),  # orange
    "S":  (0.980, 0.800, 0.216),  # yellow
    "Cl": (0.263, 0.584, 0.525),  # teal-green
    "Ar": (0.510, 0.675, 0.635),  # grey-teal
    # Period 4
    "K":  (0.851, 0.612, 0.137),  # dark gold
    "Ca": (0.988, 0.694, 0.192),  # amber
    "Sc": (0.498, 0.694, 0.667),  # sage
    "Ti": (0.251, 0.690, 0.749),  # bright teal
    "V":  (0.165, 0.545, 0.596),  # teal-blue
    "Cr": (0.008, 0.286, 0.271),  # dark teal
    "Mn": (0.541, 0.263, 0.208),  # brown-red
    "Fe": (0.812, 0.118, 0.067),  # red
    "Co": (0.169, 0.435, 0.400),  # medium teal
    "Ni": (0.067, 0.373, 0.333),  # teal
    "Cu": (0.663, 0.357, 0.165),  # copper brown
    "Zn": (0.106, 0.435, 0.576),  # steel blue
    "Ga": (0.651, 0.463, 0.314),  # warm brown
    "Ge": (0.541, 0.376, 0.310),  # brown
    "As": (0.482, 0.804, 0.796),  # light teal
    "Se": (0.996, 0.494, 0.239),  # orange
    "Br": (0.443, 0.016, 0.016),  # dark crimson
    "Kr": (0.667, 0.894, 0.824),  # light mint
    # Period 5
    "Rb": (0.325, 0.224, 0.086),  # dark brown
    "Sr": (0.435, 0.455, 0.110),  # olive green
    "Y":  (0.220, 0.478, 0.576),  # blue
    "Zr": (0.000, 0.243, 0.353),  # dark blue
    "Nb": (0.016, 0.255, 0.302),  # very dark teal
    "Mo": (0.000, 0.188, 0.204),  # very dark teal
    "Tc": (0.302, 0.502, 0.498),  # grey-teal
    "Ru": (0.302, 0.502, 0.498),  # grey-teal
    "Rh": (0.910, 0.510, 0.588),  # pink
    "Pd": (0.992, 0.722, 0.741),  # light pink
    "Ag": (0.710, 0.824, 0.871),  # light silver-blue
    "Cd": (0.992, 0.804, 0.545),  # peach
    "In": (0.541, 0.376, 0.310),  # brown
    "Sn": (0.675, 0.675, 0.675),  # grey
    "Sb": (0.780, 0.796, 0.329),  # yellow-green
    "Te": (0.851, 0.671, 0.000),  # gold
    "I":  (0.251, 0.173, 0.212),  # dark plum
    "Xe": (0.012, 0.741, 0.722),  # bright teal
    # Period 6
    "Cs": (0.561, 0.431, 0.024),  # dark gold
    "Ba": (0.765, 0.098, 0.129),  # red
    "La": (0.667, 0.894, 0.824),  # mint
    "Ce": (0.969, 0.718, 0.506),  # light orange
    "Pr": (0.600, 0.753, 0.631),  # sage green
    "Nd": (0.498, 0.694, 0.667),  # sage
    "Pm": (0.435, 0.655, 0.620),  # sage
    "Sm": (0.373, 0.616, 0.580),  # teal-sage
    "Eu": (0.310, 0.576, 0.545),  # teal
    "Gd": (0.263, 0.545, 0.510),  # teal
    "Tb": (0.220, 0.510, 0.478),  # teal
    "Dy": (0.169, 0.478, 0.443),  # teal
    "Ho": (0.130, 0.443, 0.408),  # teal
    "Er": (0.090, 0.408, 0.373),  # teal
    "Tm": (0.067, 0.373, 0.333),  # dark teal
    "Yb": (0.043, 0.341, 0.302),  # dark teal
    "Lu": (0.024, 0.310, 0.271),  # dark teal
    "Hf": (0.220, 0.478, 0.576),  # blue
    "Ta": (0.165, 0.435, 0.545),  # blue
    "W":  (0.106, 0.400, 0.510),  # blue
    "Re": (0.067, 0.365, 0.478),  # blue
    "Os": (0.035, 0.333, 0.443),  # dark blue
    "Ir": (0.016, 0.302, 0.408),  # dark blue
    "Pt": (0.886, 0.796, 0.478),  # pale gold
    "Au": (0.984, 0.796, 0.067),  # gold
    "Hg": (0.600, 0.600, 0.667),  # grey
    "Tl": (0.541, 0.376, 0.310),  # brown
    "Pb": (0.388, 0.373, 0.357),  # dark grey
    "Bi": (0.882, 0.416, 0.384),  # coral
    "Po": (0.541, 0.376, 0.200),  # brown
    "At": (0.400, 0.280, 0.240),  # dark brown
    "Rn": (0.510, 0.675, 0.635),  # grey-teal
    # Period 7
    "Fr": (0.325, 0.224, 0.086),  # dark brown
    "Ra": (0.435, 0.455, 0.110),  # olive
    "Ac": (0.373, 0.616, 0.682),  # blue
    "Th": (0.106, 0.435, 0.576),  # steel blue
    "Pa": (0.067, 0.400, 0.545),  # steel blue
    "U":  (0.016, 0.255, 0.329),  # navy
    "Np": (0.016, 0.224, 0.302),  # navy
    "Pu": (0.016, 0.200, 0.271),  # navy
    "Am": (0.310, 0.333, 0.580),  # muted blue
    "Cm": (0.420, 0.333, 0.545),  # muted purple
    "Bk": (0.480, 0.290, 0.545),  # muted purple
    "Cf": (0.541, 0.247, 0.510),  # muted purple
    "Es": (0.580, 0.200, 0.478),  # muted purple
    "Fm": (0.580, 0.180, 0.435),  # muted purple
    "Md": (0.580, 0.157, 0.400),  # muted purple
    "No": (0.600, 0.137, 0.365),  # muted purple
    "Lr": (0.620, 0.118, 0.329),  # muted purple
}

# Covalent radii in angstroms (Cordero et al., Dalton Trans. 2008).
COVALENT_RADII: dict[str, float] = {
    "H":  0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B":  0.84,
    "C":  0.76,
    "N":  0.71,
    "O":  0.66,
    "F":  0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P":  1.07,
    "S":  1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K":  2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V":  1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y":  1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I":  1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W":  1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.40,
    "At": 1.50,
    "Rn": 1.50,
    "Fr": 2.60,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2.00,
    "U":  1.96,
    "Np": 1.90,
    "Pu": 1.87,
    "Am": 1.80,
    "Cm": 1.69,
    "Bk": 1.68,
    "Cf": 1.68,
    "Es": 1.65,
    "Fm": 1.67,
    "Md": 1.73,
    "No": 1.76,
    "Lr": 1.61,
}


def default_bond_specs(
    species: list[str],
    *,
    tolerance: float = 0.4,
    bond_radius: float = 0.1,
    bond_colour: Colour = 0.5,
) -> list[BondSpec]:
    """Generate BondSpec rules from covalent radii for a set of species.

    Creates one spec per unique pair (including self-pairs) using the
    sum-of-covalent-radii heuristic: ``max_length = r_a + r_b + tolerance``.
    Species not found in :data:`COVALENT_RADII` are silently skipped.

    Args:
        species: Species labels to generate rules for.
        tolerance: Distance added to the sum of covalent radii (angstroms).
        bond_radius: Visual radius of the bond cylinder.
        bond_colour: Default colour for all generated bonds.

    Returns:
        A list of BondSpec rules, one per unique pair.
    """
    unique = sorted(set(species))
    # Filter to species with known radii.
    known = [s for s in unique if s in COVALENT_RADII]

    specs: list[BondSpec] = []
    for i, sp_a in enumerate(known):
        for sp_b in known[i:]:
            max_len = COVALENT_RADII[sp_a] + COVALENT_RADII[sp_b] + tolerance
            specs.append(BondSpec(
                species_a=sp_a,
                species_b=sp_b,
                min_length=0.0,
                max_length=max_len,
                radius=bond_radius,
                colour=bond_colour,
            ))
    return specs


def default_atom_style(element: str) -> AtomStyle:
    """Return a default AtomStyle for the given element symbol.

    Uses Cordero covalent radii and a muted colour palette.  Falls
    back to grey and a radius of 1.0 for unknown elements.

    Args:
        element: Chemical element symbol (e.g. ``"C"``, ``"Fe"``).

    Returns:
        An AtomStyle with default colour and radius.
    """
    colour = ELEMENT_COLOURS.get(element, (0.5, 0.5, 0.5))
    radius = COVALENT_RADII.get(element, 1.0)
    return AtomStyle(radius=radius, colour=colour)
