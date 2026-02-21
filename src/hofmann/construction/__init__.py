"""Scene construction: parsing, bond detection, defaults, and builders."""

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.rendering_set import (
    RenderingSet,
    build_rendering_set,
    deduplicate_molecules,
)
from hofmann.construction.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
    default_bond_specs,
)
from hofmann.construction.parser import parse_bs, parse_mv
from hofmann.construction.polyhedra import compute_polyhedra
from hofmann.construction.scene_builders import from_pymatgen, from_xbs
from hofmann.construction.styles import StyleSet, load_styles, save_styles

__all__ = [
    "COVALENT_RADII",
    "ELEMENT_COLOURS",
    "RenderingSet",
    "StyleSet",
    "build_rendering_set",
    "compute_bonds",
    "compute_polyhedra",
    "deduplicate_molecules",
    "default_atom_style",
    "default_bond_specs",
    "from_pymatgen",
    "from_xbs",
    "load_styles",
    "parse_bs",
    "parse_mv",
    "save_styles",
]
