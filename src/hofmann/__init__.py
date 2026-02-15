"""Hofmann: a modern Python reimagining of the XBS ball-and-stick viewer.

Hofmann renders crystal and molecular structures as ball-and-stick images,
supporting both static publication-quality output (matplotlib) and
interactive 3D viewing (plotly).

Example usage::

    from hofmann import StructureScene

    scene = StructureScene.from_xbs("structure.bs")
    scene.render_mpl("output.svg")
"""

from hofmann.bonds import compute_bonds
from hofmann.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
    default_bond_specs,
)
from hofmann.model import (
    AtomStyle,
    Bond,
    BondSpec,
    Colour,
    Frame,
    RenderStyle,
    StructureScene,
    ViewState,
    normalise_colour,
)
from hofmann.scene import from_pymatgen, from_xbs

__all__ = [
    "AtomStyle",
    "Bond",
    "BondSpec",
    "COVALENT_RADII",
    "Colour",
    "ELEMENT_COLOURS",
    "Frame",
    "RenderStyle",
    "StructureScene",
    "ViewState",
    "compute_bonds",
    "default_atom_style",
    "default_bond_specs",
    "from_pymatgen",
    "from_xbs",
    "normalise_colour",
]
