"""Hofmann: a modern Python reimagining of the XBS ball-and-stick viewer.

Hofmann renders crystal and molecular structures as ball-and-stick images,
with static publication-quality output via matplotlib.

Example usage::

    from hofmann import StructureScene

    scene = StructureScene.from_xbs("structure.bs")
    scene.render_mpl("output.svg")
"""

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.polyhedra import compute_polyhedra
from hofmann.construction.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
    default_bond_specs,
)
from hofmann.model import (
    AtomLegendItem,
    AtomStyle,
    AxesStyle,
    Bond,
    BondSpec,
    CellEdgeStyle,
    CmapSpec,
    Colour,
    Frame,
    LegendItem,
    LegendStyle,
    Polyhedron,
    PolygonLegendItem,
    PolyhedronLegendItem,
    PolyhedronSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    WidgetCorner,
    normalise_colour,
    resolve_atom_colours,
)
from hofmann.construction.scene_builders import from_pymatgen, from_xbs
from hofmann.construction.styles import StyleSet, load_styles, save_styles

__all__ = [
    "AtomLegendItem",
    "AtomStyle",
    "AxesStyle",
    "Bond",
    "BondSpec",
    "CellEdgeStyle",
    "CmapSpec",
    "COVALENT_RADII",
    "Colour",
    "ELEMENT_COLOURS",
    "Frame",
    "LegendItem",
    "LegendStyle",
    "Polyhedron",
    "PolygonLegendItem",
    "PolyhedronLegendItem",
    "PolyhedronSpec",
    "RenderStyle",
    "SlabClipMode",
    "StructureScene",
    "StyleSet",
    "ViewState",
    "WidgetCorner",
    "compute_bonds",
    "compute_polyhedra",
    "default_atom_style",
    "default_bond_specs",
    "from_pymatgen",
    "from_xbs",
    "load_styles",
    "normalise_colour",
    "resolve_atom_colours",
    "save_styles",
]
