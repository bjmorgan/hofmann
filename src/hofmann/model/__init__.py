"""Core data model for hofmann: dataclasses, colour handling, and projection.

This package provides all the data types used throughout hofmann.
Everything is re-exported here so that ``from hofmann.model import
BondSpec`` continues to work.
"""

from hofmann.model.atom_style import AtomStyle
from hofmann.model.bond_spec import Bond, BondSpec
from hofmann.model.colour import (
    CmapSpec,
    Colour,
    normalise_colour,
    resolve_atom_colours,
)
from hofmann.model.frame import Frame
from hofmann.model.polyhedron_spec import Polyhedron, PolyhedronSpec
from hofmann.model.render_style import (
    AxesStyle,
    CellEdgeStyle,
    LegendStyle,
    RenderStyle,
    SlabClipMode,
    WidgetCorner,
    _DEFAULT_CIRCLE_RADIUS,
    _DEFAULT_SPACING,
)
from hofmann.model.structure_scene import StructureScene
from hofmann.model.view_state import ViewState

__all__ = [
    "AtomStyle",
    "AxesStyle",
    "Bond",
    "BondSpec",
    "CellEdgeStyle",
    "CmapSpec",
    "LegendStyle",
    "Colour",
    "Frame",
    "Polyhedron",
    "PolyhedronSpec",
    "RenderStyle",
    "SlabClipMode",
    "StructureScene",
    "ViewState",
    "WidgetCorner",
    "normalise_colour",
    "resolve_atom_colours",
]
