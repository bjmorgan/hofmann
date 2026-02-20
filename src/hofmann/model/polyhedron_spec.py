from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hofmann.model._util import _field_defaults
from hofmann.model.colour import Colour, normalise_colour


@dataclass
class PolyhedronSpec:
    """Declarative rule for rendering coordination polyhedra.

    A polyhedron is drawn around each atom whose species matches
    *centre*, using its bonded neighbours as vertices of a convex
    hull.  Species names support fnmatch-style wildcards.

    Attributes:
        centre: Species pattern for the centre atom (e.g. ``"Ti"``).
        colour: Face colour, or ``None`` to inherit from the centre
            atom's style colour.
        alpha: Face transparency (0 = fully transparent, 1 = opaque).
        edge_colour: Colour for face wireframe edges.
        edge_width: Line width for face wireframe edges (points).
        hide_centre: Whether to hide the centre atom circle when
            a polyhedron is drawn.
        hide_bonds: Whether to hide bonds from the centre atom to
            its coordinating neighbours when a polyhedron is drawn.
        hide_vertices: Whether to hide the vertex atom circles.
            An atom is only hidden if *every* polyhedron it
            participates in has ``hide_vertices=True``.
        min_vertices: Minimum number of bonded neighbours required
            to draw a polyhedron.  Centre atoms with fewer neighbours
            are skipped.  ``None`` uses the default minimum of 3.
    """

    centre: str
    colour: Colour | None = None
    alpha: float = 0.4
    edge_colour: Colour = (0.15, 0.15, 0.15)
    edge_width: float = 1.0
    hide_centre: bool = False
    hide_bonds: bool = False
    hide_vertices: bool = False
    min_vertices: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be between 0.0 and 1.0, got {self.alpha}"
            )
        if self.edge_width < 0:
            raise ValueError(
                f"edge_width must be non-negative, got {self.edge_width}"
            )
        if self.min_vertices is not None and self.min_vertices < 3:
            raise ValueError(
                f"min_vertices must be >= 3, got {self.min_vertices}"
            )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted.  Colours are
        normalised to ``[r, g, b]`` lists.
        """
        # centre and colour are handled separately above.
        defaults = _field_defaults(
            type(self), exclude=frozenset({"centre", "colour"}),
        )
        d: dict = {"centre": self.centre}
        if self.colour is not None:
            d["colour"] = list(normalise_colour(self.colour))
        for field_name, default in defaults.items():
            val = getattr(self, field_name)
            if field_name == "edge_colour":
                if normalise_colour(val) != normalise_colour(default):
                    d[field_name] = list(normalise_colour(val))
            else:
                if val != default:
                    d[field_name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PolyhedronSpec:
        """Deserialise from a dictionary."""
        defaults = _field_defaults(cls, exclude=frozenset({"centre", "colour"}))
        kwargs: dict = {"centre": d["centre"]}
        if "colour" in d:
            val = d["colour"]
            kwargs["colour"] = tuple(val) if isinstance(val, list) else val
        for field_name in defaults:
            if field_name in d:
                val = d[field_name]
                if field_name == "edge_colour" and isinstance(val, list):
                    val = tuple(val)
                kwargs[field_name] = val
        return cls(**kwargs)


@dataclass(frozen=True)
class Polyhedron:
    """A computed coordination polyhedron.

    Attributes:
        centre_index: Index of the centre atom.
        neighbour_indices: Indices of the coordinating atoms.
        faces: List of faces, each a 1-D array of vertex indices
            into *neighbour_indices*.  Triangular faces have length 3;
            merged coplanar faces may have 4 or more vertices.
        spec: The PolyhedronSpec that produced this polyhedron.
    """

    centre_index: int
    neighbour_indices: tuple[int, ...]
    faces: list[np.ndarray]
    spec: PolyhedronSpec
