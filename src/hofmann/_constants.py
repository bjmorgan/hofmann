"""Shared constants used across the model and rendering layers."""

VALID_POLYHEDRA: frozenset[str] = frozenset({
    "octahedron", "tetrahedron", "cuboctahedron",
})
"""Recognised polyhedron shape names for legend icons."""

POLYHEDRON_RADIUS_SCALE: float = 2.0
"""Polyhedron icons default to this multiple of the flat-marker radius."""
