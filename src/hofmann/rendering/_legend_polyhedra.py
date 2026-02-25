"""Canonical polyhedra data for 3D legend icons.

Provides unit-radius vertex sets, a fixed rotation matrix for the
legend viewing angle, cached face computation, and a per-face
shading helper.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Canonical vertex sets (origin-centred, unit circumradius)
# ---------------------------------------------------------------------------

CANONICAL_VERTICES: dict[str, np.ndarray] = {
    "octahedron": np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float),
    "tetrahedron": np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) / np.sqrt(3),
    "cuboctahedron": np.array([
        [0,  1,  1], [0,  1, -1], [0, -1,  1], [0, -1, -1],
        [1,  0,  1], [1,  0, -1], [-1,  0,  1], [-1,  0, -1],
        [1,  1,  0], [1, -1,  0], [-1,  1,  0], [-1, -1,  0],
    ], dtype=float) / np.sqrt(2),
}

SUPPORTED_POLYHEDRA: frozenset[str] = frozenset(CANONICAL_VERTICES)
"""Recognised polyhedron shape names for legend icons.

Derived from :data:`CANONICAL_VERTICES` so the two cannot drift
out of sync.  Must match ``_VALID_POLYHEDRA`` in
:mod:`hofmann.model.render_style`.
"""

# ---------------------------------------------------------------------------
# Fixed legend rotation: Ry(-15 deg) @ Rx(10 deg)
# ---------------------------------------------------------------------------

_ANGLE_Y = np.radians(-15)
_ANGLE_X = np.radians(10)

_Ry = np.array([
    [np.cos(_ANGLE_Y), 0, np.sin(_ANGLE_Y)],
    [0, 1, 0],
    [-np.sin(_ANGLE_Y), 0, np.cos(_ANGLE_Y)],
])
_Rx = np.array([
    [1, 0, 0],
    [0, np.cos(_ANGLE_X), -np.sin(_ANGLE_X)],
    [0, np.sin(_ANGLE_X), np.cos(_ANGLE_X)],
])

LEGEND_ROTATION: np.ndarray = _Ry @ _Rx
"""Fixed oblique rotation matrix for legend polyhedron icons."""


# ---------------------------------------------------------------------------
# Cached face computation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _get_faces(name: str) -> list[np.ndarray]:
    """Return merged convex-hull faces for a canonical polyhedron.

    Faces are computed once and cached for the lifetime of the process.

    Args:
        name: Polyhedron shape name (must be a key in
            :data:`CANONICAL_VERTICES`).

    Returns:
        List of face vertex-index arrays (each a 1-D ``ndarray``).

    Raises:
        ValueError: If *name* is not a recognised shape.
    """
    if name not in CANONICAL_VERTICES:
        raise ValueError(
            f"Unknown polyhedron shape {name!r}; "
            f"expected one of {sorted(CANONICAL_VERTICES)}"
        )
    from hofmann.construction.polyhedra import _triangulate

    vertices = CANONICAL_VERTICES[name]
    faces = _triangulate(vertices)
    if faces is None:  # pragma: no cover — canonical shapes always succeed
        raise RuntimeError(f"Triangulation failed for {name!r}")
    return faces


# ---------------------------------------------------------------------------
# Per-face shading
# ---------------------------------------------------------------------------

def shade_face(
    face_vertices_rotated: np.ndarray,
    base_rgb: tuple[float, float, float],
    polyhedra_shading: float,
) -> tuple[float, float, float]:
    """Compute shaded colour for a single polyhedron face.

    Uses the same Lambertian-style formula as the main painter:
    ``shading = 1.0 - polyhedra_shading * 0.6 * (1.0 - cos_angle)``
    where *cos_angle* is the absolute cosine between the face normal
    and the viewer direction (positive z).

    Args:
        face_vertices_rotated: Rotated vertex coordinates for the
            face, shape ``(n, 3)``.
        base_rgb: Base face colour before shading.
        polyhedra_shading: Shading strength (0 = flat, 1 = full).

    Returns:
        Shaded ``(R, G, B)`` tuple, each channel clamped to [0, 1].
    """
    normal = np.cross(
        face_vertices_rotated[1] - face_vertices_rotated[0],
        face_vertices_rotated[2] - face_vertices_rotated[0],
    )
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-12:
        cos_angle = abs(normal[2] / norm_len)
    else:
        cos_angle = 0.0
    shading = 1.0 - polyhedra_shading * 0.6 * (1.0 - cos_angle)
    return tuple(min(1.0, c * shading) for c in base_rgb)  # type: ignore[return-value]
