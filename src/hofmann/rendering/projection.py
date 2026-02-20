"""Projection helpers and viewport sizing."""

from __future__ import annotations

import numpy as np

from hofmann.model import StructureScene, ViewState

# Default unit circle for atom rendering (closed polygon).
_N_CIRCLE = 24
_UNIT_CIRCLE = np.column_stack([
    np.cos(np.linspace(0, 2 * np.pi, _N_CIRCLE + 1)),
    np.sin(np.linspace(0, 2 * np.pi, _N_CIRCLE + 1)),
])


def _make_unit_circle(n: int) -> np.ndarray:
    """Build a unit circle polygon with *n* segments."""
    if n == _N_CIRCLE:
        return _UNIT_CIRCLE
    return np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, n + 1)),
        np.sin(np.linspace(0, 2 * np.pi, n + 1)),
    ])


def _project_point(
    pt: np.ndarray,
    view: ViewState,
) -> tuple[np.ndarray, float]:
    """Project a single 3D rotated point to 2D screen coordinates.

    Args:
        pt: 3D point in rotated (camera) coordinates.
        view: The ViewState defining the projection.

    Returns:
        Tuple of (xy, scale) where *xy* is the 2D position and *scale*
        is the perspective scale factor at this depth.
    """
    z = pt[2]
    if view.perspective > 0:
        s = view.view_distance / (view.view_distance - z * view.perspective)
    else:
        s = 1.0
    xy = pt[:2] * s * view.zoom
    return xy, s


# Fractional coordinates of the 8 cube corners (row-order matches
# the bit-pattern vertex indexing: 0->(0,0,0), 1->(1,0,0), ..., 7->(1,1,1)).
_FRAC_CORNERS = np.array([
    [(v >> 0) & 1, (v >> 1) & 1, (v >> 2) & 1]
    for v in range(8)
], dtype=float)


def _scene_extent(
    scene: StructureScene,
    view: ViewState,
    frame_index: int,
    atom_scale: float,
) -> float:
    """Compute rotation-invariant viewport half-extent for *scene*.

    Returns the radius of a 2D bounding circle centred at the origin
    that encloses all atoms regardless of rotation.  This is simply the
    maximum 3D distance from the view centre plus the largest display
    radius, scaled by zoom.
    """
    coords = scene.frames[frame_index].coords
    dists = np.linalg.norm(coords - view.centre, axis=1)

    radii_3d = np.empty(len(scene.species))
    for i, sp in enumerate(scene.species):
        style = scene.atom_styles.get(sp)
        radii_3d[i] = style.radius if style is not None else 0.5

    max_extent = np.max(dists + radii_3d * atom_scale)

    # Include cell corners when a lattice is present.
    if scene.lattice is not None:
        corners = _FRAC_CORNERS @ scene.lattice  # (8, 3)
        corner_dists = np.linalg.norm(corners - view.centre, axis=1)
        max_extent = max(max_extent, float(np.max(corner_dists)))

    # Under perspective, atoms near the camera appear larger.  The
    # worst-case magnification for an atom at distance *d* from the
    # view centre is when it is rotated to depth z = +d (closest to
    # the camera).
    if view.perspective > 0:
        worst_depth = np.max(dists)
        denom = view.view_distance - worst_depth * view.perspective
        if denom > 0:
            persp_scale = view.view_distance / denom
        else:
            persp_scale = view.view_distance / 1e-6
        max_extent *= persp_scale

    return float(max_extent * view.zoom)
