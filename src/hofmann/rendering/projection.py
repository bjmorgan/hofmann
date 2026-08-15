"""Projection helpers and viewport sizing."""

from __future__ import annotations

import numpy as np

from hofmann.model import (
    Oblique,
    Orthographic,
    Perspective,
    StructureScene,
    ViewState,
)
from hofmann.model.composition import Composition, _OCCUPANCY_TOLERANCE
from hofmann.rendering.precompute import _compute_atom_radii

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

    Thin single-point wrapper around :meth:`ViewState.project_camera`.

    Args:
        pt: 3D point in rotated (camera) coordinates.
        view: The ViewState defining the projection.

    Returns:
        Tuple of (xy, scale) where *xy* is the 2D position and *scale*
        is the perspective scale factor at this depth.
    """
    xy, scale = view.project_camera(pt[np.newaxis, :])
    return xy[0], float(scale[0])


# Fractional coordinates of the 8 unit cube corners.
_FRAC_CORNERS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
], dtype=float)


def _scene_extent(
    scene: StructureScene,
    view: ViewState,
    frame_index: int,
    atom_scale: float,
) -> float:
    """Compute rotation-invariant viewport half-extent for *scene*.

    Returns the radius of a 2D bounding circle centred at the origin
    that encloses all atoms regardless of rotation.  The base bound is
    the maximum 3D distance from the view centre plus the largest
    display radius; the projection then applies its own allowance on
    top.  The bound is zoom-independent: zoom acts
    through the projected coordinates, so this fixed viewport must not
    itself scale with it.
    """
    coords = scene.frames[frame_index].coords
    dists = np.linalg.norm(coords - view.centre, axis=1)

    if len(dists) > 0:
        radii_3d = _compute_atom_radii(scene.species, scene.atom_styles)
        max_extent = float(np.max(dists + radii_3d * atom_scale))
    else:
        max_extent = 0.0

    # Include cell corners when a lattice is present.
    lattice = scene.frames[frame_index].lattice
    if lattice is not None:
        corners = _FRAC_CORNERS @ lattice  # (8, 3)
        corner_dists = np.linalg.norm(corners - view.centre, axis=1)
        max_extent = max(max_extent, float(np.max(corner_dists)))

    # Ensure a positive extent even for empty scenes.
    if max_extent == 0.0:
        max_extent = 1.0

    match view._validated_projection():
        case Oblique():
            # An oblique shear stretches screen positions by at most
            # screen_scale_bound (sqrt(1 + f^2), the screen matrix's
            # largest singular value).  Applying it to the whole
            # dists-plus-radii bound is slightly conservative —
            # silhouette radii stay unscaled under oblique — which is
            # harmless over-padding.
            max_extent *= view.screen_scale_bound
        case Perspective() as p:
            # Under perspective, geometry near the camera appears
            # larger.  The worst-case magnification for a point at
            # distance *d* from the view centre is when it is rotated
            # to depth z = +d (closest to the camera); the scene
            # bounding radius (atoms plus display radii, and cell
            # corners) bounds that depth.
            denom = p.view_distance - max_extent * p.strength
            if denom > 0:
                max_extent *= p.view_distance / denom
            # Once the eye reaches the bounding sphere no bounded
            # magnification exists, so the viewport falls back to the
            # unmagnified scene bound.
        case Orthographic():
            pass

    return float(max_extent)


def _make_wedges(
    composition: Composition,
    n_segments_total: int,
    start_angle: float,
) -> list[tuple[str, np.ndarray]]:
    """Build wedge polygons for a mixed-site composition.

    Each wedge is a closed polygon (centre + arc vertices + closing
    vertex back to the centre) sweeping an angle of exactly
    ``2π · occ``.  Wedge *angles* are therefore always proportional
    to occupancy — independent of any segment allocation.

    *n_segments_total* controls only the smoothness of the rendered
    arcs.  Each wedge gets ``max(1, round(n_segments_total * occ))``
    arc segments: a per-wedge target proportional to occupancy, with
    a minimum of one segment so that every wedge has at least two
    arc vertices.  The sum across wedges is therefore a soft target,
    not a hard cap — small constituents that round below one segment
    are bumped up.  In practice the overshoot is at most a couple
    of segments at the default budget, well below display resolution.

    Args:
        composition: The site composition (iteration order is canonical).
        n_segments_total: Target arc-segment count for a full circle,
            controlling arc smoothness.  Per-wedge counts are derived
            proportionally with a minimum of one segment.
        start_angle: Starting angle in radians (counter-clockwise from
            the +x axis).

    Returns:
        A list of ``(species_label, polygon)`` pairs in canonical
        composition order.  ``polygon`` has shape ``(k, 2)`` where
        ``k`` is the number of vertices (``k >= 4``: centre, at
        least two arc vertices, closing centre).
    """
    occupancies = list(composition.items())
    total_occ = sum(occ for _, occ in occupancies)
    if total_occ <= 0.0:
        return []

    raw_alloc = [
        max(1, int(round(n_segments_total * occ)))
        for _, occ in occupancies
    ]

    wedges: list[tuple[str, np.ndarray]] = []
    angle = start_angle
    for (species_label, occ), n_seg in zip(occupancies, raw_alloc):
        wedge_angle = 2.0 * np.pi * occ
        thetas = np.linspace(angle, angle + wedge_angle, n_seg + 1)
        arc = np.column_stack([np.cos(thetas), np.sin(thetas)])
        polygon = np.vstack([
            np.array([[0.0, 0.0]]),
            arc,
            np.array([[0.0, 0.0]]),
        ])
        wedges.append((species_label, polygon))
        angle += wedge_angle

    return wedges


def _make_vacancy_wedge(
    composition: Composition,
    n_segments_total: int,
    start_angle: float,
) -> np.ndarray | None:
    """Build the leftover-arc polygon for a partially occupied composition.

    Returns a closed polygon (centre + arc + close) covering the
    vacancy fraction of *composition*.  Returns ``None`` when the
    composition is fully occupied.

    Args:
        composition: The site composition.
        n_segments_total: Total arc segments allocated to a full circle.
        start_angle: Starting angle for the species wedges (radians).

    Returns:
        A polygon array of shape ``(k, 2)``, or ``None`` if no vacancy.
    """
    total_occ = sum(composition.values())
    if total_occ >= 1.0 - _OCCUPANCY_TOLERANCE:
        return None
    vacancy_frac = 1.0 - total_occ
    angle = start_angle + 2.0 * np.pi * total_occ
    n_seg = max(1, int(round(n_segments_total * vacancy_frac)))
    thetas = np.linspace(angle, angle + 2.0 * np.pi * vacancy_frac, n_seg + 1)
    arc = np.column_stack([np.cos(thetas), np.sin(thetas)])
    polygon = np.vstack([
        np.array([[0.0, 0.0]]),
        arc,
        np.array([[0.0, 0.0]]),
    ])
    return polygon
