"""Depth-sorted matplotlib renderer for publication-quality output.

Implements the painter's algorithm: atoms are sorted back-to-front, and
each atom draws its associated bonds before the next atom is painted on
top.

Bond-sphere intersections are computed in 3D (rotated coordinates) and
then projected to screen space.

Provides both a static renderer (:func:`render_mpl`) and an interactive
viewer (:func:`render_mpl_interactive`) with mouse-driven rotation and
zoom.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

from hofmann.construction.bonds import compute_bonds
from hofmann.model import (
    AxesStyle,
    Bond,
    CellEdgeStyle,
    CmapSpec,
    Colour,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    WidgetCorner,
    normalise_colour,
    resolve_atom_colours,
)
from hofmann.construction.polyhedra import compute_polyhedra

# Font size (points) for scene titles rendered inside the viewport.
_TITLE_FONT_SIZE = 12.0

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


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _clip_bond_3d(
    p_a: np.ndarray,
    p_b: np.ndarray,
    r_a: float,
    r_b: float,
    bond_r: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute 3D clipped bond endpoints where the cylinder meets each sphere.

    The cylinder of radius *bond_r* is tangent to each atom sphere.
    The tangent circle sits at distance ``sqrt(atom_r^2 - bond_r^2)``
    from the sphere centre along the bond axis (Pythagoras).

    Args:
        p_a: 3D position of atom a in rotated coordinates.
        p_b: 3D position of atom b in rotated coordinates.
        r_a: 3D radius of atom a.
        r_b: 3D radius of atom b.
        bond_r: 3D radius of the bond cylinder.

    Returns:
        ``(clip_start, clip_end)`` in 3D, or ``None`` if the bond is
        fully occluded (clipped to zero or negative length).
    """
    bond_vec = p_b - p_a
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-12:
        return None
    bond_unit = bond_vec / bond_len

    w_a = np.sqrt(max(r_a**2 - bond_r**2, 0.0)) if r_a > bond_r else 0.0
    w_b = np.sqrt(max(r_b**2 - bond_r**2, 0.0)) if r_b > bond_r else 0.0

    clip_start = p_a + bond_unit * w_a
    clip_end = p_b - bond_unit * w_b

    if np.dot(clip_end - clip_start, bond_unit) <= 0:
        return None

    return clip_start, clip_end


def _stick_polygon(
    start: np.ndarray,
    end: np.ndarray,
    hw_start: float,
    hw_end: float,
) -> np.ndarray | None:
    """Compute vertices for a bond stick drawn as a filled rectangle.

    Supports different half-widths at each end for perspective tapering.

    Returns an (4, 2) array of rectangle corners, or ``None`` if
    degenerate.
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return None
    unit = direction / length
    perp = np.array([-unit[1], unit[0]])

    return np.array([
        start + perp * hw_start,
        start - perp * hw_start,
        end - perp * hw_end,
        end + perp * hw_end,
    ])


def _clip_polygon_to_half_plane(
    verts: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Clip a polygon to the half-plane where dot(v - point, normal) >= 0.

    Uses the Sutherland-Hodgman algorithm on a single clipping edge.
    Returns the clipped polygon vertices in order, or an empty array
    if nothing remains.

    Args:
        verts: ``(n, 2)`` polygon vertices in winding order.
        point: A point on the clipping line.
        normal: Outward normal of the half-plane to keep.

    Returns:
        ``(m, 2)`` clipped polygon vertices.
    """
    n = len(verts)
    if n < 3:
        return verts
    out: list[np.ndarray] = []
    d = np.dot(verts - point, normal)
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        if di >= 0:
            out.append(verts[i])
        # Edge crosses the line if signs differ.
        if (di >= 0) != (dj >= 0):
            t = di / (di - dj)
            out.append(verts[i] + t * (verts[j] - verts[i]))
    if len(out) < 3:
        return np.empty((0, 2))
    return np.array(out)



# Default semicircular arc (5 points per half-turn).
_N_ARC = 5
_ARC = np.column_stack([
    -np.sin(np.linspace(0, np.pi, _N_ARC)),
     np.cos(np.linspace(0, np.pi, _N_ARC)),
])


def _make_arc(n: int) -> np.ndarray:
    """Build a semicircular arc with *n* points."""
    if n == _N_ARC:
        return _ARC
    return np.column_stack([
        -np.sin(np.linspace(0, np.pi, n)),
         np.cos(np.linspace(0, np.pi, n)),
    ])


def _bond_polygon(
    p_a: np.ndarray,
    p_b: np.ndarray,
    r_a: float,
    r_b: float,
    bond_r: float,
    zr_a: float,
    zr_b: float,
    view: ViewState,
    arc: np.ndarray = _ARC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute bond polygon vertices with perspective-correct arc ends.

    The bond cylinder is clipped in 3D where it meets each atom
    sphere, then the clipped endpoints are projected to 2D.  Each
    bond end is drawn as a semicircular arc transformed by a 2D
    affine matrix that encodes the projected bond radius
    (perpendicular) and foreshortening (along-axis), following the
    XBS approach.

    Args:
        p_a: 3D position of atom a in rotated coordinates.
        p_b: 3D position of atom b in rotated coordinates.
        r_a: Scaled 3D radius of atom a (after atom_scale).
        r_b: Scaled 3D radius of atom b (after atom_scale).
        bond_r: Scaled bond radius (after bond_scale).
        zr_a: Projected display radius of atom a (after atom_scale).
        zr_b: Projected display radius of atom b (after atom_scale).
        view: The ViewState for projection.

    Returns:
        Tuple of ``(verts, start_2d, end_2d)`` where *verts* is a
        ``(2 * _N_ARC, 2)`` array of polygon vertices and *start_2d*,
        *end_2d* are the projected 2D centres of the two bond ends.
        Returns ``None`` if the bond is fully occluded.
    """
    # 3D clip using scaled radii.
    clip_result = _clip_bond_3d(p_a, p_b, r_a, r_b, bond_r)
    if clip_result is None:
        return None

    bond_vec = p_b - p_a
    bond_len = np.linalg.norm(bond_vec)

    # Tangent offsets in 3D (same as _clip_bond_3d computes internally).
    w_a = np.sqrt(max(r_a**2 - bond_r**2, 0.0)) if r_a > bond_r else 0.0
    w_b = np.sqrt(max(r_b**2 - bond_r**2, 0.0)) if r_b > bond_r else 0.0

    # Foreshortening: cosine of angle between bond axis and eye-to-atom
    # vector.  This determines how much the arc squashes along the bond
    # direction (a bond pointing at the viewer has cth~1, one
    # perpendicular to the view has cth~0).
    # For orthographic projection, push the eye to effective infinity
    # so all view rays are parallel (matching XBS pmode==0 behaviour).
    eye_dist = view.view_distance if view.perspective > 0 else 1e6
    eye = np.array([0.0, 0.0, eye_dist])
    q_a = eye - p_a
    q_b = eye - p_b
    denom_a = np.linalg.norm(q_a) * bond_len
    denom_b = np.linalg.norm(q_b) * bond_len
    if denom_a < 1e-12 or denom_b < 1e-12:
        return None
    cth_a = abs(np.dot(q_a, bond_vec) / denom_a)
    cth_b = abs(np.dot(q_b, bond_vec) / denom_b)
    sth_a = np.sqrt(max(1.0 - cth_a * cth_a, 0.0))
    sth_b = np.sqrt(max(1.0 - cth_b * cth_b, 0.0))

    # Project atom centres to 2D, then offset along the 2D bond
    # direction by the projected tangent distance.
    atom_a_2d, _ = _project_point(p_a, view)
    atom_b_2d, _ = _project_point(p_b, view)

    bond_2d = atom_b_2d - atom_a_2d
    bond_2d_len = np.linalg.norm(bond_2d)
    if bond_2d_len < 1e-12:
        return None
    bx, by = bond_2d / bond_2d_len

    # Ratio of projected display radius to 3D radius — encodes both
    # perspective scaling and atom_scale.
    zr_rk_a = zr_a / r_a if r_a > 1e-12 else 0.0
    zr_rk_b = zr_b / r_b if r_b > 1e-12 else 0.0

    # Projected tangent offset: w * sin(theta) * zr/r.
    ww_a = w_a * sth_a * zr_rk_a
    ww_b = w_b * sth_b * zr_rk_b

    start_2d = atom_a_2d + np.array([bx, by]) * ww_a
    end_2d = atom_b_2d - np.array([bx, by]) * ww_b

    # Affine matrix components for the arc transformation.
    # bb = perpendicular half-width, aa = along-axis foreshortening.
    bb_a = bond_r * zr_rk_a
    aa_a = bond_r * cth_a * zr_rk_a
    bb_b = bond_r * zr_rk_b
    aa_b = bond_r * cth_b * zr_rk_b

    # Build polygon: arc at start, reversed arc at end.
    pts_start = np.column_stack([
        bx * aa_a * arc[:, 0] + (-by) * bb_a * arc[:, 1] + start_2d[0],
        by * aa_a * arc[:, 0] +   bx  * bb_a * arc[:, 1] + start_2d[1],
    ])
    pts_end = np.column_stack([
        -bx * aa_b * arc[:, 0] + (-by) * bb_b * arc[:, 1] + end_2d[0],
        -by * aa_b * arc[:, 0] +   bx  * bb_b * arc[:, 1] + end_2d[1],
    ])
    pts_end = pts_end[::-1]

    verts = np.vstack([pts_start, pts_end])
    return verts, start_2d, end_2d


def _bond_polygons_batch(
    rotated: np.ndarray,
    xy: np.ndarray,
    radii_3d: np.ndarray,
    screen_radii: np.ndarray,
    bond_ia: np.ndarray,
    bond_ib: np.ndarray,
    bond_radii: np.ndarray,
    view: ViewState,
    arc: np.ndarray = _ARC,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray,
]:
    """Compute all bond polygon vertices in a single vectorised pass.

    This is the batch equivalent of calling :func:`_bond_polygon` for
    each bond individually.  All intermediate quantities are computed
    on ``(n_bonds,)`` arrays via numpy broadcasting.

    Args:
        rotated: Rotated 3D coordinates, shape ``(n_atoms, 3)``.
        xy: Projected 2D coordinates, shape ``(n_atoms, 2)``.
        radii_3d: Scaled 3D atom radii, shape ``(n_atoms,)``.
        screen_radii: Projected screen radii, shape ``(n_atoms,)``.
        bond_ia: Index of atom a for each bond, shape ``(n_bonds,)``.
        bond_ib: Index of atom b for each bond, shape ``(n_bonds,)``.
        bond_radii: Scaled bond cylinder radius, shape ``(n_bonds,)``.
        view: The ViewState for projection parameters.

    Returns:
        Tuple of ``(full_verts, start_2d, end_2d, bb_a, bb_b, bx, by,
        valid)`` where:

        - *full_verts*: ``(n_bonds, 2 * _N_ARC, 2)`` polygon vertices.
        - *start_2d*: ``(n_bonds, 2)`` projected bond start centres.
        - *end_2d*: ``(n_bonds, 2)`` projected bond end centres.
        - *bb_a*: ``(n_bonds,)`` perpendicular half-width at start.
        - *bb_b*: ``(n_bonds,)`` perpendicular half-width at end.
        - *bx*: ``(n_bonds,)`` bond 2D direction x component.
        - *by*: ``(n_bonds,)`` bond 2D direction y component.
        - *valid*: ``(n_bonds,)`` boolean mask (``False`` for
          occluded/degenerate bonds).
    """
    n_arc = len(arc)
    n_bonds = len(bond_ia)
    if n_bonds == 0:
        empty2 = np.empty((0, 2))
        empty1 = np.empty((0,))
        return (
            np.empty((0, 2 * n_arc, 2)),
            empty2, empty2,
            empty1, empty1, empty1, empty1,
            np.empty((0,), dtype=bool),
        )

    # Gather per-bond atom data.
    p_a = rotated[bond_ia]              # (n_bonds, 3)
    p_b = rotated[bond_ib]              # (n_bonds, 3)
    r_a = radii_3d[bond_ia]             # (n_bonds,)
    r_b = radii_3d[bond_ib]             # (n_bonds,)
    zr_a = screen_radii[bond_ia]        # (n_bonds,)
    zr_b = screen_radii[bond_ib]        # (n_bonds,)

    # Bond vectors and lengths.
    bond_vec = p_b - p_a                                    # (n_bonds, 3)
    bond_len = np.linalg.norm(bond_vec, axis=1)             # (n_bonds,)
    valid = bond_len > 1e-12
    bond_len_safe = np.where(valid, bond_len, 1.0)

    # Tangent offsets (vectorised _clip_bond_3d).
    w_a = np.where(
        r_a > bond_radii,
        np.sqrt(np.maximum(r_a**2 - bond_radii**2, 0.0)),
        0.0,
    )
    w_b = np.where(
        r_b > bond_radii,
        np.sqrt(np.maximum(r_b**2 - bond_radii**2, 0.0)),
        0.0,
    )
    valid &= (bond_len_safe - w_a - w_b) > 0

    # Foreshortening angles.
    eye_dist = view.view_distance if view.perspective > 0 else 1e6
    eye = np.array([0.0, 0.0, eye_dist])
    q_a = eye - p_a                                         # (n_bonds, 3)
    q_b = eye - p_b                                         # (n_bonds, 3)
    q_a_len = np.linalg.norm(q_a, axis=1)                   # (n_bonds,)
    q_b_len = np.linalg.norm(q_b, axis=1)                   # (n_bonds,)
    denom_a = q_a_len * bond_len_safe
    denom_b = q_b_len * bond_len_safe
    valid &= (denom_a > 1e-12) & (denom_b > 1e-12)
    denom_a_safe = np.where(valid, denom_a, 1.0)
    denom_b_safe = np.where(valid, denom_b, 1.0)

    dot_qa = np.sum(q_a * bond_vec, axis=1)
    dot_qb = np.sum(q_b * bond_vec, axis=1)
    cth_a = np.abs(dot_qa / denom_a_safe)
    cth_b = np.abs(dot_qb / denom_b_safe)
    sth_a = np.sqrt(np.maximum(1.0 - cth_a**2, 0.0))
    sth_b = np.sqrt(np.maximum(1.0 - cth_b**2, 0.0))

    # 2D bond direction.
    a_2d = xy[bond_ia]                                      # (n_bonds, 2)
    b_2d = xy[bond_ib]                                      # (n_bonds, 2)
    bond_2d = b_2d - a_2d                                   # (n_bonds, 2)
    bond_2d_len = np.linalg.norm(bond_2d, axis=1)           # (n_bonds,)
    valid &= bond_2d_len > 1e-12
    bond_2d_len_safe = np.where(valid, bond_2d_len, 1.0)
    bx = bond_2d[:, 0] / bond_2d_len_safe
    by = bond_2d[:, 1] / bond_2d_len_safe

    # Projection ratio.
    zr_rk_a = np.where(r_a > 1e-12, zr_a / r_a, 0.0)
    zr_rk_b = np.where(r_b > 1e-12, zr_b / r_b, 0.0)

    # Projected tangent offsets.
    ww_a = w_a * sth_a * zr_rk_a
    ww_b = w_b * sth_b * zr_rk_b
    bxy = np.column_stack([bx, by])                         # (n_bonds, 2)
    start_2d = a_2d + bxy * ww_a[:, np.newaxis]
    end_2d = b_2d - bxy * ww_b[:, np.newaxis]

    # Affine components.
    bb_a = bond_radii * zr_rk_a
    aa_a = bond_radii * cth_a * zr_rk_a
    bb_b = bond_radii * zr_rk_b
    aa_b = bond_radii * cth_b * zr_rk_b

    # Arc vertex construction via broadcasting.
    # arc is (n_arc, 2); arc_x/arc_y are (n_arc,).
    arc_x = arc[:, 0]                                        # (n_arc,)
    arc_y = arc[:, 1]                                        # (n_arc,)

    # Start arc: (n_bonds, n_arc, 2).
    pts_s_x = (bx[:, None] * aa_a[:, None] * arc_x
               - by[:, None] * bb_a[:, None] * arc_y
               + start_2d[:, 0:1])
    pts_s_y = (by[:, None] * aa_a[:, None] * arc_x
               + bx[:, None] * bb_a[:, None] * arc_y
               + start_2d[:, 1:2])
    pts_start = np.stack([pts_s_x, pts_s_y], axis=-1)

    # End arc: (n_bonds, _N_ARC, 2) — negated bond direction.
    pts_e_x = (-bx[:, None] * aa_b[:, None] * arc_x
               - by[:, None] * bb_b[:, None] * arc_y
               + end_2d[:, 0:1])
    pts_e_y = (-by[:, None] * aa_b[:, None] * arc_x
               + bx[:, None] * bb_b[:, None] * arc_y
               + end_2d[:, 1:2])
    pts_end = np.stack([pts_e_x, pts_e_y], axis=-1)
    pts_end = pts_end[:, ::-1, :]  # Reverse winding per bond.

    full_verts = np.concatenate([pts_start, pts_end], axis=1)

    return full_verts, start_2d, end_2d, bb_a, bb_b, bx, by, valid


def _half_bond_verts_batch(
    full_verts: np.ndarray,
    start_2d: np.ndarray,
    end_2d: np.ndarray,
    bb_a: np.ndarray,
    bb_b: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    n_arc: int = _N_ARC,
) -> tuple[np.ndarray, np.ndarray]:
    """Build half-bond polygons directly without Sutherland-Hodgman clipping.

    Each half-bond has a fixed vertex count: ``_N_ARC`` arc vertices on
    the atom side plus 2 straight-cut vertices at the midpoint, giving
    ``_N_ARC + 2`` vertices per half.

    The start arc in *full_verts* traces from the +perp side to the
    -perp side of the bond.  The midpoint cut closes the polygon with
    two points at the interpolated half-width.

    Args:
        full_verts: ``(n_bonds, 2 * _N_ARC, 2)`` from
            :func:`_bond_polygons_batch`.
        start_2d: ``(n_bonds, 2)`` projected start centres.
        end_2d: ``(n_bonds, 2)`` projected end centres.
        bb_a: ``(n_bonds,)`` perpendicular half-width at start.
        bb_b: ``(n_bonds,)`` perpendicular half-width at end.
        bx: ``(n_bonds,)`` bond 2D direction x.
        by: ``(n_bonds,)`` bond 2D direction y.

    Returns:
        Tuple of ``(half_a, half_b)`` each of shape
        ``(n_bonds, _N_ARC + 2, 2)``.
    """
    mid_2d = (start_2d + end_2d) / 2.0                      # (n_bonds, 2)

    # Perpendicular to bond direction (rotated 90 degrees CCW).
    perp = np.column_stack([-by, bx])                        # (n_bonds, 2)

    # Interpolated half-width at the midpoint.
    mid_hw = (bb_a + bb_b) / 2.0                             # (n_bonds,)

    # Midpoint corners.
    mid_top = mid_2d + perp * mid_hw[:, None]                # (n_bonds, 2)
    mid_bot = mid_2d - perp * mid_hw[:, None]                # (n_bonds, 2)

    # Extract the two arcs from the full polygon.
    # full_verts[:, :n_arc] = start arc (+perp → -perp)
    # full_verts[:, n_arc:] = end arc (already reversed)
    arc_start = full_verts[:, :n_arc, :]                     # (n_bonds, n_arc, 2)
    arc_end = full_verts[:, n_arc:, :]                       # (n_bonds, n_arc, 2)

    # Half A: arc at atom a + straight cut at midpoint.
    # Winding: arc (+perp → -perp), then mid_bot (-perp), mid_top (+perp).
    half_a = np.concatenate([
        arc_start,
        mid_bot[:, None, :],
        mid_top[:, None, :],
    ], axis=1)

    # Half B: straight cut at midpoint + arc at atom b.
    # Winding: mid_top (+perp), mid_bot (-perp), then end arc
    # (which winds -perp → +perp, closing back to mid_top).
    half_b = np.concatenate([
        mid_top[:, None, :],
        mid_bot[:, None, :],
        arc_end,
    ], axis=1)

    return half_a, half_b


# ---------------------------------------------------------------------------
# Scene drawing (shared by static and interactive renderers)
# ---------------------------------------------------------------------------

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


@dataclass
class _PrecomputedScene:
    """Frame-independent data cached between redraws.

    Built once by :func:`_precompute_scene` and reused across rotation /
    zoom changes in the interactive viewer.
    """

    coords: np.ndarray
    radii_3d: np.ndarray
    atom_colours: list[tuple[float, float, float]]
    bond_half_colours: list[tuple[float, float, float]]
    adjacency: dict[int, list[tuple[int, Bond]]]
    bond_ia: np.ndarray
    bond_ib: np.ndarray
    bond_radii: np.ndarray
    bond_index: dict[int, int]
    polyhedra: list
    style_hidden_atoms: set[int]
    style_hidden_bond_ids: set[int]
    hidden_atoms: set[int]
    hidden_bond_ids: set[int]
    poly_base_colours: list[tuple[float, float, float]]
    poly_alphas: list[float]
    poly_edge_colours: list[tuple[float, float, float]]
    poly_edge_widths: list[float]


def _precompute_scene(
    scene: StructureScene,
    frame_index: int,
    render_style: RenderStyle | None = None,
    *,
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
) -> _PrecomputedScene:
    """Pre-compute frame-independent data for repeated rendering.

    Returns a :class:`_PrecomputedScene` of radii, colours, bonds, and
    adjacency that stay constant across rotation / zoom changes.
    """
    frame = scene.frames[frame_index]
    coords = frame.coords
    n_atoms = len(scene.species)
    radii_3d = np.empty(n_atoms)

    for i in range(n_atoms):
        sp = scene.species[i]
        style = scene.atom_styles.get(sp)
        radii_3d[i] = style.radius if style is not None else 0.5

    atom_colours = resolve_atom_colours(
        scene.species, scene.atom_styles, scene.atom_data,
        colour_by=colour_by, cmap=cmap, colour_range=colour_range,
    )
    bond_half_colours = list(atom_colours)

    bonds = compute_bonds(scene.species, coords, scene.bond_specs)
    adjacency: dict[int, list[tuple[int, Bond]]] = defaultdict(list)
    for bond in bonds:
        adjacency[bond.index_a].append((bond.index_b, bond))
        adjacency[bond.index_b].append((bond.index_a, bond))

    # Stacked bond arrays for vectorised geometry computation.
    n_bonds = len(bonds)
    bond_ia = np.empty(n_bonds, dtype=int)
    bond_ib = np.empty(n_bonds, dtype=int)
    bond_radii = np.empty(n_bonds)
    bond_index: dict[int, int] = {}
    for i, bond in enumerate(bonds):
        bond_ia[i] = bond.index_a
        bond_ib[i] = bond.index_b
        bond_radii[i] = bond.spec.radius
        bond_index[id(bond)] = i

    # ---- Polyhedra ----
    polyhedra = compute_polyhedra(
        scene.species, coords, bonds, scene.polyhedra,
    )

    # Atoms hidden by AtomStyle.visible=False — always applied,
    # regardless of show_polyhedra.
    style_hidden_atoms: set[int] = set()
    style_hidden_bond_ids: set[int] = set()
    for i, sp in enumerate(scene.species):
        style = scene.atom_styles.get(sp)
        if style is not None and not style.visible:
            style_hidden_atoms.add(i)
    if style_hidden_atoms:
        for bond in bonds:
            if bond.index_a in style_hidden_atoms or bond.index_b in style_hidden_atoms:
                style_hidden_bond_ids.add(id(bond))

    # Atoms/bonds hidden by polyhedra options (hide_centre, hide_bonds,
    # hide_vertices) — only applied when show_polyhedra is True.
    hidden_atoms: set[int] = set()
    hidden_bond_ids: set[int] = set()
    # For hide_vertices: an atom is hidden only if *every* polyhedron
    # it participates in has hide_vertices=True AND it has no bonds
    # to atoms outside those polyhedra (e.g. Li-O bonds keep O visible
    # even when Zr-O polyhedra hide vertices).
    vertex_hide_candidates: set[int] = set()
    vertex_keep: set[int] = set()
    poly_centres: set[int] = set()
    poly_members: set[int] = set()  # All centres + vertices in any polyhedron.
    for poly in polyhedra:
        poly_centres.add(poly.centre_index)
        poly_members.add(poly.centre_index)
        poly_members.update(poly.neighbour_indices)
        if poly.spec.hide_centre:
            hidden_atoms.add(poly.centre_index)
        # Always hide centre-to-vertex bonds when a polyhedron is
        # drawn.  These bonds are entirely inside the convex hull and
        # cannot be depth-sorted correctly against the polyhedral
        # faces in a painter's algorithm.
        neighbour_set = set(poly.neighbour_indices)
        for kk, bond in adjacency.get(poly.centre_index, []):
            if poly.spec.hide_bonds or kk in neighbour_set:
                hidden_bond_ids.add(id(bond))
        for ni in poly.neighbour_indices:
            if poly.spec.hide_vertices:
                vertex_hide_candidates.add(ni)
            else:
                vertex_keep.add(ni)
    # A vertex with bonds to non-polyhedron atoms must stay visible.
    for vi in vertex_hide_candidates - vertex_keep:
        for neighbour_idx, _ in adjacency.get(vi, []):
            if neighbour_idx not in poly_centres:
                vertex_keep.add(vi)
                break
    hidden_atoms |= vertex_hide_candidates - vertex_keep

    # Resolve face base colours per polyhedron.
    poly_base_colours: list[tuple[float, float, float]] = []
    poly_alphas: list[float] = []
    poly_edge_colours: list[tuple[float, float, float]] = []
    poly_edge_widths: list[float] = []
    edge_width_override = (
        render_style.polyhedra_outline_width if render_style is not None
        else None
    )
    for poly in polyhedra:
        if poly.spec.colour is not None:
            base_rgb = normalise_colour(poly.spec.colour)
        else:
            # Inherit from centre atom's resolved colour, which
            # accounts for colour_by / cmap when active.
            base_rgb = atom_colours[poly.centre_index]
        poly_base_colours.append(base_rgb)
        poly_alphas.append(poly.spec.alpha)
        poly_edge_colours.append(normalise_colour(poly.spec.edge_colour))
        poly_edge_widths.append(
            edge_width_override if edge_width_override is not None
            else poly.spec.edge_width
        )

    return _PrecomputedScene(
        coords=coords,
        radii_3d=radii_3d,
        atom_colours=atom_colours,
        bond_half_colours=bond_half_colours,
        adjacency=adjacency,
        bond_ia=bond_ia,
        bond_ib=bond_ib,
        bond_radii=bond_radii,
        bond_index=bond_index,
        polyhedra=polyhedra,
        style_hidden_atoms=style_hidden_atoms,
        style_hidden_bond_ids=style_hidden_bond_ids,
        hidden_atoms=hidden_atoms,
        hidden_bond_ids=hidden_bond_ids,
        poly_base_colours=poly_base_colours,
        poly_alphas=poly_alphas,
        poly_edge_colours=poly_edge_colours,
        poly_edge_widths=poly_edge_widths,
    )


def _apply_slab_clip(
    slab_visible: np.ndarray,
    slab_clip_mode: SlabClipMode,
    polyhedra_list: list,
    adjacency: dict[int, list[tuple[int, Bond]]],
    show_polyhedra: bool,
) -> tuple[np.ndarray, set[int], set[int]]:
    """Apply polyhedra-aware slab-clip overrides.

    Returns:
        Updated *slab_visible* array (copied if modified),
        *poly_skip* (polyhedron indices to skip entirely), and
        *poly_clip_hidden_bonds* (bond ``id()`` values to hide).
    """
    poly_skip: set[int] = set()
    poly_clip_hidden_bonds: set[int] = set()
    if (not show_polyhedra or not polyhedra_list
            or slab_clip_mode == SlabClipMode.PER_FACE):
        return slab_visible, poly_skip, poly_clip_hidden_bonds

    slab_force_visible: set[int] = set()
    for pi, poly in enumerate(polyhedra_list):
        all_vertices = set(poly.neighbour_indices) | {poly.centre_index}
        if slab_clip_mode == SlabClipMode.CLIP_WHOLE:
            if not all(slab_visible[v] for v in all_vertices):
                poly_skip.add(pi)
                for kk, bond in adjacency.get(poly.centre_index, []):
                    if kk in poly.neighbour_indices:
                        poly_clip_hidden_bonds.add(id(bond))
        elif slab_clip_mode == SlabClipMode.INCLUDE_WHOLE:
            if slab_visible[poly.centre_index]:
                slab_force_visible.update(all_vertices)
            else:
                poly_skip.add(pi)
    if slab_clip_mode == SlabClipMode.INCLUDE_WHOLE and slab_force_visible:
        slab_visible = slab_visible.copy()
        for v in slab_force_visible:
            slab_visible[v] = True
    return slab_visible, poly_skip, poly_clip_hidden_bonds


def _collect_polyhedra_faces(
    precomputed: _PrecomputedScene,
    polyhedra_list: list,
    poly_skip: set[int],
    slab_visible: np.ndarray,
    show_polyhedra: bool,
    polyhedra_shading: float,
    rotated: np.ndarray,
    depth: np.ndarray,
    xy: np.ndarray,
    order: np.ndarray,
) -> tuple[
    dict[int, list[tuple[np.ndarray, tuple, tuple, float, float]]],
    dict[int, int],
]:
    """Build per-face draw data and assign each face to a depth slot.

    Each face is slotted at its mean vertex depth.

    Returns:
        Tuple of ``(face_by_depth_slot, vertex_max_face_slot)`` where
        *face_by_depth_slot* maps depth-slot index to a list of
        ``(verts_2d, face_rgba, edge_rgba, edge_width, face_depth)``
        tuples sorted back-to-front, and *vertex_max_face_slot* maps
        each vertex atom index to the highest (most front-facing)
        depth-slot containing one of its connected faces.
    """
    face_by_depth_slot: dict[
        int, list[tuple[np.ndarray, tuple, tuple, float, float]]
    ] = defaultdict(list)
    vertex_max_face_slot: dict[int, int] = {}
    if not show_polyhedra or not polyhedra_list:
        return face_by_depth_slot, vertex_max_face_slot

    poly_base_colours = precomputed.poly_base_colours
    poly_alphas = precomputed.poly_alphas
    poly_edge_colours = precomputed.poly_edge_colours
    poly_edge_widths = precomputed.poly_edge_widths

    atom_depths_sorted = depth[order]
    for pi, poly in enumerate(polyhedra_list):
        if pi in poly_skip:
            continue
        base_rgb = poly_base_colours[pi]
        alpha = poly_alphas[pi]
        edge_rgb = poly_edge_colours[pi]
        edge_w = poly_edge_widths[pi]
        for face_row in poly.faces:
            global_idx = [poly.neighbour_indices[j] for j in face_row]

            # Slab check: all vertices must be visible.
            # (In include_whole mode, slab_visible has already been
            # updated to force polyhedron vertices visible.)
            if not all(slab_visible[gi] for gi in global_idx):
                continue

            # Face normal from first two edges (works for any polygon).
            face_verts = rotated[global_idx]
            normal = np.cross(
                face_verts[1] - face_verts[0],
                face_verts[2] - face_verts[0],
            )
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                cos_angle = abs(normal[2] / norm_len)
            else:
                cos_angle = 0.0
            shading = 1.0 - polyhedra_shading * 0.6 * (1.0 - cos_angle)
            shaded = tuple(min(1.0, c * shading) for c in base_rgb)

            face_depth = np.mean(depth[global_idx])
            verts_2d = xy[global_idx]

            slot = int(np.searchsorted(atom_depths_sorted, face_depth))

            face_by_depth_slot[slot].append((
                verts_2d,
                (*shaded, alpha),
                (*edge_rgb, 1.0),
                edge_w,
                float(face_depth),
            ))

            # Track the latest (most front-facing) slot for each vertex.
            for gi in global_idx:
                prev = vertex_max_face_slot.get(gi, -1)
                if slot > prev:
                    vertex_max_face_slot[gi] = slot

    # Sort faces within each slot back-to-front (ascending depth).
    for slot in face_by_depth_slot:
        face_by_depth_slot[slot].sort(key=lambda entry: entry[4])

    return face_by_depth_slot, vertex_max_face_slot


# ---------------------------------------------------------------------------
# Unit cell edge geometry
# ---------------------------------------------------------------------------

# The 12 edges of a unit cube, as pairs of vertex indices.
# Vertices are the 8 corners at fractional coordinates {0,1}^3.
_CUBE_EDGES: list[tuple[int, int]] = []
for _i in range(8):
    for _bit in range(3):
        _j = _i ^ (1 << _bit)  # flip one bit
        if _j > _i:
            _CUBE_EDGES.append((_i, _j))

# Fractional coordinates of the 8 cube corners (row-order matches
# the bit-pattern vertex indexing: 0→(0,0,0), 1→(1,0,0), …, 7→(1,1,1)).
_FRAC_CORNERS = np.array([
    [(v >> 0) & 1, (v >> 1) & 1, (v >> 2) & 1]
    for v in range(8)
], dtype=float)


def _cell_edges_3d(
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 12 unit cell edges from a 3x3 lattice matrix.

    Args:
        lattice: ``(3, 3)`` lattice matrix (rows are lattice vectors).

    Returns:
        Tuple of ``(starts, ends)`` each of shape ``(12, 3)``
        in Cartesian coordinates.
    """
    corners = _FRAC_CORNERS @ lattice  # (8, 3)
    starts = np.array([corners[i] for i, _ in _CUBE_EDGES])
    ends = np.array([corners[j] for _, j in _CUBE_EDGES])
    return starts, ends


# Dash patterns as repeating (mark, gap) ratios relative to ``pad``.
_DASH_PATTERNS: dict[str, list[float]] = {
    "dashed": [0.03, 0.015],
    "dotted": [0.005, 0.01],
    "dashdot": [0.03, 0.01, 0.005, 0.01],
}


def _split_dashes(
    start_2d: np.ndarray,
    end_2d: np.ndarray,
    pattern: list[float],
    pad: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a 2D segment into dash (mark) sub-segments.

    Args:
        start_2d: Start point ``(2,)``.
        end_2d: End point ``(2,)``.
        pattern: Alternating mark/gap lengths as fractions of *pad*.
        pad: Viewport half-extent (data units).

    Returns:
        List of ``(dash_start, dash_end)`` pairs for drawn segments.
    """
    direction = end_2d - start_2d
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return []
    unit = direction / length

    # Convert pattern from pad-relative fractions to data units.
    lengths = [r * pad for r in pattern]

    dashes: list[tuple[np.ndarray, np.ndarray]] = []
    pos = 0.0
    idx = 0
    drawing = True  # First entry is always a mark.
    while pos < length:
        seg_len = min(lengths[idx % len(lengths)], length - pos)
        if drawing:
            dashes.append((
                start_2d + unit * pos,
                start_2d + unit * (pos + seg_len),
            ))
        pos += seg_len
        idx += 1
        drawing = not drawing

    return dashes


def _clip_edge_at_atoms(
    start_3d: np.ndarray,
    end_3d: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a cell edge into visible segments that avoid atom spheres.

    Each atom whose sphere intersects the edge line produces a gap.
    The returned segments are the visible portions between (and
    outside) those gaps.

    Args:
        start_3d: Edge start point ``(3,)``.
        end_3d: Edge end point ``(3,)``.
        coords: Atom positions ``(n_atoms, 3)``.
        radii: Atom radii ``(n_atoms,)`` (already scaled).

    Returns:
        List of ``(seg_start, seg_end)`` pairs in 3D for the visible
        portions of the edge, or an empty list if fully occluded.
    """
    edge_vec = end_3d - start_3d
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return []
    edge_unit = edge_vec / edge_len

    # Project atom centres onto the edge axis.  t=0 is start,
    # t=edge_len is end.
    rel = coords - start_3d  # (n_atoms, 3)
    t = rel @ edge_unit  # projection along edge axis
    # Perpendicular distance squared from atom centre to edge line.
    perp_sq = np.sum(rel**2, axis=1) - t**2
    perp_sq = np.maximum(perp_sq, 0.0)

    # An atom's sphere intersects the edge line when r^2 > perp_sq.
    # The intersection interval along the axis is
    # [t - half_chord, t + half_chord].
    inside = radii**2 - perp_sq
    can_clip = inside > 0

    if not np.any(can_clip):
        return [(start_3d, end_3d)]

    half_chord = np.sqrt(inside[can_clip])
    t_vals = t[can_clip]

    # Collect all gap intervals [enter, exit] along the edge axis,
    # clamped to [0, edge_len].
    enters = np.maximum(t_vals - half_chord, 0.0)
    exits = np.minimum(t_vals + half_chord, edge_len)

    # Discard gaps that are entirely outside the edge.
    valid = exits > enters
    enters = enters[valid]
    exits = exits[valid]

    if len(enters) == 0:
        return [(start_3d, end_3d)]

    # Sort gaps by their entry point.
    sort_idx = np.argsort(enters)
    enters = enters[sort_idx]
    exits = exits[sort_idx]

    # Merge overlapping gaps into a single sorted list of
    # non-overlapping intervals.
    merged_enters: list[float] = [float(enters[0])]
    merged_exits: list[float] = [float(exits[0])]
    for i in range(1, len(enters)):
        if enters[i] <= merged_exits[-1]:
            # Overlapping or adjacent — extend the current gap.
            merged_exits[-1] = max(merged_exits[-1], float(exits[i]))
        else:
            merged_enters.append(float(enters[i]))
            merged_exits.append(float(exits[i]))

    # Build visible segments from the gaps in [0, edge_len].
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0.0
    for gap_in, gap_out in zip(merged_enters, merged_exits):
        if gap_in > cursor:
            segments.append((
                start_3d + edge_unit * cursor,
                start_3d + edge_unit * gap_in,
            ))
        cursor = gap_out

    if cursor < edge_len:
        segments.append((
            start_3d + edge_unit * cursor,
            end_3d,
        ))

    return segments


def _collect_cell_edges(
    scene: StructureScene,
    view: ViewState,
    cell_style: CellEdgeStyle,
    depth: np.ndarray,
    order: np.ndarray,
    pad: float,
    coords: np.ndarray,
    radii_3d: np.ndarray,
) -> dict[int, list[tuple[np.ndarray, tuple[float, ...], float]]]:
    """Compute cell edge polygons and assign them to depth slots.

    Each edge is clipped at any overlapping atom spheres (so edges
    stop at sphere surfaces, like bonds), then projected to 2D and
    expanded into a thin rectangle polygon for inclusion in the
    ``PolyCollection``.  For non-solid linestyles, each edge is split
    into dash segments.

    Args:
        scene: The scene (must have a non-None ``lattice``).
        view: Camera / projection state.
        cell_style: Visual style for the edges.
        depth: Per-atom depth array from projection.
        order: Back-to-front atom sort indices.
        pad: Viewport half-extent in data units.
        coords: Atom positions ``(n_atoms, 3)`` in world space.
        radii_3d: Atom radii ``(n_atoms,)`` already scaled by
            ``atom_scale``.

    Returns:
        Mapping from depth-slot index to list of
        ``(polygon_verts, rgba, edge_depth)`` tuples.
    """
    cell_edge_by_depth_slot: dict[
        int, list[tuple[np.ndarray, tuple[float, ...], float]]
    ] = defaultdict(list)

    if scene.lattice is None:
        return cell_edge_by_depth_slot

    starts_3d, ends_3d = _cell_edges_3d(scene.lattice)
    colour_rgb = normalise_colour(cell_style.colour)
    rgba = (*colour_rgb, 1.0)

    # Half-width in data units (scales with viewport).
    hw = cell_style.line_width * pad / 400.0

    # Slab clipping: check each endpoint against slab bounds.
    if view.slab_near is not None or view.slab_far is not None:
        all_pts = np.vstack([starts_3d, ends_3d])  # (24, 3)
        slab_mask = view.slab_mask(all_pts)
        slab_s = slab_mask[:12]
        slab_e = slab_mask[12:]
    else:
        slab_s = np.ones(12, dtype=bool)
        slab_e = np.ones(12, dtype=bool)

    atom_depths_sorted = depth[order]

    # Determine dash pattern.
    linestyle = cell_style.linestyle
    dash_pattern = _DASH_PATTERNS.get(linestyle)

    for ei in range(12):
        if not (slab_s[ei] and slab_e[ei]):
            continue

        # Clip edge at atom spheres in 3D, producing visible segments.
        visible_segs = _clip_edge_at_atoms(
            starts_3d[ei], ends_3d[ei], coords, radii_3d,
        )

        for seg_start_3d, seg_end_3d in visible_segs:
            # Rotate segment endpoints into camera space.
            c_s = (seg_start_3d - view.centre) @ view.rotation.T
            c_e = (seg_end_3d - view.centre) @ view.rotation.T
            d_s = float(c_s[2])
            d_e = float(c_e[2])

            # Split the 3D segment at atom depth boundaries so each
            # sub-piece gets its own correct depth slot.  Without
            # this, a long edge spanning many depth layers would be
            # assigned a single mid-depth slot and occlude incorrectly.
            lo_d, hi_d = min(d_s, d_e), max(d_s, d_e)
            # Atom depths that fall strictly inside the segment range.
            inside = (atom_depths_sorted > lo_d) & (
                atom_depths_sorted < hi_d
            )
            cut_depths = atom_depths_sorted[inside]

            if len(cut_depths) == 0:
                # No atom depths inside this segment — single piece.
                sub_fracs = [(0.0, 1.0)]
            else:
                # Convert each cut depth to a fraction along the
                # segment (0 = start, 1 = end).
                depth_span = d_e - d_s
                if abs(depth_span) < 1e-12:
                    sub_fracs = [(0.0, 1.0)]
                else:
                    ts = np.sort((cut_depths - d_s) / depth_span)
                    ts = np.clip(ts, 0.0, 1.0)
                    boundaries = np.concatenate(
                        [[0.0], ts, [1.0]],
                    )
                    sub_fracs = [
                        (float(boundaries[i]),
                         float(boundaries[i + 1]))
                        for i in range(len(boundaries) - 1)
                        if boundaries[i + 1] - boundaries[i] > 1e-12
                    ]

            for t0, t1 in sub_fracs:
                sub_c_s = c_s + (c_e - c_s) * t0
                sub_c_e = c_s + (c_e - c_s) * t1
                sub_d = (sub_c_s[2] + sub_c_e[2]) / 2.0

                if view.perspective > 0:
                    s_s = view.view_distance / (
                        view.view_distance
                        - sub_c_s[2] * view.perspective
                    )
                    s_e = view.view_distance / (
                        view.view_distance
                        - sub_c_e[2] * view.perspective
                    )
                    xy_s_i = sub_c_s[:2] * s_s * view.zoom
                    xy_e_i = sub_c_e[:2] * s_e * view.zoom
                else:
                    xy_s_i = sub_c_s[:2] * view.zoom
                    xy_e_i = sub_c_e[:2] * view.zoom

                if dash_pattern is not None:
                    dash_segs = _split_dashes(
                        xy_s_i, xy_e_i, dash_pattern, pad,
                    )
                else:
                    dash_segs = [(xy_s_i, xy_e_i)]

                for seg_start, seg_end in dash_segs:
                    direction = seg_end - seg_start
                    seg_len = np.linalg.norm(direction)
                    if seg_len < 1e-12:
                        continue
                    perp = np.array(
                        [-direction[1], direction[0]],
                    ) / seg_len

                    offset = perp * hw
                    verts = np.array([
                        seg_start + offset,
                        seg_end + offset,
                        seg_end - offset,
                        seg_start - offset,
                    ])

                    slot = int(np.searchsorted(
                        atom_depths_sorted, sub_d,
                    ))
                    cell_edge_by_depth_slot[slot].append(
                        (verts, rgba, float(sub_d)),
                    )

    # Sort within each slot back-to-front.
    for slot in cell_edge_by_depth_slot:
        cell_edge_by_depth_slot[slot].sort(key=lambda entry: entry[2])

    return cell_edge_by_depth_slot


def _draw_scene(
    ax,
    scene: StructureScene,
    view: ViewState,
    style: RenderStyle,
    *,
    frame_index: int = 0,
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    viewport_extent: float | None = None,
    precomputed: _PrecomputedScene | None = None,
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
) -> None:
    """Paint atoms and bonds onto *ax* using the painter's algorithm.

    Clears *ax* and redraws the full scene.  Does **not** create or
    show the figure — the caller owns the figure lifecycle.

    Args:
        ax: A matplotlib ``Axes`` to draw into.
        scene: The structure to render.
        view: Camera / projection state (may differ from ``scene.view``
            in interactive mode).
        style: Visual style settings.
        frame_index: Which frame to render.
        bg_rgb: Normalised background colour.
        viewport_extent: If given, use this as the fixed half-extent
            for axis limits instead of computing from projected coords.
            Avoids the "view distance changing" artefact during
            interactive rotation.
        precomputed: Pre-computed scene data from
            :func:`_precompute_scene`.  If ``None``, computed on the
            fly.
        colour_by: Key into ``scene.atom_data`` for colourmap-based
            colouring.
        cmap: Matplotlib colourmap name, object, or callable.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
    """
    atom_scale = style.atom_scale
    bond_scale = style.bond_scale
    bond_colour = style.bond_colour
    half_bonds = style.half_bonds
    show_bonds = style.show_bonds
    show_polyhedra = style.show_polyhedra
    show_outlines = style.show_outlines
    outline_rgb = normalise_colour(style.outline_colour)
    atom_outline_width = style.atom_outline_width
    bond_outline_width = style.bond_outline_width
    slab_clip_mode = style.slab_clip_mode
    unit_circle = _make_unit_circle(style.circle_segments)
    arc = _make_arc(style.arc_segments)
    n_arc = len(arc)
    polyhedra_shading = style.polyhedra_shading
    # Remove previous draw's collection(s) and leftover artists.
    while ax.collections:
        ax.collections[0].remove()
    for t in ax.texts[:]:
        t.remove()
    for line in ax.lines[:]:
        line.remove()

    if precomputed is None:
        precomputed = _precompute_scene(
            scene, frame_index, style,
            colour_by=colour_by, cmap=cmap, colour_range=colour_range,
        )

    coords = precomputed.coords
    radii_3d = precomputed.radii_3d
    atom_colours = precomputed.atom_colours
    bond_half_colours = precomputed.bond_half_colours
    adjacency = precomputed.adjacency

    # ---- Projection ----
    xy, depth, atom_screen_radii = view.project(
        coords, radii_3d * atom_scale,
    )
    rotated = (coords - view.centre) @ view.rotation.T

    # ---- Sort atoms back-to-front (furthest first) ----
    order = np.argsort(depth)

    # ---- Slab visibility ----
    slab_visible = view.slab_mask(coords)

    # ---- Slab-clip overrides for polyhedra ----
    polyhedra_list = precomputed.polyhedra
    slab_visible, poly_skip, poly_clip_hidden_bonds = _apply_slab_clip(
        slab_visible, slab_clip_mode, polyhedra_list, adjacency,
        show_polyhedra,
    )

    ax.set_facecolor(bg_rgb)

    # ---- Batch bond geometry ----
    bond_ia = precomputed.bond_ia
    bond_ib = precomputed.bond_ib
    bond_radii_arr = precomputed.bond_radii
    bond_index = precomputed.bond_index

    use_half = half_bonds and bond_colour is None
    batch_valid = np.empty(0, dtype=bool)
    batch_full_verts = np.empty((0, 2 * n_arc, 2))
    batch_half_a = np.empty((0, n_arc + 2, 2))
    batch_half_b = np.empty((0, n_arc + 2, 2))

    if show_bonds and len(bond_ia) > 0:
        (batch_full_verts, batch_start_2d, batch_end_2d,
         batch_bb_a, batch_bb_b, batch_bx, batch_by,
         batch_valid) = _bond_polygons_batch(
            rotated, xy,
            radii_3d * atom_scale, atom_screen_radii,
            bond_ia, bond_ib, bond_radii_arr * bond_scale,
            view, arc,
        )
        if use_half:
            batch_half_a, batch_half_b = _half_bond_verts_batch(
                batch_full_verts, batch_start_2d, batch_end_2d,
                batch_bb_a, batch_bb_b, batch_bx, batch_by, n_arc,
            )

    # Pre-compute bond spec colours for the non-half-bond path.
    if not use_half:
        bond_spec_colours: dict[int, tuple[float, float, float]] = {}

    # ---- Polyhedra face data ----
    # AtomStyle.visible=False hiding is always applied.  Polyhedra-
    # driven hiding (hide_centre, hide_bonds, hide_vertices) is only
    # applied when polyhedra are actually being drawn.
    hidden_atoms = set(precomputed.style_hidden_atoms)
    hidden_bond_ids = set(precomputed.style_hidden_bond_ids)
    if show_polyhedra:
        hidden_atoms |= precomputed.hidden_atoms
        hidden_bond_ids |= precomputed.hidden_bond_ids

    face_by_depth_slot, vertex_max_face_slot = _collect_polyhedra_faces(
        precomputed, polyhedra_list, poly_skip, slab_visible,
        show_polyhedra, polyhedra_shading, rotated, depth, xy, order,
    )

    # Defer vertex atoms so they paint after all their connected
    # polyhedral faces.  Each vertex draws right after the latest
    # depth-slot containing one of its faces.
    deferred_vertex_atoms: set[int] = set()
    in_front_after_slot: dict[int, list[int]] = defaultdict(list)
    if show_polyhedra and vertex_max_face_slot:
        atom_depths_sorted = depth[order]
        for vi, max_slot in vertex_max_face_slot.items():
            if vi in hidden_atoms:
                continue
            vi_slot = int(np.searchsorted(atom_depths_sorted, depth[vi]))
            if vi_slot <= max_slot:
                deferred_vertex_atoms.add(vi)
                in_front_after_slot[max_slot].append(vi)
    # Sort deferred vertices within each slot back-to-front.
    for slot in in_front_after_slot:
        in_front_after_slot[slot].sort(key=lambda v: depth[v])

    # ---- Cell edge data ----
    draw_cell = style.show_cell
    if draw_cell is None:
        draw_cell = scene.lattice is not None
    if draw_cell and scene.lattice is None:
        raise ValueError("show_cell=True but scene has no lattice")

    cell_edge_by_depth_slot: dict[
        int, list[tuple[np.ndarray, tuple[float, ...], float]]
    ] = {}
    if draw_cell:
        # Compute pad for line-width scaling (max of x/y half-extents).
        if viewport_extent is not None:
            cell_pad = viewport_extent * 1.15
        elif len(xy) == 0:
            cell_pad = 1.0
        else:
            cell_margin = np.max(atom_screen_radii) + 1.0
            cell_pad = max(
                (xy[:, 0].max() - xy[:, 0].min()) / 2,
                (xy[:, 1].max() - xy[:, 1].min()) / 2,
            ) + cell_margin
        cell_edge_by_depth_slot = _collect_cell_edges(
            scene, view, style.cell_style, depth, order, cell_pad,
            coords, radii_3d * atom_scale,
        )

    # Collect raw vertex arrays in painter's order, then batch-add
    # via PolyCollection (avoids costly Patch object creation).
    all_verts: list[np.ndarray] = []
    face_colours: list[tuple[float, ...]] = []
    edge_colours: list[tuple[float, ...]] = []
    line_widths: list[float] = []

    drawn_bonds: set[int] = set()

    # ---- Paint back-to-front ----
    for order_pos, k in enumerate(order):
        # Draw cell edges at this depth slot (behind bonds and atoms).
        for edge_verts, edge_rgba, _ in (
            cell_edge_by_depth_slot.get(order_pos, [])
        ):
            all_verts.append(edge_verts)
            face_colours.append(edge_rgba)
            edge_colours.append(edge_rgba)
            line_widths.append(0.0)

        if slab_visible[k]:
            neighbours = adjacency.get(k, []) if show_bonds else []
            neighbours_sorted = sorted(
                neighbours, key=lambda nb: depth[nb[0]],
            )

            for kk, bond in neighbours_sorted:
                bond_id = id(bond)
                if bond_id in drawn_bonds:
                    continue
                if bond_id in hidden_bond_ids:
                    drawn_bonds.add(bond_id)
                    continue
                if bond_id in poly_clip_hidden_bonds:
                    drawn_bonds.add(bond_id)
                    continue

                if depth[k] < depth[kk]:
                    continue
                if depth[k] == depth[kk] and k > kk:
                    continue

                drawn_bonds.add(bond_id)

                # Skip bonds where either atom is outside the slab.
                ia, ib = bond.index_a, bond.index_b
                if not slab_visible[ia] or not slab_visible[ib]:
                    continue

                bi = bond_index[bond_id]
                if not batch_valid[bi]:
                    continue

                if use_half and bond_half_colours[ia] != bond_half_colours[ib]:
                    fc_a = (*bond_half_colours[ia], 1.0)
                    all_verts.append(batch_half_a[bi])
                    face_colours.append(fc_a)
                    edge_colours.append(
                        (*outline_rgb, 1.0) if show_outlines else fc_a,
                    )
                    line_widths.append(
                        bond_outline_width if show_outlines else 0.0,
                    )
                    fc_b = (*bond_half_colours[ib], 1.0)
                    all_verts.append(batch_half_b[bi])
                    face_colours.append(fc_b)
                    edge_colours.append(
                        (*outline_rgb, 1.0) if show_outlines else fc_b,
                    )
                    line_widths.append(
                        bond_outline_width if show_outlines else 0.0,
                    )
                elif use_half:
                    # Same colour on both halves — draw as single polygon
                    # to avoid a spurious outline at the midpoint.
                    fc = (*bond_half_colours[ia], 1.0)
                    all_verts.append(batch_full_verts[bi])
                    face_colours.append(fc)
                    edge_colours.append(
                        (*outline_rgb, 1.0) if show_outlines else fc,
                    )
                    line_widths.append(
                        bond_outline_width if show_outlines else 0.0,
                    )
                else:
                    spec_id = id(bond.spec)
                    if spec_id not in bond_spec_colours:
                        if bond_colour is not None:
                            brgb = normalise_colour(bond_colour)
                        else:
                            brgb = normalise_colour(bond.spec.colour)
                        bond_spec_colours[spec_id] = brgb
                    brgb = bond_spec_colours[spec_id]

                    all_verts.append(batch_full_verts[bi])
                    face_colours.append((*brgb, 1.0))
                    edge_colours.append(
                        (*outline_rgb, 1.0)
                        if show_outlines
                        else (*brgb, 1.0),
                    )
                    line_widths.append(
                        bond_outline_width if show_outlines else 0.0,
                    )

        # Draw polyhedron faces assigned to this depth slot.
        # Faces are drawn regardless of whether the atom at this slot is
        # slab-visible — a face's slot is determined by its mean depth,
        # which may coincide with a clipped atom.
        for face_verts, face_fc, face_ec, face_lw, _ in (
            face_by_depth_slot.get(order_pos, [])
        ):
            all_verts.append(face_verts)
            face_colours.append(face_fc)
            edge_colours.append(face_ec if show_outlines else face_fc)
            line_widths.append(face_lw if show_outlines else 0.0)

        # IN_FRONT: draw vertex atoms whose last connected face was
        # in this slot.  They paint on top of all their faces but
        # still behind any faces in later (more front-facing) slots.
        for vi in in_front_after_slot.get(order_pos, []):
            if not slab_visible[vi]:
                continue
            fc_atom = (*atom_colours[vi], 1.0)
            all_verts.append(unit_circle * atom_screen_radii[vi] + xy[vi])
            face_colours.append(fc_atom)
            edge_colours.append(
                (*outline_rgb, 1.0) if show_outlines else fc_atom,
            )
            line_widths.append(
                atom_outline_width if show_outlines else 0.0,
            )

        if not slab_visible[k]:
            continue

        # Draw atom circle (unless hidden by a polyhedron spec or
        # deferred for polyhedron vertex ordering).
        if (k not in hidden_atoms
                and k not in deferred_vertex_atoms):
            fc_atom = (*atom_colours[k], 1.0)
            all_verts.append(unit_circle * atom_screen_radii[k] + xy[k])
            face_colours.append(fc_atom)
            edge_colours.append((*outline_rgb, 1.0) if show_outlines else fc_atom)
            line_widths.append(atom_outline_width if show_outlines else 0.0)

    # Draw cell edges in front of all atoms.
    for edge_verts, edge_rgba, _ in (
        cell_edge_by_depth_slot.get(len(order), [])
    ):
        all_verts.append(edge_verts)
        face_colours.append(edge_rgba)
        edge_colours.append(edge_rgba)
        line_widths.append(0.0)

    # Draw any faces whose mean depth is in front of all atoms.
    for face_verts, face_fc, face_ec, face_lw, _ in (
        face_by_depth_slot.get(len(order), [])
    ):
        all_verts.append(face_verts)
        face_colours.append(face_fc)
        edge_colours.append(face_ec if show_outlines else face_fc)
        line_widths.append(face_lw if show_outlines else 0.0)

    # IN_FRONT vertices whose last face slot is len(order).
    for vi in in_front_after_slot.get(len(order), []):
        if not slab_visible[vi]:
            continue
        fc_atom = (*atom_colours[vi], 1.0)
        all_verts.append(unit_circle * atom_screen_radii[vi] + xy[vi])
        face_colours.append(fc_atom)
        edge_colours.append(
            (*outline_rgb, 1.0) if show_outlines else fc_atom,
        )
        line_widths.append(
            atom_outline_width if show_outlines else 0.0,
        )

    if all_verts:
        pc = PolyCollection(
            all_verts,
            closed=True,
            facecolors=face_colours,
            edgecolors=edge_colours,
            linewidths=line_widths,
        )
        ax.add_collection(pc)

    # ---- Axes and layout ----
    ax.set_aspect("equal")
    if viewport_extent is not None:
        pad_x = pad_y = viewport_extent * 1.15
        cx = cy = 0.0
    elif len(xy) == 0:
        pad_x = pad_y = 1.0
        cx = cy = 0.0
    else:
        margin = np.max(atom_screen_radii) + 1.0
        cx = (xy[:, 0].max() + xy[:, 0].min()) / 2
        cy = (xy[:, 1].max() + xy[:, 1].min()) / 2
        pad_x = (xy[:, 0].max() - xy[:, 0].min()) / 2 + margin
        pad_y = (xy[:, 1].max() - xy[:, 1].min()) / 2 + margin
    # ---- Axes orientation widget ----
    draw_axes = style.show_axes
    if draw_axes is None:
        draw_axes = scene.lattice is not None
    if draw_axes and viewport_extent is None:
        # Expand viewport so the widget doesn't overlap atoms.
        # The widget spans (margin + 2 * arrow_length) * pad from the
        # corner; halve this because expansion is applied to both sides.
        axes_style = style.axes_style
        widget_frac = axes_style.margin + 2.0 * axes_style.arrow_length
        expand_per_side = widget_frac * 0.5
        pad_x *= 1.0 + expand_per_side
        pad_y *= 1.0 + expand_per_side
    ax.set_xlim(cx - pad_x, cx + pad_x)
    ax.set_ylim(cy - pad_y, cy + pad_y)
    ax.axis("off")

    if scene.title:
        ax.text(
            0.5, 0.97, scene.title,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=_TITLE_FONT_SIZE,
            path_effects=[
                path_effects.withStroke(
                    linewidth=_TITLE_FONT_SIZE * 0.25, foreground="white",
                ),
            ],
        )

    if draw_axes:
        if scene.lattice is None:
            raise ValueError("show_axes=True but scene has no lattice")
        _draw_axes_widget(
            ax, scene, view, style.axes_style,
            pad_x=pad_x, pad_y=pad_y, cx=cx, cy=cy,
        )


# ---------------------------------------------------------------------------
# Axes orientation widget
# ---------------------------------------------------------------------------


def _draw_axes_widget(
    ax: "Axes",
    scene: StructureScene,
    view: ViewState,
    style: AxesStyle,
    pad_x: float,
    pad_y: float,
    cx: float = 0.0,
    cy: float = 0.0,
) -> None:
    """Draw a crystallographic axes orientation widget on *ax*.

    Three lines representing the a, b, c lattice directions are drawn
    from a common origin in the specified corner of the viewport.  The
    lines rotate in sync with the structure via ``view.rotation``,
    with italic labels at the tips.

    This function adds ``Line2D`` artists and text labels.  These are
    cleaned up on the next call to :func:`_draw_scene` via the
    ``ax.lines[:]`` and ``ax.texts[:]`` removal.

    Args:
        ax: A matplotlib ``Axes`` to draw into.
        scene: The structure scene (provides lattice).
        view: Camera / projection state.
        style: Visual style for the widget.
        pad_x: Viewport half-extent in the x direction (data coords).
        pad_y: Viewport half-extent in the y direction (data coords).
        cx: Viewport centre x coordinate.
        cy: Viewport centre y coordinate.
    """
    lattice = scene.lattice  # (3, 3), rows are a, b, c
    assert lattice is not None, "axes widget requires a lattice"
    pad = max(pad_x, pad_y)
    arrow_len = style.arrow_length * pad

    # Widget origin in data coordinates.
    if isinstance(style.corner, tuple):
        # Explicit fractional position: (0, 0) = bottom-left, (1, 1) = top-right.
        fx, fy = style.corner
        ox = (cx - pad_x) + 2 * pad_x * fx
        oy = (cy - pad_y) + 2 * pad_y * fy
    else:
        # Named corner with margin inset.
        inset_x = style.margin * pad_x + arrow_len
        inset_y = style.margin * pad_y + arrow_len
        if style.corner in (WidgetCorner.BOTTOM_LEFT, WidgetCorner.TOP_LEFT):
            ox = (cx - pad_x) + inset_x
        else:
            ox = (cx + pad_x) - inset_x
        if style.corner in (WidgetCorner.BOTTOM_LEFT, WidgetCorner.BOTTOM_RIGHT):
            oy = (cy - pad_y) + inset_y
        else:
            oy = (cy + pad_y) - inset_y

    # Compute display-space scaling so that font_size and line_width
    # stay proportional to arrow_len regardless of axes size.  The
    # style defaults are calibrated for a 4-inch-wide figure with a
    # single subplot (axes width ~3.1 inches); scale them by comparing
    # the actual arrow length in points against a reference value.
    fig = ax.get_figure()
    if not isinstance(fig, Figure):
        raise ValueError("ax is not attached to a Figure")
    ax_width_in = fig.get_figwidth() * ax.get_position().width
    pts_per_data = ax_width_in * 72.0 / (2.0 * pad_x)
    arrow_len_pts = arrow_len * pts_per_data
    # Reference: 0.12 * 72 / 2 * 3.1 = ~13.3 pts (single subplot in 4-inch fig)
    _REFERENCE_ARROW_PTS = 0.12 * 72.0 / 2.0 * 3.1
    scale = arrow_len_pts / _REFERENCE_ARROW_PTS
    font_size = style.font_size * scale
    line_width = style.line_width * scale
    stroke_width = 3.0 * scale

    # Normalise lattice vectors to unit length.
    directions = np.empty((3, 3))
    for i in range(3):
        v = lattice[i]
        norm = float(np.linalg.norm(v))
        directions[i] = v / norm if norm > 1e-12 else np.eye(3)[i]

    # Project through the view rotation (orthographic, no perspective).
    projected = directions @ view.rotation.T  # (3, 3)
    tips_2d = projected[:, :2] * arrow_len

    # Sort by z-depth (furthest first) so nearer lines overlap.
    draw_order = np.argsort(projected[:, 2])

    label_beyond = 1.3   # label position as multiple of line length
    min_label_r = arrow_len * 0.4   # minimum label distance from origin
    fontstyle = "italic" if style.italic else "normal"

    # Compute 2D tip lengths for foreshortening detection.
    tip_lengths = np.hypot(tips_2d[:, 0], tips_2d[:, 1])

    # Draw all lines first, then all labels on top so that the
    # white background halo on each label cleanly covers the line
    # tip without obscuring the line body.
    label_data: list[tuple[float, float, str, tuple[float, float, float]]] = []

    for i in draw_order:
        colour_rgb = normalise_colour(style.colours[i])
        dx, dy = tips_2d[i]

        ax.plot(
            [ox, ox + dx], [oy, oy + dy],
            color=colour_rgb,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=10,
        )

        # Place label beyond the line tip.  When the axis is heavily
        # foreshortened (nearly pointing at the viewer), push the
        # label away from the other two axes so it doesn't pile up
        # at the origin.
        tip_len = tip_lengths[i]
        desired_r = tip_len * label_beyond
        if desired_r >= min_label_r:
            lx = ox + dx * label_beyond
            ly = oy + dy * label_beyond
        else:
            # Foreshortened: place label opposite to the mean of the
            # other two axes' tips so it stays out of the way.
            others = [j for j in range(3) if j != i]
            mean_other = tips_2d[others].mean(axis=0)
            away = -mean_other
            away_len = np.hypot(away[0], away[1])
            if away_len > 1e-12:
                away = away / away_len * min_label_r
            else:
                # All three axes are near-zero (degenerate) — fall back.
                away = np.array([0.0, min_label_r])
            lx = ox + away[0]
            ly = oy + away[1]
        label_data.append((lx, ly, style.labels[i], colour_rgb))

    for lx, ly, label, colour_rgb in label_data:
        ax.text(
            lx, ly, label,
            fontsize=font_size,
            fontstyle=fontstyle,
            color=colour_rgb,
            ha="center",
            va="center",
            zorder=11,
            path_effects=[
                path_effects.withStroke(
                    linewidth=stroke_width, foreground="white",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# Static renderer
# ---------------------------------------------------------------------------

def _axes_bg_rgb(ax: Axes) -> tuple[float, float, float]:
    """Return the axes background as an (R, G, B) tuple."""
    from matplotlib.colors import to_rgb
    return to_rgb(ax.get_facecolor())


_STYLE_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(RenderStyle))
_DEFAULT_RENDER_STYLE = RenderStyle()

def _resolve_style(
    style: RenderStyle | None,
    **kwargs: Any,
) -> RenderStyle:
    """Build a :class:`RenderStyle` from an optional base plus overrides.

    Any kwarg whose name matches a ``RenderStyle`` field replaces that
    field's value.  Unknown kwargs raise :class:`TypeError`.

    Raises:
        TypeError: If a kwarg name does not match any ``RenderStyle`` field.
    """
    from dataclasses import replace

    unknown = kwargs.keys() - _STYLE_FIELDS
    if unknown:
        raise TypeError(
            f"Unknown style keyword argument(s): {', '.join(sorted(unknown))}"
        )

    s = style if style is not None else replace(_DEFAULT_RENDER_STYLE)
    overrides = {k: (v if v is not None else getattr(_DEFAULT_RENDER_STYLE, k))
                 for k, v in kwargs.items()}
    if overrides:
        s = replace(s, **overrides)
    return s


def render_mpl(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    ax: Axes | None = None,
    style: RenderStyle | None = None,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    show: bool | None = None,
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
    **style_kwargs: object,
) -> Figure:
    """Render a StructureScene as a static matplotlib figure.

    Uses a depth-sorted painter's algorithm: atoms are sorted
    back-to-front, and each atom's bonds are drawn just before the
    atom itself is painted.  Bond-sphere intersections are computed
    in 3D and then projected to screen space.

    Example usage::

        scene = StructureScene.from_xbs("ch4.bs")

        # Save to file (no interactive window):
        scene.render_mpl("ch4.png")

        # Publication-quality SVG with custom sizing:
        scene.render_mpl("ch4.svg", figsize=(8, 8), dpi=300,
                         background="black")

        # Interactive display (no file):
        scene.render_mpl()

        # Custom style with no outlines:
        from hofmann import RenderStyle
        style = RenderStyle(show_outlines=False, atom_scale=0.8)
        scene.render_mpl("clean.svg", style=style)

        # View along the [1, 1, 1] direction with a depth slab:
        scene.view.look_along([1, 1, 1])
        scene.view.slab_near = -2.0
        scene.view.slab_far = 2.0
        scene.render_mpl("slice.png")

        # Colour by per-atom metadata:
        scene.set_atom_data("charge", charges)
        scene.render_mpl(colour_by="charge", cmap="coolwarm")

        # Render into an existing axes for multi-panel figures:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(x, y)
        scene.render_mpl(ax=ax2)
        fig.savefig("panel.pdf", bbox_inches="tight")

    Args:
        scene: The StructureScene to render.
        output: Optional file path to save the figure.  The format is
            inferred from the extension (e.g. ``.svg``, ``.pdf``,
            ``.png``).  Ignored when *ax* is provided.
        ax: Optional matplotlib :class:`~matplotlib.axes.Axes` to draw
            into.  When provided, the scene is rendered onto this axes
            and the caller retains control of the parent figure
            (saving, display, layout).  The caller is responsible
            for saving and closing the figure.  The *output*,
            *figsize*, *dpi*, *background*, and *show* parameters
            are ignored.
        style: A :class:`RenderStyle` controlling visual appearance.
            If ``None``, defaults are used.  Any :class:`RenderStyle`
            field name may also be passed as a keyword argument to
            override individual fields (e.g. ``show_bonds=False``,
            ``half_bonds=False``).
        frame_index: Which frame to render (default 0).
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution for raster output formats.
        background: Background colour (CSS name, hex string, grey
            float, or RGB tuple).
        show: Whether to call ``plt.show()`` to open an interactive
            window.  Defaults to ``True`` when *output* is ``None``,
            ``False`` when saving to a file.  Pass explicitly to
            override (e.g. ``show=True`` to both save and display).
        colour_by: Key into ``scene.atom_data`` to colour atoms by.
            When ``None`` (the default), species-based colouring is
            used.
        cmap: Matplotlib colourmap name (e.g. ``"viridis"``),
            ``Colormap`` object, or callable mapping a float in
            ``[0, 1]`` to an ``(r, g, b)`` tuple.
        colour_range: Explicit ``(vmin, vmax)`` for normalising
            numerical data.  ``None`` auto-ranges from the data.
        **style_kwargs: Any :class:`RenderStyle` field name as a
            keyword argument.  Unknown names raise :class:`TypeError`.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    resolved = _resolve_style(style, **style_kwargs)

    n_frames = len(scene.frames)
    if not 0 <= frame_index < n_frames:
        raise ValueError(
            f"frame_index {frame_index} out of range for scene "
            f"with {n_frames} frame(s)"
        )

    if ax is not None:
        fig = ax.get_figure()
        if not isinstance(fig, Figure):
            raise ValueError("ax is not attached to a Figure")

        _draw_scene(
            ax, scene, scene.view, resolved,
            frame_index=frame_index,
            bg_rgb=_axes_bg_rgb(ax),
            colour_by=colour_by,
            cmap=cmap,
            colour_range=colour_range,
        )

        return fig

    bg_rgb = normalise_colour(background)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)

    _draw_scene(
        ax, scene, scene.view, resolved,
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        colour_by=colour_by,
        cmap=cmap,
        colour_range=colour_range,
    )

    fig.tight_layout()

    if output is not None:
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")

    if show is None:
        show = output is None

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix about the X axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ])


def _rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix about the Y axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  0.0,  s],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c],
    ])


def _rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix about the Z axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Interactive renderer — constants
# ---------------------------------------------------------------------------

_KEY_ROTATION_STEP = 0.05  # radians (~3 degrees) per key press
_KEY_ZOOM_FACTOR = 1.1  # multiplicative zoom per key press / scroll step
_KEY_PAN_FRACTION = 0.05  # fraction of scene extent per key press
_PERSPECTIVE_STEP = 0.1  # perspective increment per key press
_DISTANCE_FACTOR = 1.05  # viewing distance multiplier per key press

_HELP_TEXT = """\
Arrows     Rotate          Shift+Arrows  Pan
,  .       Roll            +  =  -       Zoom
p  P       Perspective     d  D          Distance
b          Bonds           o             Outlines
e          Polyhedra       u             Unit cell
a          Axes            r             Reset view
[  ]       Prev/next frame {  }          First/last
h          Toggle help     Scroll        Zoom
Drag       Rotate"""


def _apply_key_action(
    key: str,
    view: "ViewState",
    style: "RenderStyle",
    state: dict,
    *,
    n_frames: int,
    base_extent: float,
    initial_view: dict,
    has_lattice: bool = False,
) -> str:
    """Apply a keyboard action, mutating *view*, *style*, and *state*.

    Returns a string indicating the required redraw kind:

    - ``"view"`` — view-only change (rotation, zoom, pan, etc.).
    - ``"full"`` — style or frame change needing recomputation.
    - ``"none"`` — unrecognised key, no redraw needed.
    """
    # -- Rotation --
    if key == "left":
        view.rotation = _rotation_y(-_KEY_ROTATION_STEP) @ view.rotation
    elif key == "right":
        view.rotation = _rotation_y(_KEY_ROTATION_STEP) @ view.rotation
    elif key == "up":
        view.rotation = _rotation_x(-_KEY_ROTATION_STEP) @ view.rotation
    elif key == "down":
        view.rotation = _rotation_x(_KEY_ROTATION_STEP) @ view.rotation
    elif key == ",":
        view.rotation = _rotation_z(_KEY_ROTATION_STEP) @ view.rotation
    elif key == ".":
        view.rotation = _rotation_z(-_KEY_ROTATION_STEP) @ view.rotation

    # -- Zoom --
    elif key in ("+", "="):
        view.zoom = min(100.0, view.zoom * _KEY_ZOOM_FACTOR)
    elif key == "-":
        view.zoom = max(0.01, view.zoom / _KEY_ZOOM_FACTOR)

    # -- Pan (shift + arrows) --
    # Moving the centre in screen-right shifts the *camera* right,
    # so the scene appears to move left.  Negate so that the arrow
    # direction matches the apparent scene movement.
    elif key == "shift+left":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre + step * view.rotation[0]
    elif key == "shift+right":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre - step * view.rotation[0]
    elif key == "shift+down":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre + step * view.rotation[1]
    elif key == "shift+up":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre - step * view.rotation[1]

    # -- Perspective / distance --
    elif key == "p":
        view.perspective = min(1.0, view.perspective + _PERSPECTIVE_STEP)
    elif key == "P":
        view.perspective = max(0.0, view.perspective - _PERSPECTIVE_STEP)
    elif key == "d":
        view.view_distance *= _DISTANCE_FACTOR
    elif key == "D":
        view.view_distance = max(0.1, view.view_distance / _DISTANCE_FACTOR)

    # -- Style toggles (no recomputation needed) --
    elif key == "b":
        style.show_bonds = not style.show_bonds
    elif key == "o":
        style.show_outlines = not style.show_outlines
    elif key == "e":
        style.show_polyhedra = not style.show_polyhedra
    elif key == "u":
        # Resolve None (auto-detect) to the effective value before toggling.
        effective = style.show_cell if style.show_cell is not None else has_lattice
        style.show_cell = not effective
    elif key == "a":
        effective = style.show_axes if style.show_axes is not None else has_lattice
        style.show_axes = not effective

    # -- Frame navigation --
    elif key == "]" and n_frames > 1:
        state["frame_index"] = (state["frame_index"] + 1) % n_frames
        return "full"
    elif key == "[" and n_frames > 1:
        state["frame_index"] = (state["frame_index"] - 1) % n_frames
        return "full"
    elif key == "}" and n_frames > 1:
        state["frame_index"] = n_frames - 1
        return "full"
    elif key == "{" and n_frames > 1:
        state["frame_index"] = 0
        return "full"

    # -- Reset --
    elif key == "r":
        view.rotation = initial_view["rotation"].copy()
        view.zoom = initial_view["zoom"]
        view.centre = initial_view["centre"].copy()
        view.perspective = initial_view["perspective"]
        view.view_distance = initial_view["view_distance"]

    # -- Help overlay --
    elif key == "h":
        state["help_visible"] = not state["help_visible"]

    else:
        return "none"

    return "view"


# ---------------------------------------------------------------------------
# Interactive renderer
# ---------------------------------------------------------------------------

def render_mpl_interactive(
    scene: StructureScene,
    *,
    style: RenderStyle | None = None,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
    **style_kwargs: object,
) -> tuple["ViewState", "RenderStyle"]:
    """Interactive matplotlib viewer with mouse and keyboard controls.

    Opens a matplotlib window where the user can manipulate the view
    with the mouse and keyboard:

    **Mouse:**

    - **Left-drag** to rotate the structure.
    - **Scroll** to zoom in/out.

    **Keyboard:**

    - **Arrow keys** rotate around the horizontal/vertical axes.
    - **,** / **.** roll in the screen plane.
    - **+** / **=** / **-** zoom in/out.
    - **Shift+Arrow** keys pan the view.
    - **p** / **P** increase/decrease perspective strength.
    - **d** / **D** increase/decrease viewing distance.
    - **b** toggle bonds, **o** toggle outlines, **e** toggle polyhedra,
      **u** toggle unit cell, **a** toggle axes widget.
    - **[** / **]** step to the previous/next frame;
      **{** / **}** jump to the first/last frame.
    - **r** reset the view to its initial state.
    - **h** toggle a help overlay listing all keybindings.

    When the window is closed the updated :class:`ViewState` and
    :class:`RenderStyle` are returned, allowing the user to re-use
    both for static rendering::

        view, style = scene.render_mpl_interactive()
        scene.view = view
        scene.render_mpl("output.svg", style=style)

    Args:
        scene: The StructureScene to render.
        style: A :class:`RenderStyle` controlling visual appearance.
            Any :class:`RenderStyle` field name may also be passed as
            a keyword argument to override individual fields.
        frame_index: Which frame to render initially.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution.
        background: Background colour.
        colour_by: Key into ``scene.atom_data`` to colour atoms by.
        cmap: Matplotlib colourmap name, object, or callable.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
        **style_kwargs: Any :class:`RenderStyle` field name as a
            keyword argument.  Unknown names raise :class:`TypeError`.

    Returns:
        A ``(ViewState, RenderStyle)`` tuple reflecting any view and
        style changes applied during the interactive session.
    """
    resolved = _resolve_style(style, **style_kwargs)

    n_frames = len(scene.frames)
    if not 0 <= frame_index < n_frames:
        raise ValueError(
            f"frame_index {frame_index} out of range for scene "
            f"with {n_frames} frame(s)"
        )

    bg_rgb = normalise_colour(background)

    # Use lower-fidelity polygon counts for interactive responsiveness.
    # Save the static values so we can restore them before returning.
    static_circle_segments = resolved.circle_segments
    static_arc_segments = resolved.arc_segments
    resolved.circle_segments = resolved.interactive_circle_segments
    resolved.arc_segments = resolved.interactive_arc_segments

    # Work on a copy so we don't mutate the original scene's view.
    view = ViewState(
        rotation=scene.view.rotation.copy(),
        zoom=scene.view.zoom,
        centre=scene.view.centre.copy(),
        perspective=scene.view.perspective,
        view_distance=scene.view.view_distance,
        slab_origin=(
            scene.view.slab_origin.copy()
            if scene.view.slab_origin is not None else None
        ),
        slab_near=scene.view.slab_near,
        slab_far=scene.view.slab_far,
    )

    # Fixed viewport extent — rotation-invariant so the scene doesn't
    # appear to shift or rescale while dragging.
    base_extent = _scene_extent(scene, view, frame_index, resolved.atom_scale)

    # Pre-compute bonds, colours, adjacency once — these don't change
    # during interactive rotation / zoom.
    colour_kwargs: dict[str, Any] = dict(
        colour_by=colour_by,
        cmap=cmap,
        colour_range=colour_range,
    )
    pre = _precompute_scene(scene, frame_index, resolved, **colour_kwargs)

    draw_kwargs: dict[str, Any] = dict(
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        precomputed=pre,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _draw_scene(ax, scene, view, resolved, viewport_extent=base_extent, **draw_kwargs)

    # ---- Interaction state ----

    state: dict = {
        "drag_active": False,
        "drag_last_xy": None,
        "last_draw_t": 0.0,
        "frame_index": frame_index,
        "help_visible": False,
        "precomputed": pre,
        "base_extent": base_extent,
    }

    # Snapshot for the reset key.
    initial_view = {
        "rotation": view.rotation.copy(),
        "zoom": view.zoom,
        "centre": view.centre.copy(),
        "perspective": view.perspective,
        "view_distance": view.view_distance,
    }

    _DRAG_SENSITIVITY = 0.01  # radians per pixel
    _MIN_INTERVAL = 0.03  # seconds between redraws (~30 fps cap)

    # ---- Redraw helpers ----

    def _redraw() -> None:
        """Repaint the scene using the fixed viewport extent."""
        _draw_scene(
            ax, scene, view, resolved,
            viewport_extent=state["base_extent"], **draw_kwargs,
        )
        if state["help_visible"]:
            _add_help_overlay()
        fig.canvas.draw_idle()
        state["last_draw_t"] = time.monotonic()

    def _throttled_redraw() -> None:
        """Redraw only if enough time has elapsed since the last draw."""
        if time.monotonic() - state["last_draw_t"] >= _MIN_INTERVAL:
            _redraw()

    def _full_redraw() -> None:
        """Recompute bonds/colours and repaint (for frame or style changes)."""
        state["precomputed"] = _precompute_scene(
            scene, state["frame_index"], resolved, **colour_kwargs,
        )
        draw_kwargs["precomputed"] = state["precomputed"]
        draw_kwargs["frame_index"] = state["frame_index"]
        state["base_extent"] = _scene_extent(
            scene, view, state["frame_index"], resolved.atom_scale,
        )
        _redraw()

    def _add_help_overlay() -> None:
        """Add the keybinding help text to the axes."""
        ax.text(
            0.02, 0.98, _HELP_TEXT,
            transform=ax.transAxes,
            fontsize=7,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="grey",
            ),
            zorder=1000,
        )

    # ---- Mouse handlers ----

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        state["drag_active"] = True
        state["drag_last_xy"] = (event.x, event.y)

    def on_motion(event):
        if not state["drag_active"] or state["drag_last_xy"] is None:
            return
        x0, y0 = state["drag_last_xy"]
        dx = event.x - x0
        dy = event.y - y0
        state["drag_last_xy"] = (event.x, event.y)
        # Incremental rotation in screen space: horizontal drag rotates
        # around the screen Y axis, vertical drag around screen X.
        # Applying to the *current* rotation gives intuitive "grab and
        # drag the object" behaviour regardless of accumulated rotation.
        view.rotation = (
            _rotation_y(dx * _DRAG_SENSITIVITY)
            @ _rotation_x(-dy * _DRAG_SENSITIVITY)
            @ view.rotation
        )
        _throttled_redraw()

    def on_release(event):
        if state["drag_active"]:
            state["drag_active"] = False
            # Final redraw to ensure the last position is rendered.
            _redraw()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        factor = _KEY_ZOOM_FACTOR ** event.step
        view.zoom = max(0.01, min(100.0, view.zoom * factor))
        _redraw()

    # ---- Keyboard handler ----

    def on_key_press(event):
        if event.key is None:
            return
        kind = _apply_key_action(
            event.key, view, resolved, state,
            n_frames=len(scene.frames),
            base_extent=base_extent,
            initial_view=initial_view,
            has_lattice=scene.lattice is not None,
        )
        if kind == "full":
            _full_redraw()
        elif kind == "view":
            _throttled_redraw()

    # ---- Connect events ----

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # Disconnect matplotlib's default key handler to avoid conflicts
    # (e.g. 'p' for pan tool, 'o' for zoom-to-rect).
    manager = fig.canvas.manager
    if manager is not None:
        handler_id = getattr(manager, "key_press_handler_id", None)
        if handler_id is not None:
            fig.canvas.mpl_disconnect(handler_id)

    plt.show()

    # Restore static-quality segment counts so the returned style
    # is ready for publication rendering.
    resolved.circle_segments = static_circle_segments
    resolved.arc_segments = static_arc_segments

    return view, resolved
