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

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

from hofmann.bonds import compute_bonds
from hofmann.model import Colour, StructureScene, ViewState, normalise_colour


_OUTLINE_COLOUR = (0.15, 0.15, 0.15)

# Pre-computed unit circle for atom rendering (closed polygon).
_N_CIRCLE = 24
_UNIT_CIRCLE = np.column_stack([
    np.cos(np.linspace(0, 2 * np.pi, _N_CIRCLE + 1)),
    np.sin(np.linspace(0, 2 * np.pi, _N_CIRCLE + 1)),
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



# Pre-computed semicircular arc (5 points per half-turn).
_N_ARC = 5
_ARC = np.column_stack([
    -np.sin(np.linspace(0, np.pi, _N_ARC)),
     np.cos(np.linspace(0, np.pi, _N_ARC)),
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
        bx * aa_a * _ARC[:, 0] + (-by) * bb_a * _ARC[:, 1] + start_2d[0],
        by * aa_a * _ARC[:, 0] +   bx  * bb_a * _ARC[:, 1] + start_2d[1],
    ])
    pts_end = np.column_stack([
        -bx * aa_b * _ARC[:, 0] + (-by) * bb_b * _ARC[:, 1] + end_2d[0],
        -by * aa_b * _ARC[:, 0] +   bx  * bb_b * _ARC[:, 1] + end_2d[1],
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
    n_bonds = len(bond_ia)
    if n_bonds == 0:
        empty2 = np.empty((0, 2))
        empty1 = np.empty((0,))
        return (
            np.empty((0, 2 * _N_ARC, 2)),
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
    # _ARC is (_N_ARC, 2); arc_x/arc_y are (_N_ARC,).
    arc_x = _ARC[:, 0]                                      # (_N_ARC,)
    arc_y = _ARC[:, 1]                                      # (_N_ARC,)

    # Start arc: (n_bonds, _N_ARC, 2).
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
    # full_verts[:, :_N_ARC] = start arc (+perp → -perp)
    # full_verts[:, _N_ARC:] = end arc (already reversed)
    arc_start = full_verts[:, :_N_ARC, :]                    # (n_bonds, _N_ARC, 2)
    arc_end = full_verts[:, _N_ARC:, :]                      # (n_bonds, _N_ARC, 2)

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
    return float(max_extent * view.zoom)


def _precompute_scene(
    scene: StructureScene,
    frame_index: int,
    bg_rgb: tuple[float, float, float],
) -> dict:
    """Pre-compute frame-independent data for repeated rendering.

    Returns a dict of radii, colours, bonds, and adjacency that stay
    constant across rotation / zoom changes.
    """
    frame = scene.frames[frame_index]
    coords = frame.coords
    n_atoms = len(scene.species)
    bg_bright = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]

    radii_3d = np.empty(n_atoms)
    atom_colours: list[tuple[float, float, float]] = []
    bond_half_colours: list[tuple[float, float, float]] = []

    for i in range(n_atoms):
        sp = scene.species[i]
        style = scene.atom_styles.get(sp)
        if style is not None:
            radii_3d[i] = style.radius
            rgb = normalise_colour(style.colour)
        else:
            radii_3d[i] = 0.5
            rgb = (0.5, 0.5, 0.5)
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if abs(brightness - bg_bright) < 0.15:
            bond_half_colours.append((0.5, 0.5, 0.5))
        else:
            bond_half_colours.append(rgb)
        if abs(brightness - bg_bright) < 0.15:
            rgb = (0.95, 0.95, 0.95)
        atom_colours.append(rgb)

    bonds = compute_bonds(scene.species, coords, scene.bond_specs)
    adjacency: dict[int, list[tuple[int, object]]] = defaultdict(list)
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

    return {
        "coords": coords,
        "radii_3d": radii_3d,
        "atom_colours": atom_colours,
        "bond_half_colours": bond_half_colours,
        "bonds": bonds,
        "adjacency": adjacency,
        "bg_bright": bg_bright,
        "bond_ia": bond_ia,
        "bond_ib": bond_ib,
        "bond_radii": bond_radii,
        "bond_index": bond_index,
    }


def _draw_scene(
    ax,
    scene: StructureScene,
    view: ViewState,
    *,
    frame_index: int = 0,
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    atom_scale: float = 0.5,
    bond_scale: float = 1.0,
    bond_colour: Colour | None = None,
    half_bonds: bool = True,
    viewport_extent: float | None = None,
    precomputed: dict | None = None,
) -> None:
    """Paint atoms and bonds onto *ax* using the painter's algorithm.

    Clears *ax* and redraws the full scene.  Does **not** create or
    show the figure — the caller owns the figure lifecycle.

    Args:
        ax: A matplotlib ``Axes`` to draw into.
        scene: The structure to render.
        view: Camera / projection state (may differ from ``scene.view``
            in interactive mode).
        frame_index: Which frame to render.
        bg_rgb: Normalised background colour.
        atom_scale: Scale factor for atom display radii.
        bond_scale: Scale factor for bond display width.
        bond_colour: Override colour for all bonds, or ``None`` for
            per-spec / half-bond colouring.
        half_bonds: Split each bond at the midpoint and colour halves
            to match the nearest atom.
        viewport_extent: If given, use this as the fixed half-extent
            for axis limits instead of computing from projected coords.
            Avoids the "view distance changing" artefact during
            interactive rotation.
        precomputed: Pre-computed scene data from :func:`_precompute_scene`.
            If ``None``, computed on the fly.
    """
    # Remove previous draw's collection(s) and leftover artists.
    while ax.collections:
        ax.collections[0].remove()
    for t in ax.texts[:]:
        t.remove()

    if precomputed is None:
        precomputed = _precompute_scene(scene, frame_index, bg_rgb)

    coords = precomputed["coords"]
    radii_3d = precomputed["radii_3d"]
    atom_colours = precomputed["atom_colours"]
    bond_half_colours = precomputed["bond_half_colours"]
    adjacency = precomputed["adjacency"]
    bg_bright = precomputed["bg_bright"]

    # ---- Projection ----
    xy, depth, atom_screen_radii = view.project(
        coords, radii_3d * atom_scale,
    )
    rotated = (coords - view.centre) @ view.rotation.T

    # ---- Sort atoms back-to-front (furthest first) ----
    order = np.argsort(depth)

    # ---- Slab visibility ----
    slab_visible = view.slab_mask(coords)

    ax.set_facecolor(bg_rgb)

    # ---- Batch bond geometry ----
    bond_ia = precomputed["bond_ia"]
    bond_ib = precomputed["bond_ib"]
    bond_radii_arr = precomputed["bond_radii"]
    bond_index = precomputed["bond_index"]

    use_half = half_bonds and bond_colour is None
    batch_valid = np.empty(0, dtype=bool)
    batch_full_verts = np.empty((0, 2 * _N_ARC, 2))
    batch_half_a = np.empty((0, _N_ARC + 2, 2))
    batch_half_b = np.empty((0, _N_ARC + 2, 2))

    if len(bond_ia) > 0:
        (batch_full_verts, batch_start_2d, batch_end_2d,
         batch_bb_a, batch_bb_b, batch_bx, batch_by,
         batch_valid) = _bond_polygons_batch(
            rotated, xy,
            radii_3d * atom_scale, atom_screen_radii,
            bond_ia, bond_ib, bond_radii_arr * bond_scale,
            view,
        )
        if use_half:
            batch_half_a, batch_half_b = _half_bond_verts_batch(
                batch_full_verts, batch_start_2d, batch_end_2d,
                batch_bb_a, batch_bb_b, batch_bx, batch_by,
            )

    # Pre-compute bond spec colours for the non-half-bond path.
    if not use_half:
        bond_spec_colours: dict[int, tuple[float, float, float]] = {}

    # Collect raw vertex arrays in painter's order, then batch-add
    # via PolyCollection (avoids costly Patch object creation).
    all_verts: list[np.ndarray] = []
    face_colours: list[tuple[float, float, float]] = []
    edge_colours: list[tuple[float, float, float]] = []
    line_widths: list[float] = []

    drawn_bonds: set[int] = set()

    # ---- Paint back-to-front ----
    for k in order:
        if not slab_visible[k]:
            continue

        neighbours = adjacency.get(k, [])
        neighbours_sorted = sorted(neighbours, key=lambda nb: depth[nb[0]])

        for kk, bond in neighbours_sorted:
            bond_id = id(bond)
            if bond_id in drawn_bonds:
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

            if use_half:
                all_verts.append(batch_half_a[bi])
                face_colours.append(bond_half_colours[ia])
                edge_colours.append(_OUTLINE_COLOUR)
                line_widths.append(1.0)
                all_verts.append(batch_half_b[bi])
                face_colours.append(bond_half_colours[ib])
                edge_colours.append(_OUTLINE_COLOUR)
                line_widths.append(1.0)
            else:
                spec_id = id(bond.spec)
                if spec_id not in bond_spec_colours:
                    if bond_colour is not None:
                        brgb = normalise_colour(bond_colour)
                    else:
                        brgb = normalise_colour(bond.spec.colour)
                        b = (0.299 * brgb[0] + 0.587 * brgb[1]
                             + 0.114 * brgb[2])
                        if abs(b - bg_bright) < 0.15:
                            brgb = (0.5, 0.5, 0.5)
                    bond_spec_colours[spec_id] = brgb
                brgb = bond_spec_colours[spec_id]

                all_verts.append(batch_full_verts[bi])
                face_colours.append(brgb)
                edge_colours.append(_OUTLINE_COLOUR)
                line_widths.append(1.0)

        all_verts.append(_UNIT_CIRCLE * atom_screen_radii[k] + xy[k])
        face_colours.append(atom_colours[k])
        edge_colours.append(_OUTLINE_COLOUR)
        line_widths.append(1.5)

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
        pad = viewport_extent * 1.15
    else:
        pad = np.max(atom_screen_radii) + 1.0
        pad = max(
            xy[:, 0].max() - xy[:, 0].min(),
            xy[:, 1].max() - xy[:, 1].min(),
        ) / 2 + pad
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.axis("off")

    if scene.title:
        ax.set_title(scene.title)


# ---------------------------------------------------------------------------
# Static renderer
# ---------------------------------------------------------------------------

def render_mpl(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    atom_scale: float = 0.5,
    bond_scale: float = 1.0,
    bond_colour: Colour | None = None,
    half_bonds: bool = True,
    show: bool = True,
) -> Figure:
    """Render a StructureScene as a static matplotlib figure.

    Uses a depth-sorted painter's algorithm: atoms are sorted
    back-to-front, and each atom's bonds are drawn just before the
    atom itself is painted.

    Bond-sphere intersections are computed in 3D and then projected
    to screen space.

    Args:
        scene: The StructureScene to render.
        output: Optional file path to save the figure (SVG, PDF, PNG).
        frame_index: Which frame to render (default 0).
        figsize: Figure size in inches.
        dpi: Resolution for raster output.
        background: Background colour.
        atom_scale: Scale factor for atom radii.
        bond_scale: Scale factor for bond width in data coordinates.
        bond_colour: Override colour for all bonds. If ``None``, each
            bond uses its BondSpec colour (with a grey fallback when
            the spec colour matches the background).
        half_bonds: If ``True`` (default), each bond is split at its
            midpoint and each half coloured to match the nearest atom.
            Ignored when *bond_colour* is set explicitly.
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib Figure object.
    """
    bg_rgb = normalise_colour(background)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)

    _draw_scene(
        ax, scene, scene.view,
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        atom_scale=atom_scale,
        bond_scale=bond_scale,
        bond_colour=bond_colour,
        half_bonds=half_bonds,
    )

    fig.tight_layout()

    if output is not None:
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")

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


# ---------------------------------------------------------------------------
# Interactive renderer
# ---------------------------------------------------------------------------

def render_mpl_interactive(
    scene: StructureScene,
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    atom_scale: float = 0.5,
    bond_scale: float = 1.0,
    bond_colour: Colour | None = None,
    half_bonds: bool = True,
) -> ViewState:
    """Interactive matplotlib viewer with mouse-driven rotation and zoom.

    Opens a matplotlib window where the user can:

    - **Left-drag** to rotate the structure (updates the rotation matrix).
    - **Scroll** to zoom in/out (updates the zoom factor).

    When the window is closed the updated :class:`ViewState` is returned,
    allowing the user to re-use the view for static rendering::

        view = scene.render_mpl_interactive()
        scene.view = view
        scene.render_mpl("output.svg", show=False)

    Args:
        scene: The StructureScene to render.
        frame_index: Which frame to render.
        figsize: Figure size in inches.
        dpi: Resolution.
        background: Background colour.
        atom_scale: Scale factor for atom radii.
        bond_scale: Scale factor for bond width.
        bond_colour: Override colour for all bonds.
        half_bonds: Split bonds at the midpoint and colour each half.

    Returns:
        The :class:`ViewState` reflecting any rotation/zoom applied
        during the interactive session.
    """
    bg_rgb = normalise_colour(background)

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
    base_extent = _scene_extent(scene, view, frame_index, atom_scale)

    # Pre-compute bonds, colours, adjacency once — these don't change
    # during interactive rotation / zoom.
    pre = _precompute_scene(scene, frame_index, bg_rgb)

    draw_kwargs = dict(
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        atom_scale=atom_scale,
        bond_scale=bond_scale,
        bond_colour=bond_colour,
        half_bonds=half_bonds,
        precomputed=pre,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _draw_scene(ax, scene, view, viewport_extent=base_extent, **draw_kwargs)

    # ---- Interaction state ----
    import time

    drag_state: dict = {
        "active": False,
        "last_xy": None,
    }

    sensitivity = 0.01  # radians per pixel
    _MIN_INTERVAL = 0.03  # seconds between redraws (~30 fps cap)
    _last_draw = {"t": 0.0}

    def _redraw() -> None:
        """Repaint the scene and flush to screen."""
        extent = _scene_extent(scene, view, frame_index, atom_scale)
        _draw_scene(ax, scene, view, viewport_extent=extent, **draw_kwargs)
        fig.canvas.draw_idle()
        _last_draw["t"] = time.monotonic()

    def _throttled_redraw() -> None:
        """Redraw only if enough time has elapsed since the last draw."""
        if time.monotonic() - _last_draw["t"] >= _MIN_INTERVAL:
            _redraw()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        drag_state["active"] = True
        drag_state["last_xy"] = (event.x, event.y)

    def on_motion(event):
        if not drag_state["active"] or drag_state["last_xy"] is None:
            return
        x0, y0 = drag_state["last_xy"]
        dx = event.x - x0
        dy = event.y - y0
        drag_state["last_xy"] = (event.x, event.y)
        # Incremental rotation in screen space: horizontal drag rotates
        # around the screen Y axis, vertical drag around screen X.
        # Applying to the *current* rotation gives intuitive "grab and
        # drag the object" behaviour regardless of accumulated rotation.
        view.rotation = (
            _rotation_y(dx * sensitivity)
            @ _rotation_x(-dy * sensitivity)
            @ view.rotation
        )
        _throttled_redraw()

    def on_release(event):
        if drag_state["active"]:
            drag_state["active"] = False
            # Final redraw to ensure the last position is rendered.
            _redraw()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        factor = 1.1 ** event.step
        view.zoom = max(0.01, min(100.0, view.zoom * factor))
        _redraw()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()

    return view
