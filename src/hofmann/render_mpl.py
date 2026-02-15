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
from hofmann.model import (
    Colour,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    normalise_colour,
)
from hofmann.polyhedra import compute_polyhedra

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
) -> dict:
    """Pre-compute frame-independent data for repeated rendering.

    Returns a dict of radii, colours, bonds, and adjacency that stay
    constant across rotation / zoom changes.
    """
    frame = scene.frames[frame_index]
    coords = frame.coords
    n_atoms = len(scene.species)
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
        bond_half_colours.append(rgb)
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

    # ---- Polyhedra ----
    polyhedra = compute_polyhedra(
        scene.species, coords, bonds, scene.polyhedra,
    )

    # Build sets of hidden atoms/bonds from polyhedra specs.
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
        if poly.spec.hide_bonds:
            for kk, bond in adjacency.get(poly.centre_index, []):
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
    for poly in polyhedra:
        if poly.spec.colour is not None:
            base_rgb = normalise_colour(poly.spec.colour)
        else:
            # Inherit from centre atom's style.
            sp = scene.species[poly.centre_index]
            style = scene.atom_styles.get(sp)
            base_rgb = normalise_colour(style.colour) if style else (0.5, 0.5, 0.5)
        poly_base_colours.append(base_rgb)
        poly_alphas.append(poly.spec.alpha)
        poly_edge_colours.append(normalise_colour(poly.spec.edge_colour))
        poly_edge_widths.append(poly.spec.edge_width)

    return {
        "coords": coords,
        "radii_3d": radii_3d,
        "atom_colours": atom_colours,
        "bond_half_colours": bond_half_colours,
        "bonds": bonds,
        "adjacency": adjacency,
        "bond_ia": bond_ia,
        "bond_ib": bond_ib,
        "bond_radii": bond_radii,
        "bond_index": bond_index,
        "polyhedra": polyhedra,
        "hidden_atoms": hidden_atoms,
        "hidden_bond_ids": hidden_bond_ids,
        "poly_base_colours": poly_base_colours,
        "poly_alphas": poly_alphas,
        "poly_edge_colours": poly_edge_colours,
        "poly_edge_widths": poly_edge_widths,
    }


def _draw_scene(
    ax,
    scene: StructureScene,
    view: ViewState,
    style: RenderStyle,
    *,
    frame_index: int = 0,
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
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
        style: Visual style settings.
        frame_index: Which frame to render.
        bg_rgb: Normalised background colour.
        viewport_extent: If given, use this as the fixed half-extent
            for axis limits instead of computing from projected coords.
            Avoids the "view distance changing" artefact during
            interactive rotation.
        precomputed: Pre-computed scene data from :func:`_precompute_scene`.
            If ``None``, computed on the fly.
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
    # Remove previous draw's collection(s) and leftover artists.
    while ax.collections:
        ax.collections[0].remove()
    for t in ax.texts[:]:
        t.remove()

    if precomputed is None:
        precomputed = _precompute_scene(scene, frame_index)

    coords = precomputed["coords"]
    radii_3d = precomputed["radii_3d"]
    atom_colours = precomputed["atom_colours"]
    bond_half_colours = precomputed["bond_half_colours"]
    adjacency = precomputed["adjacency"]

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
    poly_skip: set[int] = set()
    poly_clip_hidden_bonds: set[int] = set()
    polyhedra_list = precomputed["polyhedra"]
    if (show_polyhedra and polyhedra_list
            and slab_clip_mode != SlabClipMode.PER_FACE):
        slab_force_visible: set[int] = set()
        for pi, poly in enumerate(polyhedra_list):
            all_vertices = set(poly.neighbour_indices) | {poly.centre_index}
            if slab_clip_mode == SlabClipMode.CLIP_WHOLE:
                if not all(slab_visible[v] for v in all_vertices):
                    poly_skip.add(pi)
                    # Hide centre-to-vertex bonds for skipped polyhedra.
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

    if show_bonds and len(bond_ia) > 0:
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

    # ---- Polyhedra face data ----
    hidden_atoms = precomputed["hidden_atoms"]
    hidden_bond_ids = precomputed["hidden_bond_ids"]
    poly_base_colours = precomputed["poly_base_colours"]
    poly_alphas = precomputed["poly_alphas"]
    poly_edge_colours = precomputed["poly_edge_colours"]
    poly_edge_widths = precomputed["poly_edge_widths"]

    # Build per-face data: 2D vertices, shaded RGBA, edge colour/width,
    # centroid depth, and assign each face to a depth slot.
    # face_by_depth_slot[order_position] -> list of face draw tuples.
    face_by_depth_slot: dict[int, list[tuple[np.ndarray, tuple, tuple, float]]] = defaultdict(list)
    if show_polyhedra and polyhedra_list:
        atom_depths_sorted = depth[order]  # depths in back-to-front order
        for pi, poly in enumerate(polyhedra_list):
            if pi in poly_skip:
                continue
            base_rgb = poly_base_colours[pi]
            alpha = poly_alphas[pi]
            edge_rgb = poly_edge_colours[pi]
            edge_w = poly_edge_widths[pi]
            for face_row in poly.faces:
                # Resolve local face indices to global atom indices.
                global_idx = [poly.neighbour_indices[j] for j in face_row]

                # Slab check: all vertices must be visible.
                # (In include_whole mode, slab_visible has already
                # been updated to force polyhedron vertices visible.)
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
                shading = 0.4 + 0.6 * cos_angle
                shaded = tuple(min(1.0, c * shading) for c in base_rgb)

                # Face centroid depth.
                face_depth = np.mean(depth[global_idx])

                # 2D projected vertices.
                verts_2d = xy[global_idx]

                # Assign to the depth slot of the first atom in 'order'
                # whose depth >= face_depth.
                slot = int(np.searchsorted(atom_depths_sorted, face_depth))
                if slot >= len(order):
                    slot = len(order) - 1

                face_by_depth_slot[slot].append((
                    verts_2d,
                    (*shaded, alpha),
                    (*edge_rgb, 1.0),
                    edge_w,
                ))

    # Collect raw vertex arrays in painter's order, then batch-add
    # via PolyCollection (avoids costly Patch object creation).
    all_verts: list[np.ndarray] = []
    face_colours: list[tuple[float, ...]] = []
    edge_colours: list[tuple[float, ...]] = []
    line_widths: list[float] = []

    drawn_bonds: set[int] = set()

    # ---- Paint back-to-front ----
    for order_pos, k in enumerate(order):
        if not slab_visible[k]:
            continue

        neighbours = adjacency.get(k, []) if show_bonds else []
        neighbours_sorted = sorted(neighbours, key=lambda nb: depth[nb[0]])

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
                edge_colours.append((*outline_rgb, 1.0) if show_outlines else fc_a)
                line_widths.append(bond_outline_width if show_outlines else 0.0)
                fc_b = (*bond_half_colours[ib], 1.0)
                all_verts.append(batch_half_b[bi])
                face_colours.append(fc_b)
                edge_colours.append((*outline_rgb, 1.0) if show_outlines else fc_b)
                line_widths.append(bond_outline_width if show_outlines else 0.0)
            elif use_half:
                # Same colour on both halves — draw as single polygon
                # to avoid a spurious outline at the midpoint.
                fc = (*bond_half_colours[ia], 1.0)
                all_verts.append(batch_full_verts[bi])
                face_colours.append(fc)
                edge_colours.append((*outline_rgb, 1.0) if show_outlines else fc)
                line_widths.append(bond_outline_width if show_outlines else 0.0)
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
                edge_colours.append((*outline_rgb, 1.0) if show_outlines else (*brgb, 1.0))
                line_widths.append(bond_outline_width if show_outlines else 0.0)

        # Draw polyhedron faces assigned to this depth slot.
        for face_verts, face_fc, face_ec, face_lw in face_by_depth_slot.get(order_pos, []):
            all_verts.append(face_verts)
            face_colours.append(face_fc)
            edge_colours.append(face_ec)
            line_widths.append(face_lw)

        # Draw atom circle (unless hidden by a polyhedron spec).
        if k not in hidden_atoms:
            fc_atom = (*atom_colours[k], 1.0)
            all_verts.append(_UNIT_CIRCLE * atom_screen_radii[k] + xy[k])
            face_colours.append(fc_atom)
            edge_colours.append((*outline_rgb, 1.0) if show_outlines else fc_atom)
            line_widths.append(atom_outline_width if show_outlines else 0.0)

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

_STYLE_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(RenderStyle))

_RENDER_STYLE_KWARGS = {"atom_scale", "bond_scale", "show_bonds"}
"""Convenience kwargs accepted directly by render functions and forwarded
into the :class:`RenderStyle`."""


def _resolve_style(
    style: RenderStyle | None,
    **kwargs: object,
) -> RenderStyle:
    """Build a :class:`RenderStyle` from an optional base plus overrides.

    Any kwarg whose name matches a ``RenderStyle`` field replaces that
    field's value.  Unknown kwargs are silently ignored (they belong to
    the caller).
    """
    from dataclasses import replace

    s = style if style is not None else RenderStyle()
    overrides = {k: v for k, v in kwargs.items() if k in _STYLE_FIELDS and v is not None}
    if overrides:
        s = replace(s, **overrides)
    return s


def render_mpl(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    style: RenderStyle | None = None,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    atom_scale: float | None = None,
    bond_scale: float | None = None,
    show_bonds: bool | None = None,
    show: bool = True,
) -> Figure:
    """Render a StructureScene as a static matplotlib figure.

    Uses a depth-sorted painter's algorithm: atoms are sorted
    back-to-front, and each atom's bonds are drawn just before the
    atom itself is painted.  Bond-sphere intersections are computed
    in 3D and then projected to screen space.

    Example usage::

        scene = StructureScene.from_xbs("ch4.bs")
        scene.render_mpl("ch4.png")

        # Publication-quality SVG with custom sizing:
        scene.render_mpl("ch4.svg", figsize=(8, 8), dpi=300,
                         background="black", show=False)

        # Atoms only (no bonds):
        scene.render_mpl(show_bonds=False)

        # Custom style with no outlines:
        from hofmann import RenderStyle
        style = RenderStyle(show_outlines=False, atom_scale=0.8)
        scene.render_mpl("clean.svg", style=style, show=False)

        # View along the [1, 1, 1] direction with a depth slab:
        scene.view.look_along([1, 1, 1])
        scene.view.slab_near = -2.0
        scene.view.slab_far = 2.0
        scene.render_mpl("slice.png", show=False)

    Args:
        scene: The StructureScene to render.
        output: Optional file path to save the figure.  The format is
            inferred from the extension (e.g. ``.svg``, ``.pdf``,
            ``.png``).
        style: A :class:`RenderStyle` controlling visual appearance.
            If ``None``, defaults are used.  Convenience kwargs
            (*atom_scale*, *bond_scale*, *show_bonds*) override the
            corresponding style fields.
        frame_index: Which frame to render (default 0).
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution for raster output formats.
        background: Background colour (CSS name, hex string, grey
            float, or RGB tuple).
        atom_scale: Override for :attr:`RenderStyle.atom_scale`.
        bond_scale: Override for :attr:`RenderStyle.bond_scale`.
        show_bonds: Override for :attr:`RenderStyle.show_bonds`.
        show: Whether to call ``plt.show()``.  Set to ``False`` when
            saving to a file or working non-interactively.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure` object.
    """
    resolved = _resolve_style(
        style,
        atom_scale=atom_scale,
        bond_scale=bond_scale,
        show_bonds=show_bonds,
    )
    bg_rgb = normalise_colour(background)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)

    _draw_scene(
        ax, scene, scene.view, resolved,
        frame_index=frame_index,
        bg_rgb=bg_rgb,
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
    style: RenderStyle | None = None,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    atom_scale: float | None = None,
    bond_scale: float | None = None,
    show_bonds: bool | None = None,
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
        style: A :class:`RenderStyle` controlling visual appearance.
            Convenience kwargs override the corresponding style fields.
        frame_index: Which frame to render.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution.
        background: Background colour.
        atom_scale: Override for :attr:`RenderStyle.atom_scale`.
        bond_scale: Override for :attr:`RenderStyle.bond_scale`.
        show_bonds: Override for :attr:`RenderStyle.show_bonds`.

    Returns:
        The :class:`ViewState` reflecting any rotation/zoom applied
        during the interactive session.
    """
    resolved = _resolve_style(
        style,
        atom_scale=atom_scale,
        bond_scale=bond_scale,
        show_bonds=show_bonds,
    )
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
    base_extent = _scene_extent(scene, view, frame_index, resolved.atom_scale)

    # Pre-compute bonds, colours, adjacency once — these don't change
    # during interactive rotation / zoom.
    pre = _precompute_scene(scene, frame_index)

    draw_kwargs = dict(
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        precomputed=pre,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _draw_scene(ax, scene, view, resolved, viewport_extent=base_extent, **draw_kwargs)

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
        extent = _scene_extent(scene, view, frame_index, resolved.atom_scale)
        _draw_scene(ax, scene, view, resolved, viewport_extent=extent, **draw_kwargs)
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
