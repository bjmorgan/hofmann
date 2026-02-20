"""Bond polygon geometry: 3D clipping, 2D projection, and half-bond splits."""

from __future__ import annotations

import numpy as np

from hofmann.model import ViewState
from hofmann.rendering.projection import _project_point


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
    # full_verts[:, :n_arc] = start arc (+perp -> -perp)
    # full_verts[:, n_arc:] = end arc (already reversed)
    arc_start = full_verts[:, :n_arc, :]                     # (n_bonds, n_arc, 2)
    arc_end = full_verts[:, n_arc:, :]                       # (n_bonds, n_arc, 2)

    # Half A: arc at atom a + straight cut at midpoint.
    # Winding: arc (+perp -> -perp), then mid_bot (-perp), mid_top (+perp).
    half_a = np.concatenate([
        arc_start,
        mid_bot[:, None, :],
        mid_top[:, None, :],
    ], axis=1)

    # Half B: straight cut at midpoint + arc at atom b.
    # Winding: mid_top (+perp), mid_bot (-perp), then end arc
    # (which winds -perp -> +perp, closing back to mid_top).
    half_b = np.concatenate([
        mid_top[:, None, :],
        mid_bot[:, None, :],
        arc_end,
    ], axis=1)

    return half_a, half_b
