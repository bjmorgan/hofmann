"""Unit cell edge geometry: computation, clipping, and dash patterns."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hofmann.model import CellEdgeStyle, StructureScene, ViewState, normalise_colour

# The 12 edges of a unit cube, as pairs of vertex indices.
# Vertices are the 8 corners at fractional coordinates {0,1}^3.
_CUBE_EDGES: list[tuple[int, int]] = []
for _i in range(8):
    for _bit in range(3):
        _j = _i ^ (1 << _bit)  # flip one bit
        if _j > _i:
            _CUBE_EDGES.append((_i, _j))

# Fractional coordinates of the 8 cube corners (row-order matches
# the bit-pattern vertex indexing: 0->(0,0,0), 1->(1,0,0), ..., 7->(1,1,1)).
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
