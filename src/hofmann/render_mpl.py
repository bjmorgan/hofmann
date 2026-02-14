"""Depth-sorted matplotlib renderer for static publication-quality output.

Implements the painter's algorithm: atoms are sorted back-to-front, and
each atom draws its associated bonds before the next atom is painted on
top.

Bond-sphere intersections are computed in 3D (rotated coordinates) and
then projected to screen space.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from hofmann.bonds import compute_bonds
from hofmann.model import Colour, StructureScene, ViewState, normalise_colour


_OUTLINE_COLOUR = (0.15, 0.15, 0.15)


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
    """Compute bond polygon vertices with perspective-correct arcs.

    Each bond end is drawn as a semicircular arc transformed by a 2D
    affine matrix.  The transformation encodes the projected bond
    radius (perpendicular component) and the foreshortening of the
    sphere-cylinder tangent circle (along-axis component).

    Clipping uses the raw 3D radii (*r_a*, *r_b*), while the arc
    shape uses the ratio of projected display radii (*zr_a*, *zr_b*)
    to raw radii, so that ``atom_scale`` affects the displayed atom
    size but does not alter clipping geometry.

    Args:
        p_a: 3D position of atom a in rotated coordinates.
        p_b: 3D position of atom b in rotated coordinates.
        r_a: Raw 3D radius of atom a (before atom_scale).
        r_b: Raw 3D radius of atom b (before atom_scale).
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
    bond_vec = p_b - p_a
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-12:
        return None

    # 3D bond unit vector and tangent offsets using RAW radii.
    bond_unit = bond_vec / bond_len
    w_a = np.sqrt(max(r_a**2 - bond_r**2, 0.0)) if r_a > bond_r else 0.0
    w_b = np.sqrt(max(r_b**2 - bond_r**2, 0.0)) if r_b > bond_r else 0.0

    clip_start = p_a + bond_unit * w_a
    clip_end = p_b - bond_unit * w_b
    if np.dot(clip_end - clip_start, bond_unit) <= 0:
        return None

    # Per-end foreshortening angle: cosine of the angle between
    # the bond axis and the eye-to-atom vector.
    eye = np.array([0.0, 0.0, view.view_distance])
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

    # Arc centre: project atom centre to 2D, then offset along the
    # 2D bond direction by the projected tangent distance.
    atom_a_2d, _ = _project_point(p_a, view)
    atom_b_2d, _ = _project_point(p_b, view)

    # 2D bond direction from projected atom centres.
    bond_2d = atom_b_2d - atom_a_2d
    bond_2d_len = np.linalg.norm(bond_2d)
    if bond_2d_len < 1e-12:
        return None
    bx, by = bond_2d / bond_2d_len

    # Projected tangent offset: ww = w * sth * zr/r.
    zr_rk_a = zr_a / r_a if r_a > 1e-12 else 0.0
    zr_rk_b = zr_b / r_b if r_b > 1e-12 else 0.0

    ww_a = w_a * sth_a * zr_rk_a
    ww_b = w_b * sth_b * zr_rk_b

    start_2d = np.array([
        atom_a_2d[0] + bx * ww_a,
        atom_a_2d[1] + by * ww_a,
    ])
    end_2d = np.array([
        atom_b_2d[0] - bx * ww_b,
        atom_b_2d[1] - by * ww_b,
    ])

    # Affine matrix components for the arc transformation.
    # bb = bond_r * zr/r  (perpendicular half-width)
    # aa = bond_r * cth * zr/r  (along-axis foreshortening)
    bb_a = bond_r * zr_rk_a
    aa_a = bond_r * cth_a * zr_rk_a

    bb_b = bond_r * zr_rk_b
    aa_b = bond_r * cth_b * zr_rk_b

    # Build polygon vertices.  Arc points: (-sin(phi), cos(phi)).
    pts_start = np.column_stack([
        bx * aa_a * _ARC[:, 0] + (-by) * bb_a * _ARC[:, 1] + start_2d[0],
        by * aa_a * _ARC[:, 0] +   bx  * bb_a * _ARC[:, 1] + start_2d[1],
    ])

    pts_end = np.column_stack([
        -bx * aa_b * _ARC[:, 0] + (-by) * bb_b * _ARC[:, 1] + end_2d[0],
        -by * aa_b * _ARC[:, 0] +   bx  * bb_b * _ARC[:, 1] + end_2d[1],
    ])
    # Reverse end arc order so the polygon winds correctly.
    pts_end = pts_end[::-1]

    verts = np.vstack([pts_start, pts_end])
    return verts, start_2d, end_2d



# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_mpl(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] = (8.0, 8.0),
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
    frame = scene.frames[frame_index]
    coords = frame.coords
    n_atoms = len(scene.species)
    view = scene.view

    # ---- Pre-compute atom 3D radii ----
    radii_3d = np.empty(n_atoms)
    atom_colours: list[tuple[float, float, float]] = []
    bg_rgb = normalise_colour(background)
    bg_bright = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]

    # Colours for half-bond rendering (grey fallback for near-background).
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
        # For bonds: use visible grey when atom colour matches background.
        if abs(brightness - bg_bright) < 0.15:
            bond_half_colours.append((0.5, 0.5, 0.5))
        else:
            bond_half_colours.append(rgb)
        # For atoms: gentle nudge to off-white.
        if abs(brightness - bg_bright) < 0.15:
            rgb = (0.95, 0.95, 0.95)
        atom_colours.append(rgb)

    # ---- Projection ----
    xy, depth, atom_screen_radii = view.project(
        coords, radii_3d * atom_scale,
    )
    rotated = (coords - view.centre) @ view.rotation.T

    # ---- Compute bonds ----
    bonds = compute_bonds(scene.species, coords, scene.bond_specs)

    # Build adjacency: for each atom, list of (other_atom, bond_object).
    adjacency: dict[int, list[tuple[int, object]]] = defaultdict(list)
    for bond in bonds:
        adjacency[bond.index_a].append((bond.index_b, bond))
        adjacency[bond.index_b].append((bond.index_a, bond))

    # ---- Sort atoms back-to-front (furthest first) ----
    order = np.argsort(depth)  # ascending depth = back-to-front

    # Track which bonds have already been drawn (each bond drawn once).
    drawn_bonds: set[int] = set()

    # ---- Set up figure ----
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    ax.set_facecolor(bg_rgb)

    z = 0  # monotonically increasing zorder

    # ---- Paint back-to-front ----
    for k in order:
        # -- Draw bonds connected to this atom (sorted by far atom depth) --
        neighbours = adjacency.get(k, [])
        neighbours_sorted = sorted(neighbours, key=lambda nb: depth[nb[0]])

        for kk, bond in neighbours_sorted:
            bond_id = id(bond)
            if bond_id in drawn_bonds:
                continue

            # Bond drawn by the atom that is further back. Tie-break
            # by lower index.
            if depth[k] < depth[kk]:
                continue
            if depth[k] == depth[kk] and k > kk:
                continue

            drawn_bonds.add(bond_id)

            ia, ib = bond.index_a, bond.index_b
            br = bond.spec.radius

            # ---- Build bond polygon with perspective-correct arcs ----
            result = _bond_polygon(
                rotated[ia], rotated[ib],
                radii_3d[ia], radii_3d[ib],
                br * bond_scale,
                atom_screen_radii[ia], atom_screen_radii[ib],
                view,
            )
            if result is None:
                continue
            verts, start_2d, end_2d = result

            # ---- Draw bond (half-bond or single colour) ----
            use_half = half_bonds and bond_colour is None
            if use_half:
                mid_2d = (start_2d + end_2d) / 2
                # Split polygon at midpoint perpendicular.
                bond_dir = end_2d - start_2d
                n_pts = len(verts)
                half_a = []
                half_b = []
                for v in verts:
                    if np.dot(v - mid_2d, bond_dir) <= 0:
                        half_a.append(v)
                    else:
                        half_b.append(v)
                # Add midpoint edge intersections for clean split.
                bond_len_2d = np.linalg.norm(bond_dir)
                if bond_len_2d > 1e-12:
                    perp = np.array(
                        [-bond_dir[1], bond_dir[0]],
                    ) / bond_len_2d
                    # Interpolate half-width at midpoint from the
                    # start/end arc widths.
                    hw_s = br * bond_scale * view.zoom
                    mid_top = mid_2d + perp * hw_s
                    mid_bot = mid_2d - perp * hw_s
                    half_a.extend([mid_bot, mid_top])
                    half_b.extend([mid_top, mid_bot])

                for pts, ci in [(half_a, ia), (half_b, ib)]:
                    if len(pts) < 3:
                        continue
                    arr = np.array(pts)
                    # Sort by angle around centroid for correct winding.
                    c = arr.mean(axis=0)
                    angles = np.arctan2(arr[:, 1] - c[1],
                                        arr[:, 0] - c[0])
                    arr = arr[np.argsort(angles)]
                    poly = Polygon(
                        arr, closed=True,
                        facecolor=bond_half_colours[ci],
                        edgecolor=bond_half_colours[ci],
                        linewidth=0, zorder=z,
                    )
                    ax.add_patch(poly)
                    z += 1
            else:
                if bond_colour is not None:
                    brgb = normalise_colour(bond_colour)
                else:
                    brgb = normalise_colour(bond.spec.colour)
                    b = (0.299 * brgb[0] + 0.587 * brgb[1]
                         + 0.114 * brgb[2])
                    if abs(b - bg_bright) < 0.15:
                        brgb = (0.5, 0.5, 0.5)

                poly = Polygon(
                    verts, closed=True,
                    facecolor=brgb, edgecolor=brgb,
                    linewidth=0, zorder=z,
                )
                ax.add_patch(poly)
                z += 1

        # -- Draw this atom --
        circle = Circle(
            xy[k],
            atom_screen_radii[k],
            facecolor=atom_colours[k],
            edgecolor=_OUTLINE_COLOUR,
            linewidth=1.5,
            zorder=z,
        )
        ax.add_patch(circle)
        z += 1

    # ---- Axes and layout ----
    ax.set_aspect("equal")
    pad = np.max(atom_screen_radii) + 1.0
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.axis("off")

    if scene.title:
        ax.set_title(scene.title)

    fig.tight_layout()

    if output is not None:
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
