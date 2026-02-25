"""Painter's algorithm scene assembly and drawing.

Collects atoms, bonds, polyhedra faces, and cell edges into depth-sorted
order, then draws everything into a matplotlib Axes via a single
PolyCollection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.polyhedra import compute_polyhedra
from hofmann.construction.rendering_set import (
    build_rendering_set,
    deduplicate_molecules,
)
from hofmann.model import (
    AxesStyle,
    Bond,
    CmapSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    WidgetCorner,
    normalise_colour,
    resolve_atom_colours,
)
from hofmann.rendering.bond_geometry import (
    _bond_polygons_batch,
    _half_bond_verts_batch,
    _make_arc,
)
from hofmann.rendering._widget_scale import _REFERENCE_WIDGET_PTS
from hofmann.rendering.cell_edges import _collect_cell_edges
from hofmann.rendering.legend import _draw_legend_widget
from hofmann.rendering.projection import _make_unit_circle

# Font size (points) for scene titles rendered inside the viewport.
_TITLE_FONT_SIZE = 12.0


@dataclass(frozen=True)
class _PolyhedronRenderData:
    """Resolved rendering style for a single polyhedron.

    Groups the per-polyhedron visual properties computed once by
    :func:`_precompute_scene` and consumed by
    :func:`_collect_polyhedra_faces`.

    Attributes:
        base_colour: Face base colour before shading.
        alpha: Face transparency (0 = transparent, 1 = opaque).
        edge_colour: Wireframe edge colour.
        edge_width: Wireframe edge line width (points).
    """

    base_colour: tuple[float, float, float]
    alpha: float
    edge_colour: tuple[float, float, float]
    edge_width: float


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
    poly_render_data: list[_PolyhedronRenderData]


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
    rs = render_style or RenderStyle()
    frame = scene.frames[frame_index]
    coords = frame.coords

    # Run periodic bond pipeline: compute bonds (with MIC when
    # periodic), then build the expanded rendering set.
    lattice = scene.lattice if rs.pbc else None
    periodic_bonds = compute_bonds(
        scene.species, coords, scene.bond_specs, lattice=lattice,
    )

    if lattice is not None:
        rset = build_rendering_set(
            scene.species, coords, periodic_bonds,
            scene.bond_specs, lattice,
            max_recursive_depth=rs.max_recursive_depth,
            pbc_padding=rs.pbc_padding,
            polyhedra_specs=scene.polyhedra,
        )
        if rs.deduplicate_molecules:
            rset = deduplicate_molecules(rset, lattice)
        species = rset.species
        coords = rset.coords
        bonds = rset.bonds
        source_indices = rset.source_indices
    else:
        species = scene.species
        bonds = periodic_bonds
        source_indices = np.arange(len(scene.species))

    n_atoms = len(species)

    # Map atom_data through source_indices for expanded set.
    atom_data = {
        key: arr[source_indices]
        for key, arr in scene.atom_data.items()
    }

    radii_3d = np.empty(n_atoms)
    for i in range(n_atoms):
        sp = species[i]
        style = scene.atom_styles.get(sp)
        radii_3d[i] = style.radius if style is not None else 0.5

    atom_colours = resolve_atom_colours(
        species, scene.atom_styles, atom_data,
        colour_by=colour_by, cmap=cmap, colour_range=colour_range,
    )
    bond_half_colours = list(atom_colours)

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
        species, coords, bonds, scene.polyhedra,
    )

    # Atoms hidden by AtomStyle.visible=False — always applied,
    # regardless of show_polyhedra.
    style_hidden_atoms: set[int] = set()
    style_hidden_bond_ids: set[int] = set()
    for i, sp in enumerate(species):
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

    # Resolve rendering style per polyhedron.
    edge_width_override = (
        render_style.polyhedra_outline_width if render_style is not None
        else None
    )
    poly_render_data: list[_PolyhedronRenderData] = []
    for poly in polyhedra:
        if poly.spec.colour is not None:
            base_rgb = normalise_colour(poly.spec.colour)
        else:
            # Inherit from centre atom's resolved colour, which
            # accounts for colour_by / cmap when active.
            base_rgb = atom_colours[poly.centre_index]
        poly_render_data.append(_PolyhedronRenderData(
            base_colour=base_rgb,
            alpha=poly.spec.alpha,
            edge_colour=normalise_colour(poly.spec.edge_colour),
            edge_width=(
                edge_width_override if edge_width_override is not None
                else poly.spec.edge_width
            ),
        ))

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
        poly_render_data=poly_render_data,
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

    atom_depths_sorted = depth[order]
    for pi, poly in enumerate(polyhedra_list):
        if pi in poly_skip:
            continue
        prd = precomputed.poly_render_data[pi]
        base_rgb = prd.base_colour
        alpha = prd.alpha
        edge_rgb = prd.edge_colour
        edge_w = prd.edge_width
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

    if style.show_legend:
        _draw_legend_widget(
            ax, scene, style.legend_style,
            pad_x=pad_x, pad_y=pad_y, cx=cx, cy=cy,
            outline_colour=outline_rgb if show_outlines else None,
            outline_width=atom_outline_width,
        )


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
    scale = arrow_len_pts / _REFERENCE_WIDGET_PTS
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


def _axes_bg_rgb(ax: Axes) -> tuple[float, float, float]:
    """Return the axes background as an (R, G, B) tuple."""
    from matplotlib.colors import to_rgb
    return to_rgb(ax.get_facecolor())
