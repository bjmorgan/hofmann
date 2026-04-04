"""Painter's algorithm scene drawing.

Draws atoms, bonds, polyhedra faces, and cell edges in depth-sorted
order into a matplotlib Axes via a single PolyCollection.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

from hofmann.model import (
    CmapSpec,
    RenderStyle,
    StructureScene,
    ViewState,
    normalise_colour,
)
from hofmann.rendering.axes_widget import _draw_axes_widget
from hofmann.rendering.bond_geometry import (
    _bond_polygons_batch,
    _half_bond_verts_batch,
    _make_arc,
)
from hofmann.rendering.cell_edges import _collect_cell_edges
from hofmann.rendering.legend import _draw_legend_widget
from hofmann.rendering.precompute import (
    _PrecomputedScene,
    _apply_slab_clip,
    _collect_polyhedra_faces,
    _precompute_scene,
)
from hofmann.rendering.projection import _make_unit_circle

# Font size (points) for scene titles rendered inside the viewport.
_TITLE_FONT_SIZE = 12.0


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
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
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
            ``_precompute_scene()``.  If ``None``, computed on the
            fly.
        colour_by: Key (or list of keys) into ``scene.atom_data``
            for colourmap-based colouring.
        cmap: Matplotlib colourmap name, object, or callable.  When
            *colour_by* is a list, may also be a list of the same
            length.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
            When *colour_by* is a list, may also be a list of the
            same length.
        fixed_xlim: If given, use these x-axis limits instead of
            computing from the projected geometry.  Used by
            :func:`render_animation` to lock the viewport across
            frames.
        fixed_ylim: If given, use these y-axis limits instead of
            computing from the projected geometry.
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
    all_hidden_atoms = set(precomputed.style_hidden_atoms)
    all_hidden_bond_ids = set(precomputed.style_hidden_bond_ids)
    if show_polyhedra:
        all_hidden_atoms |= precomputed.hidden_atoms
        all_hidden_bond_ids |= precomputed.hidden_bond_ids

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
            if vi in all_hidden_atoms:
                continue
            vi_slot = int(np.searchsorted(atom_depths_sorted, depth[vi]))
            if vi_slot <= max_slot:
                deferred_vertex_atoms.add(vi)
                in_front_after_slot[max_slot].append(vi)
    # Sort deferred vertices within each slot back-to-front.
    for slot in in_front_after_slot:
        in_front_after_slot[slot].sort(key=lambda v: depth[v])

    # ---- Cell edge data ----
    lattice = scene.frames[frame_index].lattice
    draw_cell = style.show_cell
    if draw_cell is None:
        draw_cell = lattice is not None
    if draw_cell and lattice is None:
        raise ValueError(
            f"show_cell=True but frame {frame_index} has no lattice"
        )

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
        # Only clip cell edges at atoms that are actually drawn (#41).
        clip_visible = slab_visible.copy()
        if all_hidden_atoms:
            clip_visible[list(all_hidden_atoms)] = False
        cell_edge_by_depth_slot = _collect_cell_edges(
            lattice, view, style.cell_style, depth, order, cell_pad,
            coords[clip_visible], (radii_3d * atom_scale)[clip_visible],
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
                if bond_id in all_hidden_bond_ids:
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
        if (k not in all_hidden_atoms
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
        # Divide by zoom so that zoom > 1 crops the viewport
        # (zooms in) and zoom < 1 adds padding (zooms out).
        # The projected coordinates already have zoom applied,
        # so at zoom=1 everything fits; at zoom=2 the viewport
        # is half as wide and edge atoms are clipped.
        pad_x = ((xy[:, 0].max() - xy[:, 0].min()) / 2 + margin) / view.zoom
        pad_y = ((xy[:, 1].max() - xy[:, 1].min()) / 2 + margin) / view.zoom
    # ---- Axes orientation widget ----
    draw_axes = style.show_axes
    if draw_axes is None:
        draw_axes = lattice is not None
    if draw_axes and viewport_extent is None:
        # Expand viewport so the widget doesn't overlap atoms.
        # The widget spans (margin + 2 * arrow_length) * pad from the
        # corner; halve this because expansion is applied to both sides.
        axes_style = style.axes_style
        widget_frac = axes_style.margin + 2.0 * axes_style.arrow_length
        expand_per_side = widget_frac * 0.5
        pad_x *= 1.0 + expand_per_side
        pad_y *= 1.0 + expand_per_side
    if (fixed_xlim is None) != (fixed_ylim is None):
        raise ValueError(
            "fixed_xlim and fixed_ylim must both be provided or both "
            "be None"
        )
    if fixed_xlim is not None and fixed_ylim is not None:
        ax.set_xlim(*fixed_xlim)
        ax.set_ylim(*fixed_ylim)
    else:
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
        if lattice is None:
            raise ValueError(
                f"show_axes=True but frame {frame_index} has no lattice"
            )
        _draw_axes_widget(ax, lattice, view, style.axes_style)

    if style.show_legend:
        _draw_legend_widget(
            ax, scene, style.legend_style,
            outline_colour=outline_rgb if show_outlines else None,
            outline_width=atom_outline_width,
            polyhedra_shading=polyhedra_shading,
        )


def _axes_bg_rgb(ax: Axes) -> tuple[float, float, float]:
    """Return the axes background as an (R, G, B) tuple."""
    from matplotlib.colors import to_rgb
    return to_rgb(ax.get_facecolor())
