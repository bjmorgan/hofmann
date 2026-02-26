"""Legend widget rendering.

Builds and draws a vertical legend column of coloured markers
(circles, regular polygons, or miniature 3D polyhedra) with text
labels.
"""

from __future__ import annotations

import re

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from hofmann.model import (
    AtomLegendItem,
    LegendItem,
    LegendStyle,
    PolygonLegendItem,
    PolyhedronLegendItem,
    StructureScene,
    WidgetCorner,
    _DEFAULT_CIRCLE_RADIUS,
    _DEFAULT_SPACING,
    normalise_colour,
)
from hofmann.rendering._legend_polyhedra import (
    CANONICAL_VERTICES,
    LEGEND_ROTATION,
    _get_faces,
    shade_face,
)
from hofmann.rendering._widget_scale import _REFERENCE_WIDGET_PTS
from hofmann._constants import POLYHEDRON_RADIUS_SCALE

# Multiplier applied to the default entry spacing when any legend label
# contains mathtext (super/subscripts).  The taller glyphs need a wider
# gap to avoid looking cramped.  Only applied when the user has not
# explicitly overridden the spacing value.
_MATHTEXT_SPACING_FACTOR = 1.3

_CHARGE_RE = re.compile(r"(\d+)([+-])$")
_SUBSCRIPT_RE = re.compile(r"(\d+)")

_GREY_FALLBACK = (0.5, 0.5, 0.5)


def _format_legend_label(text: str) -> str:
    """Auto-format chemical notation in a legend label.

    Converts common shorthand into matplotlib mathtext:

    * Trailing charge (``Sr2+``, ``O2-``) → superscript with tight
      kerning (``Sr$^{2\\!+}$``).
    * Embedded digits (``TiO6``, ``H2O``) → subscripts
      (``TiO$_6$``, ``H$_2$O``).

    Labels already containing ``$`` are returned unchanged, allowing
    users to provide explicit mathtext when needed.
    """
    if "$" in text:
        return text
    # Trailing charge: e.g. "Sr2+" -> "Sr$^{2\!+}$"
    m = _CHARGE_RE.search(text)
    if m:
        prefix = text[: m.start()]
        digit, sign = m.group(1), m.group(2)
        return f"{prefix}$^{{{digit}\\!{sign}}}$"
    # Embedded digits: e.g. "TiO6" -> "TiO$_6$", "H2O" -> "H$_2$O"
    if _SUBSCRIPT_RE.search(text):
        return _SUBSCRIPT_RE.sub(r"$_{\1}$", text)
    return text


def _build_legend_items(
    scene: StructureScene,
    style: LegendStyle,
) -> list[LegendItem]:
    """Build legend items from scene species and atom styles.

    Resolves the species list (auto-detect or explicit), looks up
    each species' colour from the scene's atom styles, and optionally
    attaches per-species labels and radii from the style.

    Args:
        scene: The structure scene (provides species and atom styles).
        style: Visual style for the legend.

    Returns:
        Ordered list of legend items, one per species.
    """
    # ---- Determine species list ----
    if style.species is not None:
        species_list = list(style.species)
    else:
        # Auto-detect: unique species in first-seen order, visible only.
        seen: dict[str, None] = {}
        for sp in scene.species:
            if sp not in seen:
                atom_style = scene.atom_styles.get(sp)
                if atom_style is None or atom_style.visible:
                    seen[sp] = None
        species_list = list(seen)

    if not species_list:
        return []

    # ---- Resolve per-species colours ----
    colours: dict[str, tuple[float, float, float]] = {}
    for sp in species_list:
        atom_style = scene.atom_styles.get(sp)
        if atom_style is not None:
            colours[sp] = normalise_colour(atom_style.colour)
        else:
            colours[sp] = _GREY_FALLBACK

    # ---- Resolve per-species radii ----
    # Proportional and dict modes set explicit radii; uniform mode
    # leaves radius as None so the draw loop uses the style default.
    radii: dict[str, float | None] = {}
    if isinstance(style.circle_radius, tuple):
        r_min_pts, r_max_pts = style.circle_radius
        atom_radii = {
            sp: scene.atom_styles[sp].radius
            if sp in scene.atom_styles else 1.0
            for sp in species_list
        }
        lo = min(atom_radii.values())
        hi = max(atom_radii.values())
        if hi == lo:
            radii = {sp: r_max_pts for sp in species_list}
        else:
            radii = {
                sp: r_min_pts + (r - lo) / (hi - lo) * (r_max_pts - r_min_pts)
                for sp, r in atom_radii.items()
            }
    elif isinstance(style.circle_radius, dict):
        radii = {
            sp: style.circle_radius.get(sp, _DEFAULT_CIRCLE_RADIUS)
            for sp in species_list
        }
    else:
        radii = {sp: None for sp in species_list}

    # ---- Resolve per-species labels ----
    labels: dict[str, str | None] = {}
    if style.labels is not None:
        for sp in species_list:
            labels[sp] = style.labels.get(sp)
    else:
        labels = {sp: None for sp in species_list}

    return [
        AtomLegendItem(
            key=sp,
            colour=colours[sp],
            label=labels[sp],
            radius=radii[sp],
        )
        for sp in species_list
    ]


def _resolve_item_radius(item: LegendItem, style: LegendStyle) -> float:
    """Resolve the display radius for a legend item in points (pre-scale).

    Returns the item's explicit radius if set, otherwise falls back
    to the style's uniform ``circle_radius`` (when it is a plain
    float) or ``_DEFAULT_CIRCLE_RADIUS``.  Polyhedron items without
    an explicit radius default to twice the flat-marker radius so
    that 3D icons are legible alongside the smaller circle markers.
    """
    if item.radius is not None:
        return item.radius
    base = (
        float(style.circle_radius)
        if isinstance(style.circle_radius, (int, float))
        else _DEFAULT_CIRCLE_RADIUS
    )
    if isinstance(item, PolyhedronLegendItem):
        return POLYHEDRON_RADIUS_SCALE * base
    return base


# ---------------------------------------------------------------------------
# 3D polyhedron icon rendering
# ---------------------------------------------------------------------------


def _draw_legend_polyhedron(
    ax: Axes,
    shape: str,
    centre_x: float,
    centre_y: float,
    radius_data: float,
    base_rgb: tuple[float, float, float],
    alpha: float,
    polyhedra_shading: float = 1.0,
    edge_colour: tuple[float, float, float] | None = None,
    edge_width: float = 0.0,
    rotation: np.ndarray | None = None,
) -> None:
    """Draw a miniature 3D-shaded polyhedron icon on *ax*.

    Uses the same depth-sorted face rendering as the main painter.
    When *rotation* is ``None`` the default oblique viewing angle
    from :data:`LEGEND_ROTATION` is used; otherwise the supplied
    3x3 rotation matrix orients the icon.

    Args:
        ax: Matplotlib ``Axes`` to draw into.
        shape: Polyhedron shape name (key in :data:`CANONICAL_VERTICES`).
        centre_x: Icon centre in data coordinates (x).
        centre_y: Icon centre in data coordinates (y).
        radius_data: Icon radius in data coordinates.
        base_rgb: Base face colour before shading.
        alpha: Face opacity (0--1).
        polyhedra_shading: Shading strength (0 = flat, 1 = full).
        edge_colour: Pre-resolved edge colour, or ``None`` to
            disable edges.
        edge_width: Pre-resolved edge width in points.
        rotation: 3x3 rotation matrix, or ``None`` for the default
            legend viewing angle.
    """
    rot = rotation if rotation is not None else LEGEND_ROTATION
    vertices = CANONICAL_VERTICES[shape]
    faces = _get_faces(shape)

    # Rotate, scale, and translate to the icon position.
    rotated = vertices @ rot.T
    scaled = rotated * radius_data
    translated = scaled[:, :2] + np.array([centre_x, centre_y])

    # Depth-sort faces back-to-front (ascending mean z after rotation).
    face_depths = [np.mean(rotated[face, 2]) for face in faces]
    face_order = np.argsort(face_depths)

    show_edges = edge_colour is not None and edge_width > 0.0
    ec_rgba: tuple[float, ...] | None = (
        (*edge_colour, 1.0) if edge_colour is not None and show_edges
        else None
    )

    all_verts: list[np.ndarray] = []
    face_colours: list[tuple[float, ...]] = []
    edge_colours_list: list[tuple[float, ...]] = []
    line_widths: list[float] = []

    for fi in face_order:
        face = faces[fi]
        face_verts_rotated = rotated[face]
        shaded = shade_face(face_verts_rotated, base_rgb, polyhedra_shading)

        verts_2d = translated[face]
        fc = (*shaded, alpha)
        all_verts.append(verts_2d)
        face_colours.append(fc)
        edge_colours_list.append(ec_rgba if ec_rgba is not None else fc)
        line_widths.append(edge_width if show_edges else 0.0)

    if all_verts:
        pc = PolyCollection(
            all_verts,
            closed=True,
            facecolors=face_colours,
            edgecolors=edge_colours_list,
            linewidths=line_widths,
            zorder=10,
        )
        ax.add_collection(pc)


def _draw_legend_widget(
    ax: Axes,
    scene: StructureScene,
    style: LegendStyle,
    pad_x: float,
    pad_y: float,
    cx: float = 0.0,
    cy: float = 0.0,
    outline_colour: tuple[float, float, float] | None = None,
    outline_width: float = 1.0,
    polyhedra_shading: float = 1.0,
) -> None:
    """Draw a legend widget on *ax*.

    A vertical column of coloured markers with labels beside them.
    Each entry corresponds to one :class:`LegendItem`.  Markers may
    be circles, regular polygons, or miniature 3D polyhedra depending
    on the item's fields.

    This function adds ``Line2D`` artists (flat markers),
    ``PolyCollection`` artists (3D polyhedra), and text labels.
    These are cleaned up on the next call to :func:`_draw_scene` via
    the ``ax.lines[:]``, ``ax.collections[:]``, and ``ax.texts[:]``
    removal.

    Args:
        ax: A matplotlib ``Axes`` to draw into.
        scene: The structure scene (provides species and atom styles
            for auto-generated items).  Unused when ``style.items``
            is provided.
        style: Visual style for the widget.
        pad_x: Viewport half-extent in the x direction (data coords).
        pad_y: Viewport half-extent in the y direction (data coords).
        cx: Viewport centre x coordinate.
        cy: Viewport centre y coordinate.
        outline_colour: Outline colour for legend markers, or ``None``
            to disable outlines.
        outline_width: Line width for marker outlines in points.
        polyhedra_shading: Shading strength for 3D polyhedron icons
            (0 = flat, 1 = full Lambertian-style shading).
    """
    if style.items is not None:
        items = list(style.items)
    else:
        items = _build_legend_items(scene, style)
    if not items:
        return

    # ---- Display-space scaling ----
    pad = max(pad_x, pad_y)
    fig = ax.get_figure()
    if not isinstance(fig, Figure):
        raise ValueError("ax is not attached to a Figure")
    ax_width_in = fig.get_figwidth() * ax.get_position().width
    pts_per_data = ax_width_in * 72.0 / (2.0 * pad_x)

    arrow_len_pts = 0.12 * pad * pts_per_data
    scale = arrow_len_pts / _REFERENCE_WIDGET_PTS

    # ---- Pre-format labels ----
    # Resolve and format all labels before layout so that we can
    # detect mathtext (super/subscripts) and adjust spacing.
    formatted_labels: list[str] = [
        _format_legend_label(item.display_label) for item in items
    ]

    font_size = style.font_size * scale
    default_spacing = style.spacing * scale
    # When the user hasn't explicitly set spacing, widen the gap for
    # labels that contain mathtext (super/subscripts are taller).
    if (
        style.spacing == _DEFAULT_SPACING
        and any("$" in lbl for lbl in formatted_labels)
    ):
        default_spacing *= _MATHTEXT_SPACING_FACTOR
    stroke_width = 3.0 * scale

    # ---- Resolve per-item circle radii (in points, pre-scale) ----
    item_radius_pts = [_resolve_item_radius(item, style) for item in items]

    # Scale all radii by the display-space factor.
    item_radius = [r * scale for r in item_radius_pts]
    max_circle_radius = max(item_radius)

    # Convert sizes from points to data coordinates for positioning.
    max_circle_r_data = max_circle_radius / pts_per_data
    font_data = font_size / pts_per_data
    entry_height = max(2 * max_circle_r_data, font_data)

    # ---- Resolve per-gap spacing ----
    # Each item's gap_after overrides the style-level default when set.
    gap_spacing: list[float] = []
    for item in items[:-1]:
        if item.gap_after is not None:
            gap_spacing.append(item.gap_after * scale / pts_per_data)
        else:
            gap_spacing.append(default_spacing / pts_per_data)

    # ---- Anchor position ----
    # The anchor is the circle centre for the first legend entry.
    # Label is always to the right of the circle (circle-left,
    # label-right) for natural left-to-right reading order.
    n_entries = len(items)
    total_height = (
        n_entries * entry_height + sum(gap_spacing)
    )
    label_gap_data = style.label_gap * scale / pts_per_data
    label_offset = max_circle_r_data + label_gap_data

    if isinstance(style.corner, tuple):
        fx, fy = style.corner
        anchor_x = (cx - pad_x) + 2 * pad_x * fx
        anchor_y = (cy - pad_y) + 2 * pad_y * fy
    else:
        inset_x = style.margin * pad_x
        inset_y = style.margin * pad_y
        if style.corner in (WidgetCorner.BOTTOM_LEFT, WidgetCorner.TOP_LEFT):
            anchor_x = (cx - pad_x) + inset_x + max_circle_r_data
        else:
            anchor_x = (cx + pad_x) - inset_x - max_circle_r_data
        if style.corner in (WidgetCorner.BOTTOM_LEFT, WidgetCorner.BOTTOM_RIGHT):
            anchor_y = (cy - pad_y) + inset_y + total_height
        else:
            anchor_y = (cy + pad_y) - inset_y

    # ---- Draw entries ----
    # Always stack downward from the anchor (top of legend).
    y_i = anchor_y
    for i, item in enumerate(items):

        rgb = normalise_colour(item.colour)

        # Resolve per-item edge settings: item override → scene fallback.
        # When outline_colour is None (show_outlines=False) and the item
        # has no override, edges are disabled.
        resolved_ec = (
            item.edge_colour if item.edge_colour is not None
            else outline_colour
        )
        resolved_ew = (
            item.edge_width if item.edge_width is not None
            else outline_width
        ) if resolved_ec is not None else 0.0

        if isinstance(item, PolyhedronLegendItem):
            # 3D polyhedron icon path.
            radius_data = item_radius[i] / pts_per_data
            _draw_legend_polyhedron(
                ax,
                shape=item.shape,
                centre_x=anchor_x,
                centre_y=y_i,
                radius_data=radius_data,
                base_rgb=rgb,
                alpha=item.alpha,
                polyhedra_shading=polyhedra_shading,
                edge_colour=resolved_ec,
                edge_width=resolved_ew,
                rotation=item.rotation,
            )
        else:
            # Flat marker path (circle or polygon).
            assert isinstance(item, (AtomLegendItem, PolygonLegendItem))
            face_colour: tuple[float, ...] = rgb
            if item.alpha < 1.0:
                face_colour = (*rgb, item.alpha)

            ax.plot(
                anchor_x, y_i,
                marker=item.marker,
                markersize=item_radius[i] * 2,
                markerfacecolor=face_colour,
                markeredgecolor=resolved_ec if resolved_ec is not None else rgb,
                markeredgewidth=resolved_ew,
                linestyle="None",
                zorder=10,
            )

        # Nudge text down to visually centre with the marker.
        # va="center" includes descender space in the bounding box,
        # so caps-only labels sit too high; -0.09 * font_data
        # compensates.
        text_y = y_i - 0.09 * font_data

        ax.text(
            anchor_x + label_offset, text_y, formatted_labels[i],
            fontsize=font_size,
            color=(0.0, 0.0, 0.0),
            ha="left",
            va="center",
            zorder=11,
            path_effects=[
                path_effects.withStroke(
                    linewidth=stroke_width, foreground="white",
                ),
            ],
        )

        # Advance downward for the next entry.
        if i < len(gap_spacing):
            y_i -= entry_height + gap_spacing[i]
