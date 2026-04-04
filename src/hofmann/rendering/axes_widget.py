"""Crystallographic axes orientation widget.

Draws three lines representing the a, b, c lattice directions
from a common origin in a corner of the viewport.
"""

from __future__ import annotations

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hofmann.model import (
    AxesStyle,
    ViewState,
    WidgetCorner,
    normalise_colour,
)
from hofmann.rendering._widget_scale import _REFERENCE_WIDGET_PTS


def _draw_axes_widget(
    ax: "Axes",
    lattice: np.ndarray,
    view: ViewState,
    style: AxesStyle,
) -> None:
    """Draw a crystallographic axes orientation widget on *ax*.

    Three lines representing the a, b, c lattice directions are drawn
    from a common origin in the specified corner of the viewport.  The
    lines rotate in sync with the structure via ``view.rotation``,
    with italic labels at the tips.

    The widget position is derived from the current axes limits, so
    ``ax.set_xlim`` / ``ax.set_ylim`` must be called before this
    function.

    This function adds ``Line2D`` artists and text labels.  The caller
    is responsible for clearing ``ax.lines`` and ``ax.texts`` before
    the next redraw.

    Args:
        ax: A matplotlib ``Axes`` to draw into.
        lattice: Unit-cell matrix, shape ``(3, 3)`` with rows as
            lattice vectors.
        view: Camera / projection state.
        style: Visual style for the widget.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = (xlim[0] + xlim[1]) / 2
    cy = (ylim[0] + ylim[1]) / 2
    pad_x = (xlim[1] - xlim[0]) / 2
    pad_y = (ylim[1] - ylim[0]) / 2
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
