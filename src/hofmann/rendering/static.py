"""Static matplotlib renderer: :func:`render_mpl` entry point."""

from __future__ import annotations

from pathlib import Path
import types
import typing
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hofmann.model import (
    CmapSpec,
    Colour,
    LegendStyle,
    RenderStyle,
    StructureScene,
    normalise_colour,
)
from hofmann.rendering.painter import _axes_bg_rgb, _draw_legend_widget, _draw_scene

_STYLE_FIELDS = frozenset(f.name for f in __import__("dataclasses").fields(RenderStyle))
_DEFAULT_RENDER_STYLE = RenderStyle()

# Fields where ``None`` is a meaningful value (not just "unset").
_NULLABLE_STYLE_FIELDS = frozenset(
    name for name, tp in typing.get_type_hints(RenderStyle).items()
    if typing.get_origin(tp) is types.UnionType
    and type(None) in typing.get_args(tp)
)

def _resolve_style(
    style: RenderStyle | None,
    **kwargs: Any,
) -> RenderStyle:
    """Build a :class:`RenderStyle` from an optional base plus overrides.

    Any kwarg whose name matches a ``RenderStyle`` field replaces that
    field's value.  For most fields, passing ``None`` is treated as
    "not provided" and preserves the base value.  For fields that
    accept ``None`` as a meaningful value (e.g. ``pbc_padding``),
    ``None`` is passed through as an explicit override.

    Unknown kwargs raise :class:`TypeError`.

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
    overrides = {
        k: v for k, v in kwargs.items()
        if v is not None or k in _NULLABLE_STYLE_FIELDS
    }
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


def render_legend(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    legend_style: LegendStyle | None = None,
    show_outlines: bool = True,
    outline_colour: Colour = (0.15, 0.15, 0.15),
    outline_width: float = 1.0,
    dpi: int = 150,
    background: Colour = "white",
) -> Figure:
    """Render a standalone species legend as a tight matplotlib figure.

    Produces a figure containing only the legend — no structure, bonds,
    cell edges, or axes widget.  Useful for composing figures manually
    in external tools (Inkscape, Illustrator, LaTeX).

    The legend entries, colours, and circle sizes are determined by the
    scene's atom styles and the *legend_style* settings, using the same
    rendering code as the in-scene legend drawn by ``show_legend=True``.

    Args:
        scene: The structure scene (provides species and atom styles).
        output: Optional file path to save the figure.  The format is
            inferred from the extension (e.g. ``".svg"``, ``".png"``).
        legend_style: Visual style for the legend.  ``None`` uses
            defaults.  See :class:`~hofmann.LegendStyle`.
        show_outlines: Whether to draw outlines around legend circles.
        outline_colour: Colour for circle outlines when
            *show_outlines* is ``True``.
        outline_width: Line width for circle outlines in points.
        dpi: Resolution for raster output formats.
        background: Figure background colour.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.  When
        *output* is given the figure is saved and then closed;
        otherwise it remains open for further manipulation.

    Example::

        from hofmann import LegendStyle
        from hofmann.rendering.static import render_legend

        fig = render_legend(scene, "legend.svg")

        # Proportional circle sizes:
        style = LegendStyle(circle_radius=(3.0, 8.0))
        fig = render_legend(scene, "legend.svg", legend_style=style)
    """
    if legend_style is None:
        legend_style = LegendStyle()

    bg_rgb = normalise_colour(background)

    # Create a figure with a hidden axes for the legend widget to
    # draw into.  The coordinate system is arbitrary — the widget
    # uses display-space scaling internally.
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    fig.set_facecolor(bg_rgb)
    ax.set_facecolor(bg_rgb)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ol_rgb: tuple[float, float, float] | None = None
    if show_outlines:
        ol_rgb = normalise_colour(outline_colour)

    _draw_legend_widget(
        ax, scene, legend_style,
        pad_x=1.0, pad_y=1.0,
        outline_colour=ol_rgb,
        outline_width=outline_width,
    )

    # Crop to the legend artists.
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    bboxes = [
        t.get_window_extent(renderer) for t in ax.texts
    ] + [
        line.get_window_extent(renderer) for line in ax.lines
        if line.get_marker() == "o"
    ]
    if bboxes:
        legend_bb = Bbox.union(bboxes)
        pad_px = 15
        padded = Bbox([
            [legend_bb.x0 - pad_px, legend_bb.y0 - pad_px],
            [legend_bb.x1 + pad_px, legend_bb.y1 + pad_px],
        ])
        crop_dpi = fig.dpi
        bbox_inches = Bbox([
            [padded.x0 / crop_dpi, padded.y0 / crop_dpi],
            [padded.x1 / crop_dpi, padded.y1 / crop_dpi],
        ])
    else:
        bbox_inches = "tight"

    if output is not None:
        fig.savefig(str(output), dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)

    return fig
