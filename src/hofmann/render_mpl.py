"""Depth-sorted matplotlib renderer for static publication-quality output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from hofmann.bonds import compute_bonds
from hofmann.model import Colour, StructureScene, normalise_colour


def _darken(rgb: tuple[float, float, float], factor: float = 0.6) -> tuple[float, float, float]:
    """Darken an RGB colour by the given factor."""
    return (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor)


def render_mpl(
    scene: StructureScene,
    output: str | Path | None = None,
    *,
    frame_index: int = 0,
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 150,
    background: Colour = "white",
    atom_scale: float = 0.4,
    bond_scale: float = 15.0,
    show: bool = True,
) -> Figure:
    """Render a StructureScene as a static matplotlib figure.

    Uses a depth-sorted painter's algorithm: objects furthest from
    the viewer are drawn first, with nearer objects painted on top.

    Args:
        scene: The StructureScene to render.
        output: Optional file path to save the figure (SVG, PDF, PNG).
        frame_index: Which frame to render (default 0).
        figsize: Figure size in inches.
        dpi: Resolution for raster output.
        background: Background colour.
        atom_scale: Scale factor for atom radii.
        bond_scale: Scale factor for bond line widths (in points).
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib Figure object.
    """
    frame = scene.frames[frame_index]
    coords = frame.coords

    # Project to 2D.
    xy, depth, scale = scene.view.project(coords)

    # Compute bonds for this frame.
    bonds = compute_bonds(scene.species, coords, scene.bond_specs)

    # Build a render list: (depth, type, data).
    render_list: list[tuple[float, str, dict]] = []

    # Atoms.
    for i in range(len(scene.species)):
        sp = scene.species[i]
        style = scene.atom_styles.get(sp)
        if style is not None:
            rgb = normalise_colour(style.colour)
            radius = style.radius * atom_scale * scale[i]
        else:
            rgb = (0.5, 0.5, 0.5)
            radius = 0.5 * atom_scale * scale[i]
        render_list.append((
            depth[i],
            "atom",
            {"xy": xy[i], "radius": radius, "colour": rgb},
        ))

    # Bonds — split into two half-segments.
    for bond in bonds:
        ia, ib = bond.index_a, bond.index_b
        mid_xy = (xy[ia] + xy[ib]) / 2
        mid_depth = (depth[ia] + depth[ib]) / 2
        mid_scale = (scale[ia] + scale[ib]) / 2
        lw = bond.spec.radius * bond_scale * mid_scale

        # Colour each half by the atom at that end.
        for atom_idx, start, end in [(ia, xy[ia], mid_xy), (ib, mid_xy, xy[ib])]:
            sp = scene.species[atom_idx]
            style = scene.atom_styles.get(sp)
            if style is not None:
                rgb = normalise_colour(style.colour)
            else:
                rgb = normalise_colour(bond.spec.colour)
            render_list.append((
                mid_depth,
                "bond",
                {"start": start, "end": end, "lw": lw, "colour": rgb},
            ))

    # Sort by depth descending (furthest first = painter's algorithm).
    render_list.sort(key=lambda item: item[0], reverse=True)

    # Draw.
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    bg_rgb = normalise_colour(background)
    fig.set_facecolor(bg_rgb)
    ax.set_facecolor(bg_rgb)

    for _depth, item_type, data in render_list:
        if item_type == "bond":
            ax.plot(
                [data["start"][0], data["end"][0]],
                [data["start"][1], data["end"][1]],
                color=data["colour"],
                linewidth=data["lw"],
                solid_capstyle="round",
                zorder=1,
            )
        elif item_type == "atom":
            circle = Circle(
                data["xy"],
                data["radius"],
                facecolor=data["colour"],
                edgecolor=_darken(data["colour"]),
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min() - 2, xy[:, 0].max() + 2)
    ax.set_ylim(xy[:, 1].min() - 2, xy[:, 1].max() + 2)
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
