"""Interactive plotly 3D renderer with trajectory animation support."""

from __future__ import annotations

from hofmann.bonds import compute_bonds
from hofmann.model import Colour, StructureScene, normalise_colour


def _rgb_string(colour: Colour) -> str:
    """Convert a colour spec to a plotly-compatible ``rgb(r,g,b)`` string."""
    r, g, b = normalise_colour(colour)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def _build_traces(scene: StructureScene, frame_index: int, atom_scale: float):
    """Build atom and bond traces for a single frame.

    Returns a tuple of ``(atom_trace, bond_trace)`` or
    ``(atom_trace,)`` if there are no bonds.
    """
    import plotly.graph_objects as go

    frame = scene.frames[frame_index]
    coords = frame.coords

    # Per-atom colours and sizes.
    colours = []
    sizes = []
    for sp in scene.species:
        style = scene.atom_styles.get(sp)
        if style is not None:
            colours.append(_rgb_string(style.colour))
            sizes.append(style.radius * atom_scale)
        else:
            colours.append("rgb(128, 128, 128)")
            sizes.append(0.5 * atom_scale)

    atom_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=colours),
        name="atoms",
        hovertext=scene.species,
        hoverinfo="text",
    )

    # Bonds.
    bonds = compute_bonds(scene.species, coords, scene.bond_specs)
    if not bonds:
        return (atom_trace,)

    bond_x: list[float | None] = []
    bond_y: list[float | None] = []
    bond_z: list[float | None] = []
    for bond in bonds:
        a = coords[bond.index_a]
        b = coords[bond.index_b]
        bond_x.extend([a[0], b[0], None])
        bond_y.extend([a[1], b[1], None])
        bond_z.extend([a[2], b[2], None])

    # Use bond spec colour for the first bond's spec (simplification for v0.1).
    bond_colour = _rgb_string(bonds[0].spec.colour)
    bond_width = max(1.0, bonds[0].spec.radius * atom_scale)

    bond_trace = go.Scatter3d(
        x=bond_x,
        y=bond_y,
        z=bond_z,
        mode="lines",
        line=dict(color=bond_colour, width=bond_width),
        name="bonds",
        hoverinfo="skip",
    )

    return (atom_trace, bond_trace)


def render_plotly(
    scene: StructureScene,
    *,
    frame_index: int | None = None,
    atom_scale: float = 8.0,
    background: Colour = "white",
    width: int = 700,
    height: int = 700,
):
    """Render a StructureScene as an interactive plotly 3D figure.

    If the scene has multiple frames and ``frame_index`` is ``None``,
    renders an animated figure with a frame slider for trajectory
    playback. If ``frame_index`` is given, renders only that single
    frame.

    Args:
        scene: The StructureScene to render.
        frame_index: Specific frame to render. If ``None`` and multiple
            frames exist, creates an animation with slider.
        atom_scale: Scale factor for marker sizes.
        background: Background colour.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A plotly ``Figure`` object.

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for render_plotly(). "
            "Install it with: pip install plotly"
        )

    bg = _rgb_string(background)
    n_frames = len(scene.frames)
    single_frame = frame_index is not None or n_frames == 1
    idx = frame_index if frame_index is not None else 0

    if single_frame:
        traces = _build_traces(scene, idx, atom_scale)
        fig = go.Figure(data=traces)
    else:
        # Animated trajectory.
        initial_traces = _build_traces(scene, 0, atom_scale)
        fig = go.Figure(data=initial_traces)

        animation_frames = []
        for i in range(n_frames):
            frame_traces = _build_traces(scene, i, atom_scale)
            label = scene.frames[i].label or str(i)
            animation_frames.append(
                go.Frame(data=frame_traces, name=label)
            )
        fig.frames = animation_frames

        # Slider.
        sliders = [dict(
            active=0,
            steps=[
                dict(
                    args=[[f.name], dict(frame=dict(duration=100, redraw=True), mode="immediate")],
                    label=f.name,
                    method="animate",
                )
                for f in animation_frames
            ],
            currentvalue=dict(prefix="Frame: "),
        )]

        # Play/pause buttons.
        updatemenus = [dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                ),
            ],
        )]

        fig.update_layout(sliders=sliders, updatemenus=updatemenus)

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor=bg,
        ),
        width=width,
        height=height,
        title=scene.title or None,
        showlegend=False,
    )

    return fig
