"""Interactive matplotlib viewer with mouse and keyboard controls."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from hofmann.model import (
    CmapSpec,
    Colour,
    RenderStyle,
    StructureScene,
    ViewState,
    normalise_colour,
)
from hofmann.rendering.painter import _draw_scene, _precompute_scene
from hofmann.rendering.projection import _scene_extent
from hofmann.rendering.static import _resolve_style


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix about the X axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ])


def _rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix about the Y axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  0.0,  s],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c],
    ])


def _rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix about the Z axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Interactive renderer — constants
# ---------------------------------------------------------------------------

_KEY_ROTATION_STEP = 0.05  # radians (~3 degrees) per key press
_KEY_ZOOM_FACTOR = 1.1  # multiplicative zoom per key press / scroll step
_KEY_PAN_FRACTION = 0.05  # fraction of scene extent per key press
_PERSPECTIVE_STEP = 0.1  # perspective increment per key press
_DISTANCE_FACTOR = 1.05  # viewing distance multiplier per key press

_HELP_TEXT = """\
Arrows     Rotate          Shift+Arrows  Pan
,  .       Roll            +  =  -       Zoom
p  P       Perspective     d  D          Distance
b          Bonds           o             Outlines
e          Polyhedra       u             Unit cell
a          Axes            r             Reset view
[  ]       Prev/next frame {  }          First/last
h          Toggle help     Scroll        Zoom
Drag       Rotate"""


def _apply_key_action(
    key: str,
    view: "ViewState",
    style: "RenderStyle",
    state: dict,
    *,
    n_frames: int,
    base_extent: float,
    initial_view: dict,
    has_lattice: bool = False,
) -> str:
    """Apply a keyboard action, mutating *view*, *style*, and *state*.

    Returns a string indicating the required redraw kind:

    - ``"view"`` — view-only change (rotation, zoom, pan, etc.).
    - ``"full"`` — style or frame change needing recomputation.
    - ``"none"`` — unrecognised key, no redraw needed.
    """
    # -- Rotation --
    if key == "left":
        view.rotation = _rotation_y(-_KEY_ROTATION_STEP) @ view.rotation
    elif key == "right":
        view.rotation = _rotation_y(_KEY_ROTATION_STEP) @ view.rotation
    elif key == "up":
        view.rotation = _rotation_x(-_KEY_ROTATION_STEP) @ view.rotation
    elif key == "down":
        view.rotation = _rotation_x(_KEY_ROTATION_STEP) @ view.rotation
    elif key == ",":
        view.rotation = _rotation_z(_KEY_ROTATION_STEP) @ view.rotation
    elif key == ".":
        view.rotation = _rotation_z(-_KEY_ROTATION_STEP) @ view.rotation

    # -- Zoom --
    elif key in ("+", "="):
        view.zoom = min(100.0, view.zoom * _KEY_ZOOM_FACTOR)
    elif key == "-":
        view.zoom = max(0.01, view.zoom / _KEY_ZOOM_FACTOR)

    # -- Pan (shift + arrows) --
    # Moving the centre in screen-right shifts the *camera* right,
    # so the scene appears to move left.  Negate so that the arrow
    # direction matches the apparent scene movement.
    elif key == "shift+left":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre + step * view.rotation[0]
    elif key == "shift+right":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre - step * view.rotation[0]
    elif key == "shift+down":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre + step * view.rotation[1]
    elif key == "shift+up":
        step = _KEY_PAN_FRACTION * base_extent / view.zoom
        view.centre = view.centre - step * view.rotation[1]

    # -- Perspective / distance --
    elif key == "p":
        view.perspective = min(1.0, view.perspective + _PERSPECTIVE_STEP)
    elif key == "P":
        view.perspective = max(0.0, view.perspective - _PERSPECTIVE_STEP)
    elif key == "d":
        view.view_distance *= _DISTANCE_FACTOR
    elif key == "D":
        view.view_distance = max(0.1, view.view_distance / _DISTANCE_FACTOR)

    # -- Style toggles (no recomputation needed) --
    elif key == "b":
        style.show_bonds = not style.show_bonds
    elif key == "o":
        style.show_outlines = not style.show_outlines
    elif key == "e":
        style.show_polyhedra = not style.show_polyhedra
    elif key == "u":
        # Resolve None (auto-detect) to the effective value before toggling.
        effective = style.show_cell if style.show_cell is not None else has_lattice
        style.show_cell = not effective
    elif key == "a":
        effective = style.show_axes if style.show_axes is not None else has_lattice
        style.show_axes = not effective

    # -- Frame navigation --
    elif key == "]" and n_frames > 1:
        state["frame_index"] = (state["frame_index"] + 1) % n_frames
        return "full"
    elif key == "[" and n_frames > 1:
        state["frame_index"] = (state["frame_index"] - 1) % n_frames
        return "full"
    elif key == "}" and n_frames > 1:
        state["frame_index"] = n_frames - 1
        return "full"
    elif key == "{" and n_frames > 1:
        state["frame_index"] = 0
        return "full"

    # -- Reset --
    elif key == "r":
        view.rotation = initial_view["rotation"].copy()
        view.zoom = initial_view["zoom"]
        view.centre = initial_view["centre"].copy()
        view.perspective = initial_view["perspective"]
        view.view_distance = initial_view["view_distance"]

    # -- Help overlay --
    elif key == "h":
        state["help_visible"] = not state["help_visible"]

    else:
        return "none"

    return "view"


def render_mpl_interactive(
    scene: StructureScene,
    *,
    style: RenderStyle | None = None,
    frame_index: int = 0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
    **style_kwargs: object,
) -> tuple["ViewState", "RenderStyle"]:
    """Interactive matplotlib viewer with mouse and keyboard controls.

    Opens a matplotlib window where the user can manipulate the view
    with the mouse and keyboard:

    **Mouse:**

    - **Left-drag** to rotate the structure.
    - **Scroll** to zoom in/out.

    **Keyboard:**

    - **Arrow keys** rotate around the horizontal/vertical axes.
    - **,** / **.** roll in the screen plane.
    - **+** / **=** / **-** zoom in/out.
    - **Shift+Arrow** keys pan the view.
    - **p** / **P** increase/decrease perspective strength.
    - **d** / **D** increase/decrease viewing distance.
    - **b** toggle bonds, **o** toggle outlines, **e** toggle polyhedra,
      **u** toggle unit cell, **a** toggle axes widget.
    - **[** / **]** step to the previous/next frame;
      **{** / **}** jump to the first/last frame.
    - **r** reset the view to its initial state.
    - **h** toggle a help overlay listing all keybindings.

    When the window is closed the updated :class:`ViewState` and
    :class:`RenderStyle` are returned, allowing the user to re-use
    both for static rendering::

        view, style = scene.render_mpl_interactive()
        scene.view = view
        scene.render_mpl("output.svg", style=style)

    Args:
        scene: The StructureScene to render.
        style: A :class:`RenderStyle` controlling visual appearance.
            Any :class:`RenderStyle` field name may also be passed as
            a keyword argument to override individual fields.
        frame_index: Which frame to render initially.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution.
        background: Background colour.
        colour_by: Key into ``scene.atom_data`` to colour atoms by.
        cmap: Matplotlib colourmap name, object, or callable.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
        **style_kwargs: Any :class:`RenderStyle` field name as a
            keyword argument.  Unknown names raise :class:`TypeError`.

    Returns:
        A ``(ViewState, RenderStyle)`` tuple reflecting any view and
        style changes applied during the interactive session.
    """
    resolved = _resolve_style(style, **style_kwargs)

    n_frames = len(scene.frames)
    if not 0 <= frame_index < n_frames:
        raise ValueError(
            f"frame_index {frame_index} out of range for scene "
            f"with {n_frames} frame(s)"
        )

    bg_rgb = normalise_colour(background)

    # Use lower-fidelity polygon counts for interactive responsiveness.
    # Save the static values so we can restore them before returning.
    static_circle_segments = resolved.circle_segments
    static_arc_segments = resolved.arc_segments
    resolved.circle_segments = resolved.interactive_circle_segments
    resolved.arc_segments = resolved.interactive_arc_segments

    # Work on a copy so we don't mutate the original scene's view.
    view = ViewState(
        rotation=scene.view.rotation.copy(),
        zoom=scene.view.zoom,
        centre=scene.view.centre.copy(),
        perspective=scene.view.perspective,
        view_distance=scene.view.view_distance,
        slab_origin=(
            scene.view.slab_origin.copy()
            if scene.view.slab_origin is not None else None
        ),
        slab_near=scene.view.slab_near,
        slab_far=scene.view.slab_far,
    )

    # Fixed viewport extent — rotation-invariant so the scene doesn't
    # appear to shift or rescale while dragging.
    base_extent = _scene_extent(scene, view, frame_index, resolved.atom_scale)

    # Pre-compute bonds, colours, adjacency once — these don't change
    # during interactive rotation / zoom.
    colour_kwargs: dict[str, Any] = dict(
        colour_by=colour_by,
        cmap=cmap,
        colour_range=colour_range,
    )
    pre = _precompute_scene(scene, frame_index, resolved, **colour_kwargs)

    draw_kwargs: dict[str, Any] = dict(
        frame_index=frame_index,
        bg_rgb=bg_rgb,
        precomputed=pre,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _draw_scene(ax, scene, view, resolved, viewport_extent=base_extent, **draw_kwargs)

    # ---- Interaction state ----

    state: dict = {
        "drag_active": False,
        "drag_last_xy": None,
        "last_draw_t": 0.0,
        "frame_index": frame_index,
        "help_visible": False,
        "precomputed": pre,
        "base_extent": base_extent,
    }

    # Snapshot for the reset key.
    initial_view = {
        "rotation": view.rotation.copy(),
        "zoom": view.zoom,
        "centre": view.centre.copy(),
        "perspective": view.perspective,
        "view_distance": view.view_distance,
    }

    _DRAG_SENSITIVITY = 0.01  # radians per pixel
    _MIN_INTERVAL = 0.03  # seconds between redraws (~30 fps cap)

    # ---- Redraw helpers ----

    def _redraw() -> None:
        """Repaint the scene using the fixed viewport extent."""
        _draw_scene(
            ax, scene, view, resolved,
            viewport_extent=state["base_extent"], **draw_kwargs,
        )
        if state["help_visible"]:
            _add_help_overlay()
        fig.canvas.draw_idle()
        state["last_draw_t"] = time.monotonic()

    def _throttled_redraw() -> None:
        """Redraw only if enough time has elapsed since the last draw."""
        if time.monotonic() - state["last_draw_t"] >= _MIN_INTERVAL:
            _redraw()

    def _full_redraw() -> None:
        """Recompute bonds/colours and repaint (for frame or style changes)."""
        state["precomputed"] = _precompute_scene(
            scene, state["frame_index"], resolved, **colour_kwargs,
        )
        draw_kwargs["precomputed"] = state["precomputed"]
        draw_kwargs["frame_index"] = state["frame_index"]
        state["base_extent"] = _scene_extent(
            scene, view, state["frame_index"], resolved.atom_scale,
        )
        _redraw()

    def _add_help_overlay() -> None:
        """Add the keybinding help text to the axes."""
        ax.text(
            0.02, 0.98, _HELP_TEXT,
            transform=ax.transAxes,
            fontsize=7,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="grey",
            ),
            zorder=1000,
        )

    # ---- Mouse handlers ----

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        state["drag_active"] = True
        state["drag_last_xy"] = (event.x, event.y)

    def on_motion(event):
        if not state["drag_active"] or state["drag_last_xy"] is None:
            return
        x0, y0 = state["drag_last_xy"]
        dx = event.x - x0
        dy = event.y - y0
        state["drag_last_xy"] = (event.x, event.y)
        # Incremental rotation in screen space: horizontal drag rotates
        # around the screen Y axis, vertical drag around screen X.
        # Applying to the *current* rotation gives intuitive "grab and
        # drag the object" behaviour regardless of accumulated rotation.
        view.rotation = (
            _rotation_y(dx * _DRAG_SENSITIVITY)
            @ _rotation_x(-dy * _DRAG_SENSITIVITY)
            @ view.rotation
        )
        _throttled_redraw()

    def on_release(event):
        if state["drag_active"]:
            state["drag_active"] = False
            # Final redraw to ensure the last position is rendered.
            _redraw()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        factor = _KEY_ZOOM_FACTOR ** event.step
        view.zoom = max(0.01, min(100.0, view.zoom * factor))
        _redraw()

    # ---- Keyboard handler ----

    def on_key_press(event):
        if event.key is None:
            return
        kind = _apply_key_action(
            event.key, view, resolved, state,
            n_frames=len(scene.frames),
            base_extent=state["base_extent"],
            initial_view=initial_view,
            has_lattice=scene.lattice is not None,
        )
        if kind == "full":
            _full_redraw()
        elif kind == "view":
            _throttled_redraw()

    # ---- Connect events ----

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # Disconnect matplotlib's default key handler to avoid conflicts
    # (e.g. 'p' for pan tool, 'o' for zoom-to-rect).
    manager = fig.canvas.manager
    if manager is not None:
        handler_id = getattr(manager, "key_press_handler_id", None)
        if handler_id is not None:
            fig.canvas.mpl_disconnect(handler_id)

    plt.show()

    # Restore static-quality segment counts so the returned style
    # is ready for publication rendering.
    resolved.circle_segments = static_circle_segments
    resolved.arc_segments = static_arc_segments

    return view, resolved
