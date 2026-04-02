"""Animation rendering for multi-frame trajectories."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hofmann.model import (
    CmapSpec,
    Colour,
    RenderStyle,
    StructureScene,
    normalise_colour,
)
from hofmann.rendering.painter import _draw_scene, _precompute_scene
from hofmann.rendering.static import _resolve_style


def render_animation(
    scene: StructureScene,
    output: str | Path,
    *,
    style: RenderStyle | None = None,
    frames: range | Sequence[int] | None = None,
    fps: int = 30,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 150,
    background: Colour = "white",
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: (
        tuple[float, float]
        | None
        | list[tuple[float, float] | None]
    ) = None,
    **style_kwargs: object,
) -> Path:
    """Render a trajectory animation to a GIF or MP4 file.

    Loops over the specified frames, rendering each with the
    existing per-frame pipeline and writing it to the output file.

    Requires the ``imageio`` package.  For MP4 output,
    ``imageio-ffmpeg`` is also required.  Install both with::

        pip install imageio imageio-ffmpeg

    Args:
        scene: The StructureScene to animate.
        output: Destination file path.  Extension determines the
            format (e.g. ``.gif``, ``.mp4``).
        style: A :class:`RenderStyle` controlling visual appearance.
        frames: Which frame indices to render, in order.  ``None``
            renders all frames.  Accepts ``range(0, 100, 5)`` or
            an arbitrary sequence of indices.
        fps: Frames per second in the output file.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution in dots per inch.
        background: Background colour.
        colour_by: Key (or list of keys) into ``scene.atom_data``
            to colour atoms by.
        cmap: Matplotlib colourmap specification.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
        **style_kwargs: Any :class:`RenderStyle` field name as a
            keyword argument.

    Returns:
        The output file path as a :class:`~pathlib.Path`.

    Raises:
        ValueError: If *frames* is empty or contains out-of-range
            indices, or if *fps* is less than 1.
        ImportError: If ``imageio`` is not installed.
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for animation rendering. "
            "Install it with: pip install imageio imageio-ffmpeg"
        ) from exc

    output = Path(output)
    resolved = _resolve_style(style, **style_kwargs)

    n_scene_frames = len(scene.frames)
    if frames is None:
        frame_indices = list(range(n_scene_frames))
    else:
        frame_indices = list(frames)

    if not frame_indices:
        raise ValueError("frames must not be empty")

    for idx in frame_indices:
        if not 0 <= idx < n_scene_frames:
            raise ValueError(
                f"frame index {idx} out of range for scene "
                f"with {n_scene_frames} frame(s)"
            )

    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    view = copy.deepcopy(scene.view)
    bg_rgb = normalise_colour(background)

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    succeeded = False
    try:
        ext = output.suffix.lower()
        writer_kwargs: dict = {}
        if ext == ".gif":
            writer_kwargs["duration"] = round(1000 / fps)
            writer_kwargs["loop"] = 0
        else:
            writer_kwargs["fps"] = fps
            writer_kwargs["macro_block_size"] = 1

        with imageio.get_writer(output, **writer_kwargs) as writer:
            # Render the first frame to establish the viewport bounding
            # box.  This auto-fits to the projected geometry (tight,
            # aspect-aware).  All subsequent frames reuse these limits.
            first_pre = _precompute_scene(
                scene, frame_indices[0], resolved,
                colour_by=colour_by, cmap=cmap,
                colour_range=colour_range,
            )
            _draw_scene(
                ax, scene, view, resolved,
                frame_index=frame_indices[0], bg_rgb=bg_rgb,
                precomputed=first_pre,
            )
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])

            for frame_idx in frame_indices[1:]:
                try:
                    precomputed = _precompute_scene(
                        scene, frame_idx, resolved,
                        colour_by=colour_by, cmap=cmap,
                        colour_range=colour_range,
                    )
                    _draw_scene(
                        ax, scene, view, resolved,
                        frame_index=frame_idx, bg_rgb=bg_rgb,
                        precomputed=precomputed,
                        fixed_xlim=xlim, fixed_ylim=ylim,
                    )
                except Exception as exc:
                    raise type(exc)(
                        f"error rendering frame {frame_idx}: {exc}"
                    ) from exc
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        succeeded = True
    finally:
        fig.clear()
        if not succeeded and output.exists():
            output.unlink()

    return output
