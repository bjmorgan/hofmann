"""Animation rendering for multi-frame trajectories."""

from __future__ import annotations

import copy
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from hofmann.model import (
    CmapSpec,
    Colour,
    RenderStyle,
    StructureScene,
    normalise_colour,
)
from hofmann.rendering.painter import _draw_scene, _precompute_scene
from hofmann.rendering.projection import _scene_extent
from hofmann.rendering.static import _resolve_style


class _FrameWriter(Protocol):
    """Protocol for animation frame writers."""

    def add_frame(self, fig: Figure) -> None: ...
    def finish(self) -> None: ...


class _GifWriter:
    """Write animation frames to a GIF file using Pillow.

    Frames are collected in memory and written on :meth:`finish`.

    Args:
        output: Destination file path.
        fps: Frames per second (converted to per-frame duration).
    """

    def __init__(self, output: str | Path, *, fps: int = 10) -> None:
        self._output = Path(output)
        self._duration = round(1000 / fps)
        self._frames: list[Image.Image] = []

    def add_frame(self, fig: Figure) -> None:
        """Capture the current figure as a GIF frame."""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        arr = np.asarray(buf)
        self._frames.append(Image.fromarray(arr).convert("RGB"))

    def finish(self) -> None:
        """Write all collected frames to the GIF file."""
        if not self._frames:
            return
        self._frames[0].save(
            self._output,
            save_all=True,
            append_images=self._frames[1:],
            duration=self._duration,
            loop=0,
            disposal=2,
        )


class _Mp4Writer:
    """Write animation frames to an MP4 file via ffmpeg.

    Frames are piped to an ``ffmpeg`` subprocess as raw RGBA data.

    Args:
        output: Destination file path.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Raises:
        FileNotFoundError: If ``ffmpeg`` is not on ``PATH``.
    """

    def __init__(
        self,
        output: str | Path,
        *,
        fps: int = 10,
        width: int,
        height: int,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError(
                "ffmpeg is required for MP4 output but was not found "
                "on PATH. Install it with 'brew install ffmpeg' "
                "(macOS), 'apt install ffmpeg' (Debian/Ubuntu), or "
                "'conda install ffmpeg' (conda)."
            )
        self._output = Path(output)
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(self._output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def add_frame(self, fig: Figure) -> None:
        """Pipe the current figure as a raw RGBA frame to ffmpeg."""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        assert self._proc.stdin is not None
        self._proc.stdin.write(bytes(buf))

    def finish(self) -> None:
        """Close the ffmpeg pipe and wait for the process to exit."""
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        self._proc.wait()


def _make_writer(
    output: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> _FrameWriter:
    """Select and construct a frame writer based on file extension."""
    ext = Path(output).suffix.lower()
    if ext == ".gif":
        return _GifWriter(output, fps=fps)
    elif ext == ".mp4":
        return _Mp4Writer(output, fps=fps, width=width, height=height)
    else:
        raise ValueError(
            f"Unsupported output format {ext!r}. "
            f"Supported: .gif, .mp4"
        )


def render_animation(
    scene: StructureScene,
    output: str | Path,
    *,
    style: RenderStyle | None = None,
    frames: range | Sequence[int] | None = None,
    fps: int = 10,
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

    Args:
        scene: The StructureScene to animate.
        output: Destination file path.  Extension determines the
            format: ``.gif`` for GIF (via Pillow), ``.mp4`` for
            MP4 (via ffmpeg).
        style: A :class:`RenderStyle` controlling visual appearance.
        frames: Which frame indices to render, in order.  ``None``
            renders all frames.  Accepts ``range(0, 100, 5)`` or
            an arbitrary sequence of indices.
        fps: Frames per second in the output file.
        figsize: Figure size in inches ``(width, height)``.
        dpi: Resolution in dots per inch.
        background: Background colour.
        colour_by: Key into ``scene.atom_data`` to colour atoms by.
        cmap: Matplotlib colourmap specification.
        colour_range: Explicit ``(vmin, vmax)`` for numerical data.
        **style_kwargs: Any :class:`RenderStyle` field name as a
            keyword argument.

    Returns:
        The output file path as a :class:`~pathlib.Path`.

    Raises:
        ValueError: If *frames* is empty, contains out-of-range
            indices, or the output extension is unsupported.
        FileNotFoundError: If the output is ``.mp4`` and ``ffmpeg``
            is not on ``PATH``.
    """
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

    view = copy.deepcopy(scene.view)
    bg_rgb = normalise_colour(background)

    width_px = int(figsize[0] * dpi)
    height_px = int(figsize[1] * dpi)
    writer = _make_writer(output, fps=fps, width=width_px, height=height_px)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.set_facecolor(bg_rgb)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Render the first frame to establish the viewport bounding box.
    # This auto-fits to the projected geometry (tight, aspect-aware).
    # All subsequent frames reuse these fixed limits.
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
    fixed_xlim = ax.get_xlim()
    fixed_ylim = ax.get_ylim()
    writer.add_frame(fig)

    for frame_idx in frame_indices[1:]:
        precomputed = _precompute_scene(
            scene, frame_idx, resolved,
            colour_by=colour_by, cmap=cmap,
            colour_range=colour_range,
        )
        _draw_scene(
            ax, scene, view, resolved,
            frame_index=frame_idx, bg_rgb=bg_rgb,
            precomputed=precomputed,
        )
        ax.set_xlim(*fixed_xlim)
        ax.set_ylim(*fixed_ylim)
        writer.add_frame(fig)

    writer.finish()
    plt.close(fig)
    return output
