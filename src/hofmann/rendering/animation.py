"""Animation rendering for multi-frame trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from matplotlib.figure import Figure
from PIL import Image


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
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        self._frames.append(Image.fromarray(arr).convert("RGBA"))

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
        )
