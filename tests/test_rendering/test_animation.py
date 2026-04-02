"""Tests for animation rendering."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shutil
from PIL import Image


class TestGifWriter:
    def test_produces_valid_gif(self, tmp_path):
        """GifWriter produces a valid GIF with the correct frame count."""
        from hofmann.rendering.animation import _GifWriter

        output = tmp_path / "test.gif"
        writer = _GifWriter(output, fps=10)

        for i in range(3):
            fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
            ax.plot([0, i + 1], [0, 1])
            fig.canvas.draw()
            writer.add_frame(fig)
            plt.close(fig)

        writer.finish()

        assert output.exists()
        with Image.open(output) as img:
            assert img.format == "GIF"
            n = 0
            try:
                while True:
                    n += 1
                    img.seek(n)
            except EOFError:
                pass
            assert n == 3

    def test_single_frame_gif(self, tmp_path):
        """A single-frame GIF is valid."""
        from hofmann.rendering.animation import _GifWriter

        output = tmp_path / "single.gif"
        writer = _GifWriter(output, fps=5)
        fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        writer.add_frame(fig)
        plt.close(fig)
        writer.finish()

        assert output.exists()
        with Image.open(output) as img:
            assert img.format == "GIF"


class TestMp4Writer:
    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not installed",
    )
    def test_produces_valid_mp4(self, tmp_path):
        """Mp4Writer produces an MP4 file."""
        from hofmann.rendering.animation import _Mp4Writer

        output = tmp_path / "test.mp4"
        fig, ax = plt.subplots(figsize=(2, 2), dpi=50)
        writer = _Mp4Writer(output, fps=10, width=100, height=100)

        for _ in range(3):
            ax.clear()
            ax.plot([0, 1], [0, 1])
            fig.canvas.draw()
            writer.add_frame(fig)

        writer.finish()
        plt.close(fig)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_raises_without_ffmpeg(self, tmp_path, monkeypatch):
        """Mp4Writer raises if ffmpeg is not found."""
        from hofmann.rendering.animation import _Mp4Writer

        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(FileNotFoundError, match="ffmpeg"):
            _Mp4Writer(tmp_path / "test.mp4", fps=10, width=100, height=100)


from hofmann.model import Frame, StructureScene


def _make_scene(n_frames: int = 5) -> StructureScene:
    """Create a minimal multi-frame scene for testing.

    Atom positions vary slightly in x/y so that each frame renders
    with distinct pixels (Pillow optimises away identical GIF frames).
    Displacements are kept small to stay within the fixed viewport.
    """
    species = ["H", "H"]
    frames = []
    for i in range(n_frames):
        dx = 0.1 * i
        coords = np.array([[0.0, dx, 0.0], [1.0, -dx, 0.0]])
        frames.append(Frame(coords=coords))
    return StructureScene(species=species, frames=frames)


class TestRenderAnimation:
    def test_gif_all_frames(self, tmp_path):
        """Rendering all frames produces a GIF with correct frame count."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        output = tmp_path / "traj.gif"
        result = render_animation(scene, output, dpi=50, figsize=(2, 2))

        assert result == output
        assert output.exists()
        with Image.open(output) as img:
            n = 0
            try:
                while True:
                    n += 1
                    img.seek(n)
            except EOFError:
                pass
            assert n == 5

    def test_gif_frame_selection(self, tmp_path):
        """frames parameter selects a subset."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=10)
        output = tmp_path / "subset.gif"
        render_animation(
            scene, output, frames=range(0, 10, 2),
            dpi=50, figsize=(2, 2),
        )

        with Image.open(output) as img:
            n = 0
            try:
                while True:
                    n += 1
                    img.seek(n)
            except EOFError:
                pass
            assert n == 5

    def test_raises_on_out_of_range_frames(self, tmp_path):
        """ValueError if frame indices are out of range."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="out of range"):
            render_animation(scene, tmp_path / "bad.gif", frames=[10])

    def test_raises_on_empty_frames(self, tmp_path):
        """ValueError if frames is empty."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="empty"):
            render_animation(scene, tmp_path / "bad.gif", frames=[])

    def test_raises_on_unsupported_extension(self, tmp_path):
        """ValueError for unsupported output extensions."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="Unsupported"):
            render_animation(scene, tmp_path / "bad.avi")
