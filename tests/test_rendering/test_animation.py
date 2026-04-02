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
