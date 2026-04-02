"""Tests for animation rendering."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
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
