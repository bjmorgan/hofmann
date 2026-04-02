"""Tests for animation rendering."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

imageio = pytest.importorskip("imageio")

from hofmann.model import Frame, StructureScene


def _make_scene(n_frames: int = 5) -> StructureScene:
    """Create a minimal multi-frame scene for testing.

    Uses 4 atoms with per-frame y-displacements large enough to
    produce visibly distinct frames even at low DPI.
    """
    from hofmann import AtomStyle

    species = ["C", "C", "C", "C"]
    frames = []
    for i in range(n_frames):
        dy = 0.3 * i
        coords = np.array([
            [0.0, dy, 0.0],
            [2.0, -dy, 0.0],
            [4.0, dy, 0.0],
            [6.0, -dy, 0.0],
        ])
        frames.append(Frame(coords=coords))
    return StructureScene(
        species=species,
        frames=frames,
        atom_styles={"C": AtomStyle(radius=1.0, colour="black")},
    )


class TestRenderAnimation:
    def test_gif_all_frames(self, tmp_path):
        """Rendering all frames produces a GIF with correct frame count."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        output = tmp_path / "traj.gif"
        result = render_animation(scene, output, dpi=100, figsize=(4, 4))

        assert result == output
        assert output.exists()
        assert len(imageio.mimread(output)) == 5

    def test_gif_frame_selection(self, tmp_path):
        """frames parameter selects a subset."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=10)
        output = tmp_path / "subset.gif"
        render_animation(
            scene, output, frames=range(0, 10, 2),
            dpi=100, figsize=(4, 4),
        )

        assert len(imageio.mimread(output)) == 5

    def test_mp4_produces_file(self, tmp_path):
        """MP4 output produces a non-empty file."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=3)
        output = tmp_path / "traj.mp4"
        result = render_animation(scene, output, dpi=100, figsize=(4, 4))

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_raises_on_out_of_range_frames(self, tmp_path):
        """ValueError if frame indices are out of range."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="out of range"):
            render_animation(scene, tmp_path / "bad.gif", frames=[10])

    def test_raises_on_negative_frame_index(self, tmp_path):
        """ValueError if frame indices are negative."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="out of range"):
            render_animation(scene, tmp_path / "bad.gif", frames=[-1])

    def test_raises_on_unsupported_extension(self, tmp_path):
        """ValueError if output file has an unsupported extension."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=3)
        with pytest.raises(ValueError, match="unsupported output format"):
            render_animation(scene, tmp_path / "bad.avi")

    def test_raises_on_invalid_fps(self, tmp_path):
        """ValueError if fps is zero or negative."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=3)
        for bad_fps in (0, -5):
            with pytest.raises(ValueError, match="fps"):
                render_animation(scene, tmp_path / "bad.gif", fps=bad_fps)

    def test_raises_on_empty_frames(self, tmp_path):
        """ValueError if frames is empty."""
        from hofmann.rendering.animation import render_animation

        scene = _make_scene(n_frames=5)
        with pytest.raises(ValueError, match="empty"):
            render_animation(scene, tmp_path / "bad.gif", frames=[])

    def test_partial_output_removed_on_failure(self, tmp_path):
        """Partial output file is removed when rendering fails mid-render."""
        from unittest.mock import patch

        from hofmann.rendering.animation import render_animation
        from hofmann.rendering.painter import _precompute_scene

        scene = _make_scene(n_frames=3)
        output = tmp_path / "fail.gif"

        call_count = 0
        real_precompute = _precompute_scene

        def _fail_on_second_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("synthetic error")
            return real_precompute(*args, **kwargs)

        with patch(
            "hofmann.rendering.animation._precompute_scene",
            side_effect=_fail_on_second_call,
        ):
            with pytest.raises(ValueError, match="synthetic error"):
                render_animation(scene, output)

        assert not output.exists()

    def test_scene_method_delegates(self, tmp_path):
        """StructureScene.render_animation() produces output."""
        scene = _make_scene(n_frames=3)
        output = tmp_path / "method.gif"
        result = scene.render_animation(output, dpi=100, figsize=(4, 4))
        assert result == output
        assert output.exists()
