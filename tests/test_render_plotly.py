"""Tests for hofmann.render_plotly — interactive plotly 3D renderer."""

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from hofmann.model import AtomStyle, BondSpec, Frame, StructureScene
from hofmann.render_plotly import render_plotly
from hofmann.scene import from_xbs


def _minimal_scene(n_frames=1, with_bonds=True):
    """Create a minimal scene for testing."""
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frames = [Frame(coords=coords + i * 0.1, label=f"f{i}") for i in range(n_frames)]
    specs = []
    if with_bonds:
        specs = [BondSpec("A", "B", 0.0, 5.0, 0.1, 1.0)]
    return StructureScene(
        species=["A", "B"],
        frames=frames,
        atom_styles={
            "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
            "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
        },
        bond_specs=specs,
    )


class TestRenderPlotly:
    def test_returns_figure(self):
        scene = _minimal_scene()
        fig = render_plotly(scene)
        assert isinstance(fig, go.Figure)

    def test_has_traces_with_bonds(self):
        scene = _minimal_scene()
        fig = render_plotly(scene)
        assert len(fig.data) >= 2  # Atoms + bonds.

    def test_single_frame_no_animation(self):
        scene = _minimal_scene(n_frames=1)
        fig = render_plotly(scene)
        assert len(fig.frames) == 0

    def test_trajectory_has_frames(self):
        scene = _minimal_scene(n_frames=3)
        fig = render_plotly(scene)
        assert len(fig.frames) == 3

    def test_no_bonds(self):
        scene = _minimal_scene(with_bonds=False)
        fig = render_plotly(scene)
        assert len(fig.data) == 1  # Only atoms trace.

    def test_specific_frame_index(self):
        scene = _minimal_scene(n_frames=3)
        fig = render_plotly(scene, frame_index=2)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 0  # No animation when frame_index given.

    def test_ch4_renders(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = render_plotly(scene)
        assert isinstance(fig, go.Figure)

    def test_convenience_method(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = scene.render_plotly()
        assert isinstance(fig, go.Figure)
