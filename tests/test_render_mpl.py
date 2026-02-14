"""Tests for hofmann.render_mpl — depth-sorted matplotlib renderer."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import AtomStyle, BondSpec, Frame, StructureScene
from hofmann.render_mpl import render_mpl
from hofmann.scene import from_xbs


def _minimal_scene(n_atoms=2, with_bonds=True):
    """Create a minimal scene for testing."""
    species = ["A", "B"][:n_atoms]
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])[:n_atoms]
    styles = {
        "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
        "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
    }
    specs = []
    if with_bonds and n_atoms >= 2:
        specs = [BondSpec("A", "B", 0.0, 5.0, 0.1, 1.0)]
    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles=styles,
        bond_specs=specs,
    )


class TestRenderMpl:
    def test_returns_figure(self):
        scene = _minimal_scene()
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_path):
        scene = _minimal_scene()
        out = tmp_path / "test.png"
        render_mpl(scene, output=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_ch4_renders(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_no_bonds(self):
        scene = _minimal_scene(with_bonds=False)
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_custom_frame_index(self):
        coords1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords1), Frame(coords=coords2)],
            atom_styles={"A": AtomStyle(1.0, "grey"), "B": AtomStyle(0.8, "red")},
        )
        fig = render_mpl(scene, frame_index=1, show=False)
        assert isinstance(fig, Figure)

    def test_convenience_method(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = scene.render_mpl(show=False)
        assert isinstance(fig, Figure)

    def test_saves_svg(self, tmp_path, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        out = tmp_path / "ch4.svg"
        scene.render_mpl(output=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0
