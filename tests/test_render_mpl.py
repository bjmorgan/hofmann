"""Tests for hofmann.render_mpl — depth-sorted matplotlib renderer."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import AtomStyle, BondSpec, Frame, StructureScene, ViewState
from hofmann.render_mpl import _clip_bond_3d, _project_point, _stick_polygon, render_mpl
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


# --- Geometry helpers ---


class TestClipBond3d:
    def test_symmetric_clipping(self):
        """Two identical atoms should clip symmetrically."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([4.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        # Offset from each atom: sqrt(1.0 - 0.01) ~ 0.99499
        w = np.sqrt(1.0 - 0.01)
        np.testing.assert_allclose(clip_start, [w, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(clip_end, [4.0 - w, 0.0, 0.0], atol=1e-10)

    def test_overlapping_atoms_returns_none(self):
        """If atoms overlap so much that clipping leaves nothing, return None."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([1.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is None

    def test_coincident_atoms_returns_none(self):
        p_a = np.array([0.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_a.copy(), r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is None

    def test_bond_radius_larger_than_atom(self):
        """When bond radius >= atom radius, offset is zero (no clipping)."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([4.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=0.05, r_b=0.05, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        np.testing.assert_allclose(clip_start, p_a)
        np.testing.assert_allclose(clip_end, p_b)

    def test_diagonal_bond(self):
        """Clipping should work along any direction, not just axes."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([3.0, 4.0, 0.0])  # length = 5
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        w = np.sqrt(1.0 - 0.01)
        unit = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(clip_start, unit * w, atol=1e-10)
        np.testing.assert_allclose(clip_end, p_b - unit * w, atol=1e-10)


class TestStickPolygon:
    def test_horizontal_bond(self):
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        verts = _stick_polygon(start, end, hw_start=0.5, hw_end=0.5)
        assert verts is not None
        assert verts.shape == (4, 2)

    def test_tapered_bond(self):
        """Different half-widths should produce a trapezoid."""
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        verts = _stick_polygon(start, end, hw_start=0.5, hw_end=0.3)
        assert verts is not None
        # Start end should be wider than end
        start_width = np.linalg.norm(verts[0] - verts[1])
        end_width = np.linalg.norm(verts[2] - verts[3])
        assert start_width > end_width

    def test_zero_length_returns_none(self):
        start = np.array([1.0, 1.0])
        verts = _stick_polygon(start, start.copy(), hw_start=0.5, hw_end=0.5)
        assert verts is None


class TestProjectPoint:
    def test_orthographic(self):
        view = ViewState()
        pt = np.array([1.0, 2.0, 3.0])
        xy, s = _project_point(pt, view)
        np.testing.assert_allclose(xy, [1.0, 2.0])
        assert s == 1.0

    def test_perspective(self):
        view = ViewState(perspective=1.0, view_distance=10.0)
        pt = np.array([1.0, 0.0, 0.0])  # depth 0 -> scale = 1
        xy, s = _project_point(pt, view)
        np.testing.assert_allclose(s, 1.0)
        np.testing.assert_allclose(xy, [1.0, 0.0])
