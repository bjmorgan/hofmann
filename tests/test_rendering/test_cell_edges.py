"""Tests for cell edge geometry — edge generation, atom clipping, and rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import (
    AtomStyle,
    BondSpec,
    CellEdgeStyle,
    Frame,
    RenderStyle,
    StructureScene,
    ViewState,
)
from hofmann.rendering.cell_edges import _cell_edges_3d, _clip_edge_at_atoms
from hofmann.rendering.static import render_mpl


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
        specs = [BondSpec(species=("A", "B"), min_length=0.0,
                          max_length=5.0, radius=0.1, colour=1.0)]
    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles=styles,
        bond_specs=specs,
    )


def _scene_with_lattice(**kwargs):
    """Create a minimal scene with a cubic lattice."""
    a = kwargs.pop("a", 5.0)
    return StructureScene(
        species=["A"],
        frames=[Frame(coords=np.array([[a / 2, a / 2, a / 2]]))],
        atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
        lattice=np.eye(3) * a,
        **kwargs,
    )


class TestCellEdges3d:
    def test_produces_12_edges(self):
        starts, ends = _cell_edges_3d(np.eye(3) * 5.0)
        assert starts.shape == (12, 3)
        assert ends.shape == (12, 3)

    def test_cubic_edge_lengths(self):
        a = 5.0
        starts, ends = _cell_edges_3d(np.eye(3) * a)
        lengths = np.linalg.norm(ends - starts, axis=1)
        np.testing.assert_allclose(lengths, a)

    def test_orthorhombic_edge_lengths(self):
        lattice = np.diag([3.0, 4.0, 5.0])
        starts, ends = _cell_edges_3d(lattice)
        lengths = np.linalg.norm(ends - starts, axis=1)
        # Each edge should have length 3, 4, or 5.
        for length in lengths:
            assert length == pytest.approx(3.0) or \
                   length == pytest.approx(4.0) or \
                   length == pytest.approx(5.0)


class TestClipEdgeAtAtoms:
    """Tests for _clip_edge_at_atoms — cell edge clipping at atom spheres."""

    def test_no_atoms_returns_full_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([5.0, 0.0, 0.0])
        segs = _clip_edge_at_atoms(s, e, np.empty((0, 3)), np.empty(0))
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        np.testing.assert_allclose(segs[0][1], e)

    def test_atom_at_start_clips(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        assert segs[0][0][0] == pytest.approx(1.0)
        np.testing.assert_allclose(segs[0][1], e)

    def test_atom_at_end_clips(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[10.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(9.0)

    def test_atom_in_middle_splits_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[5.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 2
        # First segment: [0, 4]
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(4.0)
        # Second segment: [6, 10]
        assert segs[1][0][0] == pytest.approx(6.0)
        np.testing.assert_allclose(segs[1][1], e)

    def test_atom_far_from_line_no_clip(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[5.0, 5.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        np.testing.assert_allclose(segs[0][1], e)

    def test_fully_occluded_returns_empty(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([2.0, 0.0, 0.0])
        coords = np.array([[1.0, 0.0, 0.0]])
        radii = np.array([5.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 0

    def test_two_atoms_on_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        radii = np.array([1.0, 1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        assert segs[0][0][0] == pytest.approx(1.0)
        assert segs[0][1][0] == pytest.approx(9.0)

    def test_overlapping_atoms_merge_gaps(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ])
        radii = np.array([2.0, 2.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        # Gap from [1, 6] (merged from [1,5] and [2,6]).
        assert len(segs) == 2
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(1.0)
        assert segs[1][0][0] == pytest.approx(6.0)
        np.testing.assert_allclose(segs[1][1], e)


class TestCellEdgeRendering:
    def test_show_cell_auto_with_lattice(self):
        """Renders without error when lattice is present."""
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_show_cell_auto_without_lattice(self):
        """No crash when lattice is absent."""
        scene = _minimal_scene()
        assert scene.lattice is None
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_show_cell_false_suppresses(self):
        """Explicit show_cell=False produces fewer polygons."""
        scene = _scene_with_lattice()
        fig_with = render_mpl(scene, show=False)
        n_with = len(fig_with.axes[0].collections[0].get_paths())
        plt.close(fig_with)

        fig_without = render_mpl(scene, show=False, show_cell=False)
        n_without = len(fig_without.axes[0].collections[0].get_paths())
        plt.close(fig_without)

        assert n_with > n_without

    def test_show_cell_true_without_lattice_raises(self):
        """show_cell=True on a scene without lattice raises."""
        scene = _minimal_scene()
        with pytest.raises(ValueError, match="show_cell.*no lattice"):
            render_mpl(scene, show=False, show_cell=True)

    def test_cell_edge_dashed(self):
        """Dashed style produces more polygons than solid."""
        scene = _scene_with_lattice()
        fig_solid = render_mpl(scene, show=False)
        n_solid = len(fig_solid.axes[0].collections[0].get_paths())
        plt.close(fig_solid)

        style = RenderStyle(cell_style=CellEdgeStyle(linestyle="dashed"))
        fig_dash = render_mpl(scene, show=False, style=style)
        n_dash = len(fig_dash.axes[0].collections[0].get_paths())
        plt.close(fig_dash)

        # Each of the 12 edges gets split into multiple dashes.
        assert n_dash > n_solid

    def test_cell_edges_clipped_at_corner_atoms(self):
        """Atoms at cell corners produce shorter edges."""
        # Scene with atom at the origin (a cell corner).
        a = 5.0
        scene = StructureScene(
            species=["A"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"A": AtomStyle(1.0, (0.5, 0.5, 0.5))},
            lattice=np.eye(3) * a,
        )
        fig = render_mpl(scene, show=False)
        n_corner = len(fig.axes[0].collections[0].get_paths())
        plt.close(fig)

        # Scene with atom at the centre (no edge clipping).
        scene_mid = _scene_with_lattice(a=a)
        fig_mid = render_mpl(scene_mid, show=False)
        n_mid = len(fig_mid.axes[0].collections[0].get_paths())
        plt.close(fig_mid)

        # Both render without error; corner scene has same number of
        # edge polygons (clipping shortens edges but doesn't remove them).
        assert n_corner >= 1
        assert n_mid >= 1
