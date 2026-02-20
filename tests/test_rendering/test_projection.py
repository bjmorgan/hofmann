"""Tests for projection helpers — _project_point and _scene_extent."""

import numpy as np

from hofmann.model import AtomStyle, Frame, StructureScene, ViewState
from hofmann.rendering.projection import _project_point, _scene_extent


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


class TestSceneExtent:
    """Tests for _scene_extent viewport calculation."""

    def test_perspective_increases_extent(self):
        """With perspective enabled, the extent should be larger to
        account for near-camera magnification."""
        scene = StructureScene(
            species=["C", "C"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 5.0],
            ]))],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        view_no_persp = ViewState(perspective=0.0)
        view_persp = ViewState(perspective=0.5)
        e_no = _scene_extent(scene, view_no_persp, 0, atom_scale=0.5)
        e_yes = _scene_extent(scene, view_persp, 0, atom_scale=0.5)
        assert e_yes > e_no

    def test_lattice_extends_extent(self):
        """With a lattice, extent includes cell corners."""
        scene = StructureScene(
            species=["A"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
        )
        view = ViewState()
        e_no_lat = _scene_extent(scene, view, 0, atom_scale=0.5)

        scene_lat = StructureScene(
            species=["A"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
            lattice=np.eye(3) * 10.0,
        )
        e_lat = _scene_extent(scene_lat, view, 0, atom_scale=0.5)
        assert e_lat > e_no_lat
