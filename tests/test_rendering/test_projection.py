"""Tests for projection helpers — _project_point and _scene_extent."""

import math

import numpy as np

from hofmann.model import AtomStyle, Frame, StructureScene, ViewState
from hofmann.model.composition import Composition
from hofmann.rendering.projection import (
    _make_vacancy_wedge,
    _make_wedges,
    _project_point,
    _scene_extent,
)


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

    def test_empty_scene(self):
        """An empty scene (zero atoms) should return a positive extent."""
        scene = StructureScene(
            species=[],
            frames=[Frame(coords=np.empty((0, 3)))],
            atom_styles={},
        )
        view = ViewState()
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        assert extent > 0

    def test_empty_scene_with_lattice(self):
        """An empty scene with a lattice uses cell corners for extent."""
        scene = StructureScene(
            species=[],
            frames=[Frame(
                coords=np.empty((0, 3)),
                lattice=np.eye(3) * 10.0,
            )],
            atom_styles={},
        )
        view = ViewState()
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        # Should reach at least to the far corner of the cell.
        assert extent > 10.0

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
            frames=[Frame(
                coords=np.array([[0.0, 0.0, 0.0]]),
                lattice=np.eye(3) * 10.0,
            )],
            atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
        )
        e_lat = _scene_extent(scene_lat, view, 0, atom_scale=0.5)
        assert e_lat > e_no_lat


class TestMakeWedges:
    def test_pure_composition_returns_single_full_circle(self):
        comp = Composition({"Fe": 1.0})
        wedges = _make_wedges(comp, n_segments_total=24, start_angle=math.pi / 2)
        assert len(wedges) == 1
        species, polygon = wedges[0]
        assert species == "Fe"
        assert polygon.shape[1] == 2

    def test_two_species_returns_two_wedges_in_canonical_order(self):
        comp = Composition({"Fe": 0.7, "Mn": 0.3})
        wedges = _make_wedges(comp, n_segments_total=24, start_angle=math.pi / 2)
        species_in_order = [sp for sp, _ in wedges]
        assert species_in_order == ["Fe", "Mn"]

    def test_wedge_angles_proportional_to_occupancy(self):
        comp = Composition({"Fe": 0.75, "Mn": 0.25})
        wedges = _make_wedges(comp, n_segments_total=100, start_angle=0.0)
        fe_polygon = wedges[0][1]
        mn_polygon = wedges[1][1]
        # Each polygon is [centre, arc_v0..arc_vN, centre]; N = n_seg.
        fe_segs = len(fe_polygon) - 3
        mn_segs = len(mn_polygon) - 3
        assert fe_segs + mn_segs <= 100
        assert fe_segs > 2 * mn_segs

    def test_partial_composition_omits_vacancy_wedge(self):
        comp = Composition({"Fe": 0.7})  # 30% vacancy
        wedges = _make_wedges(comp, n_segments_total=24, start_angle=math.pi / 2)
        assert len(wedges) == 1
        assert wedges[0][0] == "Fe"


class TestMakeVacancyWedge:
    def test_full_composition_returns_none(self):
        comp = Composition({"Fe": 1.0})
        result = _make_vacancy_wedge(
            comp, n_segments_total=24, start_angle=math.pi / 2,
        )
        assert result is None

    def test_partial_composition_returns_polygon(self):
        comp = Composition({"Fe": 0.7})
        result = _make_vacancy_wedge(
            comp, n_segments_total=24, start_angle=math.pi / 2,
        )
        assert result is not None
        assert result.shape[1] == 2
        # Centre + arc segments + closing centre.
        assert len(result) >= 3

    def test_mixed_partial_composition_returns_polygon(self):
        comp = Composition({"Fe": 0.5, "Mn": 0.2})  # 30% vacancy
        result = _make_vacancy_wedge(
            comp, n_segments_total=24, start_angle=math.pi / 2,
        )
        assert result is not None
