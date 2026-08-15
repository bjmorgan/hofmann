"""Tests for projection helpers — _project_point and _scene_extent."""

import math
import warnings

import numpy as np
import pytest

from hofmann.model import (
    AtomStyle, Frame, Oblique, Perspective, StructureScene, ViewState,
)
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
        view = ViewState(projection=Perspective(1.0, 10.0))
        pt = np.array([1.0, 0.0, 0.0])  # depth 0 -> scale = 1
        xy, s = _project_point(pt, view)
        np.testing.assert_allclose(s, 1.0)
        np.testing.assert_allclose(xy, [1.0, 0.0])

    def test_equivalent_to_project_camera(self):
        """_project_point is the single-point view of project_camera —
        this is what pins bond rendering to the shared mapping."""
        views = [
            ViewState(zoom=1.5),
            ViewState(projection=Perspective(0.5, 8.0)),
            ViewState(projection=Oblique(35.0, 0.6), zoom=2.0),
        ]
        pts = np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0], [-4.0, 1.0, -1.5]])
        for view in views:
            batch_xy, batch_scale = view.project_camera(pts)
            for i, pt in enumerate(pts):
                xy, s = _project_point(pt, view)
                np.testing.assert_array_equal(xy, batch_xy[i])
                assert s == batch_scale[i]


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
        view_no_persp = ViewState()
        view_persp = ViewState(projection=Perspective(0.5))
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

    def test_extent_is_independent_of_zoom(self):
        """The fixed interactive viewport bound must not scale with
        zoom — zoom acts through the projected coordinates, so
        recomputing this bound with the live zoom would cancel the
        user's zoom."""
        scene = StructureScene(
            species=["C"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        view_zoom_1 = ViewState(zoom=1.0)
        view_zoom_4 = ViewState(zoom=4.0)
        e_zoom_1 = _scene_extent(scene, view_zoom_1, 0, atom_scale=0.5)
        e_zoom_4 = _scene_extent(scene, view_zoom_4, 0, atom_scale=0.5)
        assert e_zoom_1 == e_zoom_4

    def _two_atom_scene(self):
        return StructureScene(
            species=["C", "C"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 5.0],
            ]))],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )

    def test_oblique_scales_extent_by_scale_bound(self):
        """A full-length receding axis would clip at the viewport edge
        without the sqrt(1 + f^2) allowance."""
        scene = self._two_atom_scene()
        f = 0.6
        e_none = _scene_extent(scene, ViewState(), 0, atom_scale=0.5)
        e_obl = _scene_extent(
            scene, ViewState(projection=Oblique(35.0, f)), 0, atom_scale=0.5,
        )
        np.testing.assert_allclose(e_obl, e_none * math.sqrt(1.0 + f**2))

    def test_extent_unchanged_for_orthographic_projection(self):
        """Orthographic adds no allowance at all: the extent pins to
        its hand-computed base value (max atom distance 5.0 plus
        scaled radius 0.5)."""
        scene = self._two_atom_scene()
        extent = _scene_extent(scene, ViewState(), 0, atom_scale=0.5)
        assert extent == 5.5

    def test_bogus_projection_rejected(self):
        """Only the three modes carry an allowance, so an unrecognised
        projection must raise rather than quietly receive none."""
        scene = self._two_atom_scene()
        view = ViewState()
        view.projection = "bogus"
        with pytest.raises(TypeError, match="projection"):
            _scene_extent(scene, view, 0, atom_scale=0.5)

    def test_eye_inside_scene_bounding_sphere_does_not_explode_extent(self):
        """When the effective eye lies inside the scene's bounding
        sphere, denom = view_distance - bounding_radius * strength
        goes non-positive and no bounded rotation-invariant
        magnification exists.  The viewport must fall back to the
        unmagnified scene bound (content is clipped, which is honest
        and navigable)."""
        n = 36
        xs, ys = np.meshgrid(
            np.linspace(-n / 2, n / 2, n), np.linspace(-n / 2, n / 2, n),
        )
        coords = np.column_stack(
            [xs.ravel(), ys.ravel(), np.zeros(xs.size)]
        )
        scene = StructureScene(
            species=["C"] * len(coords),
            frames=[Frame(coords=coords)],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        # Effective eye plane at view_distance / strength = 25.0, well
        # inside the sheet's ~25.9 bounding radius (face-on view).
        view = ViewState(projection=Perspective(0.4, 10.0))
        bounding_radius = float(
            np.max(np.linalg.norm(coords, axis=1)) + 0.5
        )
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        # The fallback applies no magnification at all (persp_scale =
        # 1.0 exactly), so extent must equal the unmagnified bound
        # bit-for-bit — not merely be smaller than some multiple of
        # it, which a lingering partial allowance would also satisfy.
        assert extent == bounding_radius

    def test_eye_inside_scene_bounding_sphere_does_not_warn(self):
        """The fallback is silent: the camera position is legitimate,
        and the genuinely false output (atoms drawn mirrored) is
        reported separately by ViewState.project_camera when it
        actually occurs."""
        n = 36
        xs, ys = np.meshgrid(
            np.linspace(-n / 2, n / 2, n), np.linspace(-n / 2, n / 2, n),
        )
        coords = np.column_stack(
            [xs.ravel(), ys.ravel(), np.zeros(xs.size)]
        )
        scene = StructureScene(
            species=["C"] * len(coords),
            frames=[Frame(coords=coords)],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        view = ViewState(projection=Perspective(0.4, 10.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _scene_extent(scene, view, 0, atom_scale=0.5)
        assert not caught

    def test_perspective_bound_includes_cell_corners(self):
        """A large cell with atoms near the centre must bound the
        perspective magnification by the corner distance, not the atom
        distance — otherwise the cell clips when rotated eye-ward."""
        a = 20.0
        scene = StructureScene(
            species=["C"],
            frames=[Frame(
                coords=np.array([[a / 2, a / 2, a / 2]]),
                lattice=np.eye(3) * a,
            )],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        corner_dist = np.linalg.norm(np.full(3, a / 2))
        view = ViewState(
            centre=np.full(3, a / 2),
            projection=Perspective(strength=0.5, view_distance=40.0),
        )
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        # Worst case: the corner rotated to depth +corner_dist.
        expected_scale = 40.0 / (40.0 - corner_dist * 0.5)
        assert extent >= corner_dist * expected_scale


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

    def test_segment_count_respects_total_with_vacancy(self):
        """Combined wedge + vacancy segments must not exceed budget."""
        comp = Composition({"Fe": 0.7})  # 70% Fe + 30% vacancy
        n = 100
        wedges = _make_wedges(comp, n_segments_total=n, start_angle=0.0)
        vac = _make_vacancy_wedge(comp, n_segments_total=n, start_angle=0.0)
        assert vac is not None
        wedge_segs = sum(len(p) - 3 for _, p in wedges)
        vac_segs = len(vac) - 3
        # Allow rounding slack of one segment per wedge (here 2 wedges).
        assert wedge_segs + vac_segs <= n + 2

    def test_species_segments_proportional_to_absolute_occupancy(self):
        """A 30% Fe wedge gets ~30% of the budget, not 100%."""
        comp = Composition({"Fe": 0.3})  # 70% vacancy
        n = 100
        wedges = _make_wedges(comp, n_segments_total=n, start_angle=0.0)
        fe_segs = len(wedges[0][1]) - 3
        # Roughly 30, allow ±2 for rounding.
        assert 28 <= fe_segs <= 32


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
