"""Tests for projection helpers — _project_point and _scene_extent."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
    Frame,
    Orthographic,
    Perspective,
    StructureScene,
    ViewState,
)
from hofmann.model.composition import Composition
from hofmann.rendering.cell_edges import _cell_edges_3d
from hofmann.rendering.projection import (
    _make_vacancy_wedge,
    _make_wedges,
    _project_point,
    _scene_extent,
)
from hofmann.rendering.static import render_mpl


class TestProjectPoint:
    def test_orthographic(self):
        view = ViewState()
        pt = np.array([1.0, 2.0, 3.0])
        xy = _project_point(pt, view)
        np.testing.assert_allclose(xy, [1.0, 2.0])

    def test_perspective(self):
        view = ViewState(projection=Perspective(1.0, 10.0))
        pt = np.array([1.0, 0.0, 0.0])  # depth 0 -> no foreshortening
        xy = _project_point(pt, view)
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
        view_no_persp = ViewState(projection=Orthographic())
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

    def test_orthographic_extent_is_the_bounding_radius(self):
        # Guard: orthographic applies no magnification (max_magnification
        # == 1.0), so the extent is exactly the bounding radius whatever
        # the worst depth.  Only perspective may move.
        scene = StructureScene(
            species=["C"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 5.0]]))],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        view = ViewState(projection=Orthographic())
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        assert extent == 5.0 + 1.0 * 0.5  # centre distance + display radius

    def test_cell_corner_drives_perspective_magnification(self):
        # Atom at the centre (centre distance 0), so the pre-fix bound,
        # magnifying from max(centre distances), gives no allowance; the
        # far cell corner must drive the magnification instead.
        scene = StructureScene(
            species=["C"],
            frames=[Frame(
                coords=np.array([[0.0, 0.0, 0.0]]),
                lattice=np.eye(3) * 10.0,
            )],
            atom_styles={"C": AtomStyle(0.5, (0.5, 0.5, 0.5))},
        )
        view = ViewState(projection=Perspective(0.5, 50.0))
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        corner_radius = math.sqrt(3 * 10.0**2)
        assert extent > corner_radius  # pre-fix: exactly corner_radius

    def test_atom_free_lattice_gets_perspective_allowance(self):
        scene = StructureScene(
            species=[],
            frames=[Frame(
                coords=np.empty((0, 3)),
                lattice=np.eye(3) * 10.0,
            )],
            atom_styles={},
        )
        view = ViewState(projection=Perspective(0.5, 50.0))
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        corner_radius = math.sqrt(3 * 10.0**2)
        assert extent > corner_radius  # pre-fix: no magnification, == corner_radius

    def test_display_radius_drives_perspective_magnification(self):
        # A large atom at the origin: its surface, not its centre
        # (distance 0), is the outermost point and must drive the
        # magnification.
        scene = StructureScene(
            species=["C"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"C": AtomStyle(4.0, (0.5, 0.5, 0.5))},
        )
        view = ViewState(projection=Perspective(0.5, 50.0))
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        surface_radius = 4.0 * 0.5  # centre 0 + radius * atom_scale
        assert extent > surface_radius  # pre-fix: mag from centre 0, == surface_radius

    def test_extent_encloses_the_worst_case_rotation(self):
        # The viewport must ENCLOSE the outermost atom at its worst
        # (nearest-the-eye) rotation, not merely exceed the un-magnified
        # radius: sweep rotations and confirm the rotation-invariant
        # extent covers the largest projected offset the atom reaches.
        coords = np.array([[3.0, 0.0, 4.0]])  # distance 5 from the origin
        radii = np.array([0.01])
        scene = StructureScene(
            species=["C"],
            frames=[Frame(coords=coords)],
            atom_styles={"C": AtomStyle(0.01, (0.5, 0.5, 0.5))},
        )
        persp = Perspective(1.0, 10.0)
        extent = _scene_extent(
            scene, ViewState(projection=persp), 0, atom_scale=1.0
        )
        worst = 0.0
        for angle in np.linspace(0.0, 2 * np.pi, 60):
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            xy, _, srad = ViewState(
                rotation=rot, projection=persp
            ).project(coords, radii)
            worst = max(worst, float(np.hypot(xy[0, 0], xy[0, 1]) + srad[0]))
        assert extent >= worst

    def test_empty_scene_perspective_is_magnified(self):
        # No atoms and no lattice: max_extent falls back to 1.0, and the
        # dropped atom-count gate now applies the perspective allowance.
        scene = StructureScene(
            species=[],
            frames=[Frame(coords=np.empty((0, 3)))],
            atom_styles={},
        )
        view = ViewState(projection=Perspective(0.5, 50.0))
        extent = _scene_extent(scene, view, 0, atom_scale=0.5)
        assert extent > 1.0  # pre-fix: gated out, returns the 1.0 floor

    def test_extent_is_independent_of_zoom(self):
        # The extent is in scene units; the coordinates carry the zoom.
        # A frame-navigation recompute after zooming therefore returns
        # the same viewport, so the interactive zoom is not reset.
        scene = StructureScene(
            species=["C"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 5.0]]))],
            atom_styles={"C": AtomStyle(1.0, (0.5, 0.5, 0.5))},
        )
        e1 = _scene_extent(
            scene, ViewState(zoom=1.0, projection=Orthographic()), 0,
            atom_scale=0.5,
        )
        e2 = _scene_extent(
            scene, ViewState(zoom=2.0, projection=Orthographic()), 0,
            atom_scale=0.5,
        )
        assert e1 == e2


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


def _point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray,
) -> float:
    """Shortest distance from *point* to the segment *start*-*end*."""
    seg = end - start
    length_sq = float(seg @ seg)
    if length_sq == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ seg / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * seg)))


class TestDrawnGeometryMatchesProjection:
    """Every renderer must obtain screen positions from ViewState.

    The perspective scale was once open-coded in the cell-edge loop
    and in _project_point as well as in ViewState.project.  These
    tests pin the agreement, so a copy reintroduced in either place
    fails rather than drifting silently.
    """

    @staticmethod
    def _scene() -> StructureScene:
        """Two tiny atoms in a cell, so edge clipping at spheres is inert."""
        lattice = np.array([
            [4.0, 0.0, 0.0], [0.5, 3.6, 0.0], [0.3, 0.4, 5.2],
        ])
        return StructureScene(
            species=["A", "B"],
            frames=[Frame(
                coords=np.array([[1.0, 1.0, 1.0], [2.4, 1.8, 3.0]]),
                lattice=lattice,
            )],
            atom_styles={
                "A": AtomStyle(0.01, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.01, (0.8, 0.2, 0.2)),
            },
        )

    @staticmethod
    def _drawn_edge_endpoints(fig) -> np.ndarray:
        """Recover segment endpoints from drawn cell-edge rectangles.

        Edges are drawn as [start+offset, end+offset, end-offset,
        start-offset], so the midpoints of the short sides recover the
        endpoints exactly -- the half-width offsets cancel.  With atom
        radii this small, 4-vertex polygons are only ever cell edges.
        """
        endpoints = []
        for collection in fig.axes[0].collections:
            for path in collection.get_paths():
                v = np.asarray(path.vertices)
                if len(v) in (4, 5):  # 5 = closed polygon repeats vertex 0
                    q = v[:4]
                    endpoints.append((q[0] + q[3]) / 2)
                    endpoints.append((q[1] + q[2]) / 2)
        return np.array(endpoints)

    @pytest.mark.parametrize(
        "projection",
        [Orthographic(), Perspective(0.6, 12.0)],
        ids=["orthographic", "perspective"],
    )
    def test_cell_edges_agree_with_view_project(self, projection):
        scene = self._scene()
        scene.view = ViewState(projection=projection)
        scene.view.look_along([1.0, 0.6, 0.4])

        fig = render_mpl(scene, show=False)
        drawn = self._drawn_edge_endpoints(fig)
        plt.close(fig)

        # The renderer's own edge set, so the test cannot drift from
        # the geometry it checks.  Edges are subdivided for depth
        # sorting, so a drawn endpoint is generally interior to an
        # edge -- but a projection maps straight lines to straight
        # lines, so it must still lie on the projected edge.
        starts, ends = _cell_edges_3d(scene.frames[0].lattice)
        projected, _, _ = scene.view.project(np.vstack([starts, ends]))
        edges = list(zip(projected[: len(starts)], projected[len(starts):]))

        assert len(drawn) > 0
        for point in drawn:
            gap = min(
                _point_to_segment_distance(point, start, end)
                for start, end in edges
            )
            assert gap < 1e-9, f"drawn endpoint {point} lies off every edge"

    def test_project_point_agrees_with_project_camera(self):
        """_project_point is a scalar view of the same mapping."""
        view = ViewState(zoom=1.4, projection=Perspective(0.7, 9.0))
        camera = np.array([[1.0, -2.0, 3.0], [0.5, 0.25, -4.0]])
        batch_xy = view.project_camera(camera)
        for i, point in enumerate(camera):
            xy = _project_point(point, view)
            np.testing.assert_array_equal(xy, batch_xy[i])


class TestCellEdgeSubSegmentPairing:
    """Sub-segment screen positions must stay paired with their depths.

    Edges are split at atom depths and each piece is filed in the depth
    slot its own midpoint falls into, so atoms occlude the pieces
    behind them and not the ones in front.  Projecting a whole edge's
    endpoints in one call makes it possible to pair a piece with
    another piece's depth; the point-on-segment check above cannot see
    that, being invariant under permuting the pieces.
    """

    def test_piece_depth_matches_its_own_screen_position(self):
        """Recover each piece's depth from its geometry, independently."""
        from hofmann.model import CellEdgeStyle
        from hofmann.rendering.cell_edges import (
            _cell_edges_3d,
            _collect_cell_edges,
        )

        lattice = np.eye(3) * 12.0
        # Orthographic: screen position maps back to camera x/y exactly,
        # so a piece's depth can be recovered from where it was drawn.
        view = ViewState()
        view.look_along([1.0, 0.6, 0.4])
        g = np.arange(3) * 5.0
        xs, ys, zs = np.meshgrid(g, g, g, indexing="ij")
        coords = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
        depth = coords @ view.rotation[2]

        by_slot = _collect_cell_edges(
            lattice=lattice, view=view, cell_style=CellEdgeStyle(),
            depth=depth, order=np.argsort(depth), pad=30.0, coords=coords,
            radii_3d=np.full(len(coords), 0.4),
        )
        assert by_slot, "expected cell edges to be drawn"

        # Camera-space edges, against which a drawn midpoint is located.
        starts, ends = _cell_edges_3d(lattice)
        cam_s = (starts - view.centre) @ view.rotation.T
        cam_e = (ends - view.centre) @ view.rotation.T

        checked = 0
        for pieces in by_slot.values():
            for polygon, _colour, piece_depth in pieces:
                mid_xy = np.asarray(polygon).mean(axis=0) / view.zoom
                # Find the edge this piece lies on, and how far along.
                best = None
                for c_s, c_e in zip(cam_s, cam_e):
                    seg = (c_e - c_s)[:2]
                    L = float(seg @ seg)
                    if L == 0.0:
                        continue
                    t = float(np.clip((mid_xy - c_s[:2]) @ seg / L, 0.0, 1.0))
                    gap = float(np.linalg.norm(c_s[:2] + t * seg - mid_xy))
                    if best is None or gap < best[0]:
                        best = (gap, c_s, c_e, t)
                gap, c_s, c_e, t = best
                assert gap < 1e-9, "piece does not lie on any cell edge"
                # Depth varies linearly along a straight camera-space
                # edge, so the piece's own position fixes its depth.
                expected = c_s[2] + t * (c_e[2] - c_s[2])
                assert abs(expected - piece_depth) < 1e-9, (
                    f"piece drawn at t={t:.4f} along its edge has depth "
                    f"{expected:.6f}, but was filed under {piece_depth:.6f}"
                )
                checked += 1
        assert checked > 12, f"only {checked} pieces checked"
