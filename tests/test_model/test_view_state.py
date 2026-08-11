"""Tests for ViewState projection, look_along, slab clipping, and validation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hofmann.model.view_state import (
    CABINET,
    CAVALIER,
    Oblique,
    Orthographic,
    Perspective,
    ViewState,
)


class TestViewStateProject:
    def test_identity_rotation(self):
        vs = ViewState()
        coords = np.array([[1.0, 2.0, 3.0]])
        xy, depth, proj_r = vs.project(coords)
        np.testing.assert_allclose(xy, [[1.0, 2.0]])
        np.testing.assert_allclose(depth, [3.0])
        np.testing.assert_allclose(proj_r, [0.0])  # no radii given

    def test_with_centre(self):
        vs = ViewState(centre=np.array([1.0, 1.0, 1.0]))
        coords = np.array([[1.0, 1.0, 1.0]])
        xy, depth, _ = vs.project(coords)
        np.testing.assert_allclose(xy, [[0.0, 0.0]])
        np.testing.assert_allclose(depth, [0.0])

    def test_with_zoom(self):
        vs = ViewState(zoom=2.0)
        coords = np.array([[1.0, 2.0, 3.0]])
        xy, depth, _ = vs.project(coords)
        np.testing.assert_allclose(xy, [[2.0, 4.0]])

    def test_90_degree_z_rotation(self):
        # Rotate 90 degrees around z-axis: x -> y, y -> -x
        angle = np.pi / 2
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        vs = ViewState(rotation=rotation)
        coords = np.array([[1.0, 0.0, 0.0]])
        xy, depth, _ = vs.project(coords)
        np.testing.assert_allclose(xy, [[0.0, 1.0]], atol=1e-10)

    def test_perspective_scaling(self):
        vs = ViewState(projection=Perspective(1.0, 10.0))
        coords = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, -5.0],
        ])
        xy, depth, _ = vs.project(coords)
        # Closer point (depth=0) projected at x=1*10/10=1.0
        # Further point (depth=-5) projected at x=1*10/15=0.667
        np.testing.assert_allclose(xy[0, 0], 1.0)
        np.testing.assert_allclose(xy[1, 0], 10.0 / 15.0, rtol=1e-6)

    def test_multiple_points_shape(self):
        vs = ViewState()
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        xy, depth, proj_r = vs.project(coords)
        assert xy.shape == (3, 2)
        assert depth.shape == (3,)
        assert proj_r.shape == (3,)

    def test_projected_radii_orthographic(self):
        vs = ViewState(zoom=2.0)
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])
        _, _, proj_r = vs.project(coords, radii)
        np.testing.assert_allclose(proj_r, [3.0])  # r * zoom

    def test_projected_radii_perspective(self):
        vs = ViewState(projection=Perspective(1.0, 10.0))
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        _, _, proj_r = vs.project(coords, radii)
        # Silhouette: r * D / sqrt(D^2 - r^2) = 1 * 10 / sqrt(99)
        expected = 10.0 / np.sqrt(99.0)
        np.testing.assert_allclose(proj_r, [expected], rtol=1e-6)

    def test_projected_radii_larger_than_point_scale(self):
        """Silhouette radii should exceed naive r * scale under perspective."""
        vs = ViewState(projection=Perspective(1.0, 10.0))
        coords = np.array([[0.0, 0.0, 2.0]])  # closer to eye
        radii = np.array([1.0])
        _, _, proj_r = vs.project(coords, radii)
        # d = 10 - 2 = 8, naive = r * D/d = 1.25
        naive = 1.0 * 10.0 / 8.0
        assert proj_r[0] > naive  # silhouette > naive point projection


class TestViewStateLookAlong:
    """Tests for ViewState.look_along."""

    def test_default_view_is_z_axis(self):
        """Looking along [0, 0, 1] should give identity rotation."""
        vs = ViewState()
        vs.look_along([0, 0, 1])
        np.testing.assert_allclose(vs.rotation, np.eye(3), atol=1e-14)

    def test_rotation_is_orthogonal(self):
        """The resulting rotation should satisfy R^T R = I."""
        for direction in [[1, 1, 1], [1, 0, 0], [0, 1, 0], [-1, 2, 3]]:
            vs = ViewState()
            vs.look_along(direction)
            np.testing.assert_allclose(
                vs.rotation.T @ vs.rotation, np.eye(3), atol=1e-14,
            )

    def test_direction_maps_to_z(self):
        """The given direction should project to depth only (no xy offset)."""
        vs = ViewState()
        vs.look_along([1, 1, 1])
        # A point along [1,1,1] should project to xy = [0, 0].
        coords = np.array([[3.0, 3.0, 3.0]])
        xy, _, _ = vs.project(coords)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-12)

    def test_x_axis_view(self):
        """Looking along [1, 0, 0] should show the yz plane."""
        vs = ViewState()
        vs.look_along([1, 0, 0])
        # A point at [5, 0, 0] should have zero xy displacement.
        coords = np.array([[5.0, 0.0, 0.0]])
        xy, _, _ = vs.project(coords)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-12)
        # A point at [0, 1, 0] should appear in the screen plane.
        coords = np.array([[0.0, 1.0, 0.0]])
        xy, _, _ = vs.project(coords)
        assert np.linalg.norm(xy[0]) > 0.5

    def test_negative_direction(self):
        """Looking along [0, 0, -1] should flip the view."""
        vs = ViewState()
        vs.look_along([0, 0, -1])
        # A point at [1, 0, 0] should flip its x coordinate.
        coords = np.array([[1.0, 0.0, 0.0]])
        xy, _, _ = vs.project(coords)
        np.testing.assert_allclose(xy[0, 0], -1.0, atol=1e-12)

    def test_custom_up_vector(self):
        """A custom up vector should change the screen-space orientation."""
        vs1 = ViewState()
        vs1.look_along([0, 0, 1], up=[0, 1, 0])
        vs2 = ViewState()
        vs2.look_along([0, 0, 1], up=[1, 0, 0])
        # The two rotations should differ.
        assert not np.allclose(vs1.rotation, vs2.rotation)

    def test_preserves_other_state(self):
        """look_along should only change the rotation."""
        vs = ViewState(zoom=2.5, projection=Perspective(0.8, 15.0))
        vs.look_along([1, 1, 0])
        assert vs.zoom == 2.5
        assert vs.projection == Perspective(0.8, 15.0)

    def test_up_parallel_to_direction_raises(self):
        """An explicit up vector parallel to the view direction should raise."""
        vs = ViewState()
        with pytest.raises(ValueError, match="parallel"):
            vs.look_along([1, 0, 0], up=[1, 0, 0])

    def test_default_up_fallback_for_y_axis(self):
        """Looking along [0,1,0] with default up should not raise."""
        vs = ViewState()
        vs.look_along([0, 1, 0])  # should not raise
        np.testing.assert_allclose(
            vs.rotation.T @ vs.rotation, np.eye(3), atol=1e-14,
        )

    def test_returns_self_for_chaining(self):
        """look_along should return self so callers can chain."""
        vs = ViewState()
        result = vs.look_along([1, 1, 1])
        assert result is vs


class TestViewStateSlab:
    """Tests for depth-slab clipping on ViewState."""

    def test_defaults_are_none(self):
        """Slab fields default to None (no clipping)."""
        vs = ViewState()
        assert vs.slab_origin is None
        assert vs.slab_near is None
        assert vs.slab_far is None

    def test_slab_mask_no_slab(self):
        """Without slab settings, all atoms are visible."""
        vs = ViewState()
        coords = np.array([[0.0, 0.0, z] for z in range(-5, 6)])
        mask = vs.slab_mask(coords)
        assert mask.all()

    def test_slab_mask_filters_depth(self):
        """Only atoms within the slab depth range should be visible."""
        vs = ViewState()
        # Default view: looking along z, centre at origin.
        # Atoms at z = -5, -3, 0, 3, 5.
        coords = np.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0, -3.0],
            [0.0, 0.0,  0.0],
            [0.0, 0.0,  3.0],
            [0.0, 0.0,  5.0],
        ])
        vs.slab_near = -2.0
        vs.slab_far = 2.0
        # slab_origin defaults to centre (origin), so depth range is [-2, 2].
        mask = vs.slab_mask(coords)
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_slab_mask_with_custom_origin(self):
        """Slab origin shifts the depth reference point."""
        vs = ViewState()
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 7.0],
        ])
        vs.slab_origin = np.array([0.0, 0.0, 5.0])
        vs.slab_near = -1.5
        vs.slab_far = 1.5
        # Slab centred at depth of [0,0,5] (which is z=5 in default view),
        # so visible range is depth 3.5 to 6.5.
        mask = vs.slab_mask(coords)
        expected = np.array([False, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_slab_mask_respects_rotation(self):
        """Slab should work in rotated camera space."""
        vs = ViewState()
        vs.look_along([1, 0, 0])  # looking along x
        coords = np.array([
            [-5.0, 0.0, 0.0],
            [ 0.0, 0.0, 0.0],
            [ 5.0, 0.0, 0.0],
        ])
        vs.slab_near = -1.0
        vs.slab_far = 1.0
        mask = vs.slab_mask(coords)
        # Only the atom at x=0 (depth=0 when looking along x) is in range.
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_slab_near_only(self):
        """Setting only slab_near clips from one side."""
        vs = ViewState()
        coords = np.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0,  0.0],
            [0.0, 0.0,  5.0],
        ])
        vs.slab_near = -1.0
        # No far limit — everything from depth -1 onwards is visible.
        mask = vs.slab_mask(coords)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_slab_far_only(self):
        """Setting only slab_far clips from the other side."""
        vs = ViewState()
        coords = np.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0,  0.0],
            [0.0, 0.0,  5.0],
        ])
        vs.slab_far = 1.0
        mask = vs.slab_mask(coords)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_slab_mask_unchanged_by_oblique(self):
        """Slab clipping operates on camera-space depth, which the
        oblique offset must not touch — the invariant's most
        safety-critical clause."""
        rng = np.random.default_rng(7)
        coords = rng.normal(scale=4.0, size=(30, 3))
        vs_plain = ViewState(slab_near=-1.5, slab_far=2.0)
        vs_oblique = ViewState(
            slab_near=-1.5, slab_far=2.0, projection=CAVALIER,
        )
        np.testing.assert_array_equal(
            vs_oblique.slab_mask(coords), vs_plain.slab_mask(coords)
        )


class TestViewStateValidation:
    def test_zero_zoom_raises(self):
        with pytest.raises(ValueError, match="zoom"):
            ViewState(zoom=0.0)

    def test_negative_zoom_raises(self):
        with pytest.raises(ValueError, match="zoom"):
            ViewState(zoom=-1.0)

    def test_valid_view_state_accepted(self):
        vs = ViewState(zoom=2.0, projection=Perspective(view_distance=15.0))
        assert vs.zoom == 2.0

    def test_non_finite_scalars_rejected(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                ViewState(zoom=bad)


class TestViewStateProjection:
    """Tests for the projection field and with_projection."""

    def test_default_is_orthographic(self):
        assert ViewState().projection == Orthographic()

    def test_with_projection_sets_and_returns_self(self):
        vs = ViewState()
        result = vs.with_projection(CAVALIER)
        assert result is vs
        assert vs.projection == CAVALIER

    def test_with_projection_chains_with_look_along(self):
        vs = ViewState().look_along([0, -1, 0]).with_projection(CABINET)
        assert vs.projection == CABINET


class TestViewStateScreenMatrix:
    """Tests for the camera-to-screen linear map."""

    def test_identity_when_oblique_none(self):
        m = ViewState().screen_matrix
        np.testing.assert_array_equal(
            m, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )

    def test_oblique_third_column(self):
        vs = ViewState(projection=Oblique(angle=35.0, foreshortening=0.6))
        th = np.radians(35.0)
        expected = np.array([
            [1.0, 0.0, -0.6 * np.cos(th)],
            [0.0, 1.0, -0.6 * np.sin(th)],
        ])
        np.testing.assert_allclose(vs.screen_matrix, expected)

    def test_scale_bound_is_one_when_oblique_none(self):
        assert ViewState().screen_scale_bound == 1.0

    def test_scale_bound_is_largest_singular_value(self):
        vs = ViewState(projection=Oblique(angle=35.0, foreshortening=0.6))
        singular_values = np.linalg.svd(vs.screen_matrix, compute_uv=False)
        np.testing.assert_allclose(
            vs.screen_scale_bound, singular_values.max()
        )
        np.testing.assert_allclose(
            vs.screen_scale_bound, np.sqrt(1.0 + 0.6**2)
        )


class TestOblique:
    """Tests for the Oblique value type and its preset constants."""

    def test_defaults(self):
        ob = Oblique()
        assert ob.angle == 45.0
        assert ob.foreshortening == 0.5

    def test_frozen(self):
        ob = Oblique()
        with pytest.raises(FrozenInstanceError):
            ob.angle = 30.0

    def test_presets(self):
        assert CAVALIER == Oblique(45.0, 1.0)
        assert CABINET == Oblique(45.0, 0.5)

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                Oblique(angle=bad)
            with pytest.raises(ValueError, match="finite"):
                Oblique(foreshortening=bad)

    def test_negative_foreshortening_allowed(self):
        """Negative f is meaningful: equivalent to angle + 180 degrees."""
        assert Oblique(foreshortening=-0.5).foreshortening == -0.5


class TestViewStateProjectCamera:
    """Tests for the consolidated camera-to-screen mapping."""

    def test_orthographic_matches_manual(self):
        vs = ViewState(zoom=2.0)
        camera = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -2.0]])
        xy, scale = vs.project_camera(camera)
        np.testing.assert_array_equal(xy, camera[:, :2] * 2.0)
        np.testing.assert_array_equal(scale, [1.0, 1.0])

    def test_perspective_matches_manual(self):
        vs = ViewState(projection=Perspective(0.5, 10.0), zoom=1.5)
        camera = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, -2.0]])
        xy, scale = vs.project_camera(camera)
        expected_scale = 10.0 / (10.0 - camera[:, 2] * 0.5)
        np.testing.assert_allclose(scale, expected_scale)
        np.testing.assert_allclose(
            xy, camera[:, :2] * expected_scale[:, np.newaxis] * 1.5
        )

    def test_perspective_path_is_bit_exact(self):
        """The project comment promises byte-identical output to the
        pre-oblique implementation; pin the perspective path against
        the formula written with the same operations in the same
        order."""
        p = Perspective(0.7, 12.0)
        vs = ViewState(projection=p, zoom=1.3)
        rng = np.random.default_rng(11)
        pts = rng.normal(scale=2.0, size=(15, 3))
        radii = np.linspace(0.1, 0.8, 15)

        rotated = (pts - vs.centre) @ vs.rotation.T
        depth = rotated[:, 2]
        d = p.view_distance - depth * p.strength
        scale = p.view_distance / d
        expected_xy = rotated[:, :2] * scale[:, np.newaxis] * vs.zoom
        denom = np.sqrt(np.maximum(d**2 - radii**2, 1e-12))
        expected_radii = radii * p.view_distance / denom * vs.zoom

        xy, depth_out, radii_out = vs.project(pts, radii)
        np.testing.assert_array_equal(xy, expected_xy)
        np.testing.assert_array_equal(depth_out, depth)
        np.testing.assert_array_equal(radii_out, expected_radii)


class TestViewStateProjectOblique:
    """Tests for oblique projection through ViewState.project."""

    def _points(self):
        rng = np.random.default_rng(42)
        return rng.normal(scale=3.0, size=(20, 3))

    def test_oblique_none_unchanged(self):
        """Regression guard: the refactored path must be exactly the
        old orthographic computation."""
        vs = ViewState(zoom=1.7, centre=np.array([0.5, -0.2, 1.0]))
        pts = self._points()
        rotated = (pts - vs.centre) @ vs.rotation.T
        xy, depth, radii = vs.project(pts, np.full(len(pts), 0.4))
        np.testing.assert_array_equal(xy, rotated[:, :2] * 1.7)
        np.testing.assert_array_equal(depth, rotated[:, 2])
        np.testing.assert_array_equal(radii, np.full(len(pts), 0.4 * 1.7))

    def test_zero_foreshortening_is_orthographic(self):
        vs_ortho = ViewState()
        vs_oblique = ViewState(projection=Oblique(35.0, 0.0))
        pts = self._points()
        xy_o, d_o, _ = vs_ortho.project(pts)
        xy_q, d_q, _ = vs_oblique.project(pts)
        np.testing.assert_allclose(xy_q, xy_o, atol=1e-15)
        np.testing.assert_array_equal(d_q, d_o)

    def test_offset_is_minus_f_d_cos_sin(self):
        """A point at depth d is displaced by exactly -f*d*(cos, sin)
        from its orthographic position."""
        angle, f = 35.0, 0.6
        vs = ViewState(projection=Oblique(angle, f))
        pts = self._points()
        xy_ortho, depth, _ = ViewState().project(pts)
        xy_obl, depth_obl, _ = vs.project(pts)
        th = np.radians(angle)
        expected = xy_ortho - depth[:, np.newaxis] * f * np.array(
            [np.cos(th), np.sin(th)]
        )
        np.testing.assert_allclose(xy_obl, expected, atol=1e-14)

    def test_depth_unchanged_by_oblique(self):
        """Painter ordering must be identical to the orthographic case."""
        pts = self._points()
        _, depth_ortho, _ = ViewState().project(pts)
        _, depth_obl, _ = ViewState(projection=CAVALIER).project(pts)
        np.testing.assert_array_equal(depth_obl, depth_ortho)

    def test_radii_unchanged_by_oblique(self):
        """Spheres keep circular outlines by convention."""
        pts = self._points()
        radii = np.linspace(0.2, 1.0, len(pts))
        _, _, r_ortho = ViewState().project(pts, radii)
        _, _, r_obl = ViewState(projection=CAVALIER).project(pts, radii)
        np.testing.assert_array_equal(r_obl, r_ortho)

    def test_cavalier_unit_step_cabinet_half_step(self):
        """A unit step along the receding axis (depth -1, i.e. away
        from the viewer) draws as a unit step on screen under cavalier
        and a half step under cabinet, in the direction of *angle*."""
        step = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        for oblique, length in ((CAVALIER, 1.0), (CABINET, 0.5)):
            vs = ViewState(rotation=np.eye(3), projection=oblique)
            xy, _, _ = vs.project(step)
            drawn = xy[1] - xy[0]
            th = np.radians(oblique.angle)
            np.testing.assert_allclose(
                drawn, length * np.array([np.cos(th), np.sin(th)]),
                atol=1e-15,
            )

    def test_reproduces_manual_shear_figure(self):
        """The new API must reproduce the published figure's shear
        matrix (fig_p3121_legend.py in data_nbo2f_chirality): camera
        along [0, -1, 0], receding axis at 35 degrees, f = 0.6.

        The shear's depth row is [0, -1, 0], so the camera sits on
        the -y side: look_along([0, 1, 0]) does NOT match (it mirrors
        x and flips the depth sign).
        """
        th = np.radians(35.0)
        shear = np.array([
            [1.0, 0.6 * np.cos(th), 0.0],
            [0.0, 0.6 * np.sin(th), 1.0],
            [0.0, -1.0,             0.0],
        ])
        pts = self._points()
        sheared = pts @ shear.T
        expected_xy = sheared[:, :2]
        expected_depth = sheared[:, 2]

        vs = ViewState().look_along([0, -1, 0]).with_projection(
            Oblique(35.0, 0.6)
        )
        xy, depth, _ = vs.project(pts)
        np.testing.assert_allclose(xy, expected_xy, atol=1e-14)
        np.testing.assert_array_equal(depth, expected_depth)

        # Guard: the wrong camera (the one the original working note
        # proposed) must not match.
        vs_wrong = ViewState().look_along([0, 1, 0]).with_projection(
            Oblique(35.0, 0.6)
        )
        xy_wrong, depth_wrong, _ = vs_wrong.project(pts)
        assert not np.allclose(xy_wrong, expected_xy)
        assert not np.allclose(depth_wrong, expected_depth)


class TestProjectionTypes:
    """Tests for the projection mode value types."""

    def test_orthographic_instances_equal(self):
        assert Orthographic() == Orthographic()

    def test_orthographic_not_equal_to_other_modes(self):
        assert Orthographic() != Perspective()
        assert Orthographic() != Oblique()

    def test_perspective_defaults(self):
        p = Perspective()
        assert p.strength == 0.5
        assert p.view_distance == 10.0

    def test_perspective_rejects_non_positive_strength(self):
        for bad in (0.0, -0.5):
            with pytest.raises(ValueError, match="strength"):
                Perspective(strength=bad)

    def test_perspective_rejects_non_finite(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                Perspective(strength=bad)
            with pytest.raises(ValueError, match="finite"):
                Perspective(view_distance=bad)

    def test_perspective_rejects_non_positive_view_distance(self):
        with pytest.raises(ValueError, match="view_distance"):
            Perspective(view_distance=0.0)

    def test_modes_are_frozen_with_slots(self):
        # A new attribute name raises TypeError (slots, no __dict__); an
        # existing field name raises FrozenInstanceError (frozen).
        # Orthographic has no fields, so only the former applies to it.
        o = Orthographic()
        with pytest.raises(TypeError):  # slots prevents new attributes
            o.anything = 1
        assert not hasattr(o, "__dict__")

        p = Perspective()
        with pytest.raises(FrozenInstanceError):  # frozen prevents field changes
            p.strength = 0.3
        assert not hasattr(p, "__dict__")

        ob = Oblique()
        with pytest.raises(FrozenInstanceError):  # frozen prevents field changes
            ob.angle = 30.0
        assert not hasattr(ob, "__dict__")
