"""Tests for ViewState projection, look_along, slab clipping, and validation."""

import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hofmann.model.view_state import (
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

    def test_projected_radii_converge_to_orthographic_at_low_strength(self):
        """The pinhole sits at view_distance / strength; as strength
        shrinks the effective eye recedes to infinity and silhouette
        radii must converge to the orthographic value (r * zoom)."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        vs_persp = ViewState(projection=Perspective(1e-3, 10.0))
        _, _, proj_r_persp = vs_persp.project(coords, radii)
        vs_ortho = ViewState()
        _, _, proj_r_ortho = vs_ortho.project(coords, radii)
        np.testing.assert_allclose(proj_r_persp, proj_r_ortho, rtol=1e-4)

    def test_projected_radii_no_overflow_at_huge_view_distance(self):
        """At view_distance ~1e160 the silhouette denominator must stay
        finite: d**2 exceeds the float64 maximum around 1.3e154, and
        so does the product (d - r*s) * (d + r*s), so each factor is
        square-rooted before multiplying.  The eye is effectively at
        infinity, so the radius converges to the orthographic
        value."""
        vs = ViewState(projection=Perspective(1.0, 1e160))
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        _, _, proj_r = vs.project(coords, radii)
        assert np.isfinite(proj_r).all()
        np.testing.assert_allclose(proj_r, [1.0], rtol=1e-6)

    def test_projected_radii_depend_only_on_effective_eye(self):
        """Perspective(1.0, 10.0) and Perspective(0.5, 5.0) share the
        same effective eye (view_distance / strength = 10) and must
        produce identical silhouette radii."""
        coords = np.array([[0.0, 0.0, 1.0]])
        radii = np.array([1.0])
        vs_a = ViewState(projection=Perspective(1.0, 10.0))
        vs_b = ViewState(projection=Perspective(0.5, 5.0))
        _, _, proj_r_a = vs_a.project(coords, radii)
        _, _, proj_r_b = vs_b.project(coords, radii)
        np.testing.assert_allclose(proj_r_a, proj_r_b, atol=1e-12)

    def test_no_sphere_warning_for_atoms_behind_the_eye(self):
        """An atom behind the eye plane has a large |d|, so the
        silhouette denominator stays safe and no sphere contains the
        eye.  That case is reported accurately by the eye-plane
        warning; the sphere warning must not also fire and claim
        something untrue."""
        vs = ViewState(projection=Perspective(1.0, 5.0))
        coords = np.array([[0.0, 0.0, 10.0]])  # d = 5 - 10 = -5
        radii = np.array([0.5])  # denominator: 25 - 0.25, safely positive
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, proj_r = vs.project(coords, radii)
        messages = [str(w.message) for w in caught]
        assert any("eye plane" in m for m in messages)
        assert not any("contain the effective" in m for m in messages)
        # d must be used as |d| throughout: using the signed d would
        # take sqrt of a negative number here and silently yield NaN.
        assert np.isfinite(proj_r).all()
        np.testing.assert_allclose(proj_r, [0.502519], rtol=1e-6)

    def test_sphere_containing_effective_eye_warns(self):
        """When a sphere's radius (scaled by strength) reaches the
        eye-to-atom distance, the effective eye lies inside the
        sphere and the 1e-12 clamp yields an absurd radius; that must
        not happen silently."""
        vs = ViewState(projection=Perspective(1.0, 5.0))
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([5.0])  # d = 5, radii * strength = 5 -> d <= radii*strength
        with pytest.warns(UserWarning, match="view_distance"):
            _, _, proj_r = vs.project(coords, radii)
        # abs_d - rs = 0 here, so sqrt_lo * sqrt_hi = 0 exactly and the
        # denominator is clamped to the 1e-6 floor: pin the radius
        # that floor produces (r * D / 1e-6 * zoom = 5 * 5 / 1e-6),
        # so a change to the clamp's magnitude is caught rather than
        # merely tolerated by every test in the suite.
        np.testing.assert_allclose(proj_r, [25_000_000.0])

    def test_no_sphere_warning_for_normal_scene(self):
        """A normal perspective scene — atoms well in front of the
        eye, small radii relative to their eye distance — must not
        emit the sphere-contains-eye warning.  A mutated condition
        that fires on every forward-facing atom would otherwise
        survive the suite silently."""
        vs = ViewState(projection=Perspective(0.5, 10.0))
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
        radii = np.array([0.3, 0.4])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vs.project(coords, radii)
        messages = [str(w.message) for w in caught]
        assert not any("contain the effective" in m for m in messages)


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

    def test_nan_direction_raises(self):
        """A NaN direction has zero norm, so the length guard (a `<`
        comparison) silently fails to trip on NaN and an all-NaN
        rotation would otherwise install without raising."""
        vs = ViewState()
        with pytest.raises(ValueError, match="finite"):
            vs.look_along([float("nan")] * 3)

    def test_rejected_look_along_restores_previous_rotation(self):
        """look_along assigns self.rotation before validating; if
        validation then raises, the object must not be left holding
        the invalid rotation permanently — a caller who catches the
        error needs a usable view to fall back to."""
        vs = ViewState()
        vs.look_along([1, 0, 0])
        previous = vs.rotation.copy()
        with pytest.raises(ValueError, match="finite"):
            vs.look_along([float("nan")] * 3)
        np.testing.assert_array_equal(vs.rotation, previous)
        # The view must still project normally, not merely hold an
        # array that happens to look right.
        coords = np.array([[5.0, 0.0, 0.0]])
        xy, _, _ = vs.project(coords)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-12)

    def test_nan_up_raises(self):
        """A NaN up vector propagates NaN into the cross products and
        would otherwise install an all-NaN rotation without raising."""
        vs = ViewState()
        with pytest.raises(ValueError, match="finite"):
            vs.look_along([1, 0, 0], up=[float("nan")] * 3)


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
            slab_near=-1.5, slab_far=2.0, projection=Oblique(45.0, 1.0),
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

    def test_list_centre_accepted_and_coerced_to_array(self):
        """Plain list input must not raise AttributeError from calling
        .shape on a list — it must be coerced to an ndarray, per the
        ValueError the docstring promises for bad input."""
        vs = ViewState(centre=[0.0, 1.0, 2.0])
        assert isinstance(vs.centre, np.ndarray)
        np.testing.assert_array_equal(vs.centre, [0.0, 1.0, 2.0])

    def test_nested_list_rotation_accepted_and_coerced_to_array(self):
        vs = ViewState(rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert isinstance(vs.rotation, np.ndarray)
        np.testing.assert_array_equal(vs.rotation, np.eye(3))

    def test_list_slab_origin_accepted_and_coerced_to_array(self):
        vs = ViewState(slab_origin=[0.0, 1.0, 2.0])
        assert isinstance(vs.slab_origin, np.ndarray)
        np.testing.assert_array_equal(vs.slab_origin, [0.0, 1.0, 2.0])

    def test_wrong_shape_list_centre_raises_value_error(self):
        """A wrong-shape list must raise ValueError, not AttributeError
        from calling .shape on the raw list."""
        with pytest.raises(ValueError, match="centre.*shape"):
            ViewState(centre=[0.0, 1.0])

    def test_non_finite_list_centre_raises_value_error(self):
        with pytest.raises(ValueError, match="centre.*finite"):
            ViewState(centre=[0.0, float("nan"), 0.0])

    def test_wrong_shape_list_rotation_raises_value_error(self):
        with pytest.raises(ValueError, match="rotation.*shape"):
            ViewState(rotation=[[1.0, 0.0], [0.0, 1.0]])

    def test_wrong_shape_list_slab_origin_raises_value_error(self):
        with pytest.raises(ValueError, match="slab_origin.*shape"):
            ViewState(slab_origin=[0.0, 1.0])

    def test_slab_origin_shape_rejected(self):
        """A wrong-shape slab origin would otherwise broadcast against
        *centre* and silently produce a reference depth per component
        instead of one for the scene."""
        with pytest.raises(ValueError, match="slab_origin.*shape"):
            ViewState(slab_origin=np.array([[0.0], [0.0], [1.0]]))
        with pytest.raises(ValueError, match="slab_origin.*shape"):
            ViewState(slab_origin=np.array([0.0, 1.0]))

    def test_non_finite_scalars_rejected(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="zoom.*finite"):
                ViewState(zoom=bad)
            with pytest.raises(ValueError, match="slab_near.*finite"):
                ViewState(slab_near=bad)
            with pytest.raises(ValueError, match="slab_far.*finite"):
                ViewState(slab_far=bad)
            with pytest.raises(ValueError, match="slab_origin.*finite"):
                ViewState(slab_origin=np.array([0.0, bad, 0.0]))

    def test_non_finite_centre_rejected(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="centre.*finite"):
                ViewState(centre=np.array([0.0, bad, 0.0]))

    def test_wrong_shape_centre_rejected(self):
        with pytest.raises(ValueError, match="centre.*shape"):
            ViewState(centre=np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="centre.*shape"):
            ViewState(centre=np.zeros((3, 1)))

    def test_wrong_shape_rotation_rejected(self):
        with pytest.raises(ValueError, match="rotation.*shape"):
            ViewState(rotation=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="rotation.*shape"):
            ViewState(rotation=np.zeros((4, 3)))

    def test_non_finite_rotation_rejected(self):
        for bad in (float("nan"), float("inf")):
            rotation = np.eye(3)
            rotation[1, 2] = bad
            with pytest.raises(ValueError, match="rotation.*finite"):
                ViewState(rotation=rotation)

    def test_zero_rotation_rejected(self):
        """The all-zero matrix is singular (determinant 0): it
        collapses every atom to the origin, but passes the shape and
        finiteness checks."""
        with pytest.raises(ValueError, match="singular"):
            ViewState(rotation=np.zeros((3, 3)))

    def test_rank_deficient_rotation_rejected(self):
        """A rank-2 matrix is singular (determinant 0): it flattens
        all depth to zero, destroying painter ordering, but still has
        the right shape and is finite."""
        with pytest.raises(ValueError, match="singular"):
            ViewState(rotation=np.diag([1.0, 1.0, 0.0]))

    def test_non_orthonormal_rotation_accepted(self):
        """Only singular matrices are rejected.  A finite matrix that
        scales rather than rotates is applied as given."""
        vs = ViewState(rotation=np.diag([2.0, 1.0, 1.0]))
        np.testing.assert_array_equal(vs.rotation, np.diag([2.0, 1.0, 1.0]))

    def test_reflection_accepted(self):
        """A negative determinant is applied as given, rendering the
        mirror image: screen positions are unchanged and depth is
        negated."""
        vs = ViewState(rotation=np.diag([1.0, 1.0, -1.0]))
        coords = np.array([[1.0, 2.0, 3.0]])
        xy, depth, _ = vs.project(coords)
        np.testing.assert_array_equal(xy, [[1.0, 2.0]])
        np.testing.assert_array_equal(depth, [-3.0])

    def test_post_construction_nan_rotation_raises_on_use(self):
        """Assigning rotation after construction bypasses
        __post_init__; rotation is reassigned on every drag frame, so
        the consumer must validate what it uses rather than silently
        producing all-NaN coordinates."""
        vs = ViewState()
        vs.rotation = np.full((3, 3), np.nan)
        with pytest.raises(ValueError, match="rotation.*finite"):
            vs.project(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="rotation.*finite"):
            vs.slab_mask(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_wrong_shape_rotation_raises_on_use(self):
        vs = ViewState()
        vs.rotation = np.zeros((3, 2))
        with pytest.raises(ValueError, match="rotation.*shape"):
            vs.project(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="rotation.*shape"):
            vs.slab_mask(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_nan_centre_raises_on_use(self):
        """centre is reassigned on every pan keypress; validated at
        construction but not at point of use."""
        vs = ViewState()
        vs.centre = np.array([0.0, float("nan"), 0.0])
        with pytest.raises(ValueError, match="centre.*finite"):
            vs.project(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="centre.*finite"):
            vs.slab_mask(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_wrong_shape_centre_raises_on_use(self):
        """centre is reassigned on every pan keypress; validated at
        construction but not at point of use.  A (2,) centre would
        otherwise reach ``coords - self.centre`` and fail via numpy's
        own broadcast ValueError instead of our check — the match
        pattern below requires the field name, which numpy's generic
        "operands could not be broadcast together" message lacks, so
        it distinguishes our check from that accidental pass."""
        vs = ViewState()
        vs.centre = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="centre.*shape"):
            vs.project(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="centre.*shape"):
            vs.slab_mask(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_nan_zoom_raises_on_use(self):
        """Assigning zoom after construction bypasses __post_init__;
        the consumer must validate what it uses rather than silently
        blanking the figure (NaN comparisons are False)."""
        vs = ViewState()
        vs.zoom = float("nan")
        with pytest.raises(ValueError, match="zoom"):
            vs.project_camera(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_non_positive_zoom_raises_on_use(self):
        vs = ViewState()
        vs.zoom = 0.0
        with pytest.raises(ValueError, match="zoom"):
            vs.project_camera(np.array([[1.0, 2.0, 3.0]]))

    def test_post_construction_nan_slab_near_raises_on_use(self):
        vs = ViewState()
        vs.slab_near = float("nan")
        with pytest.raises(ValueError, match="slab_near.*finite"):
            vs.slab_mask(np.array([[0.0, 0.0, 0.0]]))

    def test_post_construction_nan_slab_far_raises_on_use(self):
        vs = ViewState()
        vs.slab_far = float("nan")
        with pytest.raises(ValueError, match="slab_far.*finite"):
            vs.slab_mask(np.array([[0.0, 0.0, 0.0]]))

    def test_post_construction_nan_slab_origin_raises_on_use(self):
        vs = ViewState()
        vs.slab_origin = np.array([0.0, float("nan"), 0.0])
        with pytest.raises(ValueError, match="slab_origin.*finite"):
            vs.slab_mask(np.array([[0.0, 0.0, 0.0]]))


class TestViewStateProjection:
    """Tests for the projection field."""

    def test_set_orthographic(self):
        vs = ViewState(projection=Perspective(0.5, 10.0))
        result = vs.set_orthographic()
        assert result is vs
        assert vs.projection == Orthographic()

    def test_set_perspective(self):
        vs = ViewState()
        result = vs.set_perspective(strength=0.3, view_distance=12.0)
        assert result is vs
        assert vs.projection == Perspective(0.3, 12.0)

    def test_set_perspective_defaults_match_class(self):
        """The setter's defaults must mirror Perspective's, so the two
        spellings cannot drift apart."""
        import inspect
        from dataclasses import fields as dc_fields

        sig = inspect.signature(ViewState.set_perspective)
        class_defaults = {f.name: f.default for f in dc_fields(Perspective)}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            assert param.default == class_defaults[name]

    def test_set_oblique(self):
        vs = ViewState()
        result = vs.set_oblique(angle=35.0, foreshortening=0.6)
        assert result is vs
        assert vs.projection == Oblique(35.0, 0.6)

    def test_set_oblique_parameters_required(self):
        """Mirrors Oblique itself: no default angle or foreshortening."""
        with pytest.raises(TypeError):
            ViewState().set_oblique()

    def test_setters_chain_with_look_along(self):
        vs = ViewState().look_along([0, -1, 0]).set_oblique(35.0, 0.6)
        assert vs.projection == Oblique(35.0, 0.6)

    def test_setters_validate_through_the_classes(self):
        with pytest.raises(ValueError, match="strength"):
            ViewState().set_perspective(strength=0.0)
        with pytest.raises(ValueError, match="finite"):
            ViewState().set_oblique(angle=float("nan"), foreshortening=0.5)

    def test_default_is_orthographic(self):
        assert ViewState().projection == Orthographic()

    def test_removed_field_assignment_raises(self):
        """slots keeps the break loud: no silent inert attribute."""
        vs = ViewState()
        with pytest.raises(AttributeError):
            vs.perspective = 0.3

    def test_removed_field_kwarg_raises(self):
        with pytest.raises(TypeError):
            ViewState(perspective=0.5)

    def test_constructor_rejects_bogus_projection(self):
        with pytest.raises(TypeError, match="projection"):
            ViewState(projection="bogus")

    def test_dispatch_rejects_bogus_projection_on_assignment(self):
        """Direct assignment bypasses __post_init__; every surface that
        reads projection must raise rather than silently falling back
        to orthographic behaviour."""
        vs = ViewState()
        vs.projection = "bogus"
        with pytest.raises(TypeError, match="projection"):
            vs.project_camera(np.zeros((1, 3)))
        with pytest.raises(TypeError, match="projection"):
            _ = vs.screen_matrix
        with pytest.raises(TypeError, match="projection"):
            _ = vs.screen_scale_bound
        with pytest.raises(TypeError, match="projection"):
            vs.screen_frame(np.zeros((1, 3)))

    def test_project_camera_rejects_wrong_shape(self):
        """A single (3,) point would otherwise broadcast into silently
        duplicated nonsense rows; the shape contract fails loudly."""
        vs = ViewState()
        with pytest.raises(ValueError, match="shape"):
            vs.project_camera(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="shape"):
            vs.project_camera(np.zeros((2, 4)))


class TestViewStateScreenMatrix:
    """Tests for the camera-to-screen linear map."""

    def test_identity_when_orthographic(self):
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

    def test_scale_bound_is_one_when_orthographic(self):
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
    """Tests for the Oblique value type."""

    def test_parameters_are_required(self):
        with pytest.raises(TypeError):
            Oblique()

    def test_frozen(self):
        ob = Oblique(45.0, 0.5)
        with pytest.raises(FrozenInstanceError):
            ob.angle = 30.0

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                Oblique(angle=bad, foreshortening=0.5)
            with pytest.raises(ValueError, match="finite"):
                Oblique(angle=45.0, foreshortening=bad)

    def test_negative_foreshortening_allowed(self):
        """Negative f is meaningful: equivalent to angle + 180 degrees."""
        assert Oblique(angle=45.0, foreshortening=-0.5).foreshortening == -0.5


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

    def test_warns_when_atoms_reach_eye_plane(self):
        """Static rendering must not silently mirror or blow up atoms
        at or behind the perspective eye plane."""
        vs = ViewState(projection=Perspective(1.0, 5.0))
        camera = np.array([[1.0, 1.0, 5.0], [1.0, 1.0, 8.0]])
        with pytest.warns(UserWarning, match="eye plane"):
            vs.project_camera(camera)

    def test_scale_not_clamped_at_or_behind_eye_plane(self):
        """The docstring promises inf or negative scales rather than a
        clamp when a point lies at or behind the eye plane; a clamp
        (e.g. to zero, or to the largest finite scale) would survive
        unless pinned here."""
        vs = ViewState(projection=Perspective(1.0, 5.0))
        # depth=5 -> denom=0 -> scale=inf; depth=8 -> denom=-3 -> scale<0.
        camera = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 8.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, scale = vs.project_camera(camera)
        assert scale[0] == np.inf
        assert scale[1] < 0

    def test_no_warning_for_safe_perspective(self):
        vs = ViewState(projection=Perspective(0.5, 10.0))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            vs.project_camera(np.array([[1.0, 1.0, 2.0]]))


class TestViewStateProjectOblique:
    """Tests for oblique projection through ViewState.project."""

    def _points(self):
        rng = np.random.default_rng(42)
        return rng.normal(scale=3.0, size=(20, 3))

    def test_orthographic_unchanged(self):
        """Orthographic projection applies zoom alone, with no
        depth-dependent scaling."""
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
        _, depth_obl, _ = ViewState(projection=Oblique(45.0, 1.0)).project(pts)
        np.testing.assert_array_equal(depth_obl, depth_ortho)

    def test_radii_unchanged_by_oblique(self):
        """Spheres keep circular outlines by convention."""
        pts = self._points()
        radii = np.linspace(0.2, 1.0, len(pts))
        _, _, r_ortho = ViewState().project(pts, radii)
        _, _, r_obl = ViewState(projection=Oblique(45.0, 1.0)).project(pts, radii)
        np.testing.assert_array_equal(r_obl, r_ortho)

    def test_unit_and_half_foreshortening_steps(self):
        """A unit step along the receding axis (depth -1, i.e. away
        from the viewer) draws as a unit step on screen at
        foreshortening 1.0 and a half step at foreshortening 0.5, in
        the direction of *angle*."""
        step = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        for oblique, length in (
            (Oblique(45.0, 1.0), 1.0), (Oblique(45.0, 0.5), 0.5),
        ):
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

        vs = ViewState(projection=Oblique(35.0, 0.6))
        vs.look_along([0, -1, 0])
        xy, depth, _ = vs.project(pts)
        np.testing.assert_allclose(xy, expected_xy, atol=1e-14)
        np.testing.assert_array_equal(depth, expected_depth)

        # Guard: the wrong camera (the one the original working note
        # proposed) must not match.
        vs_wrong = ViewState(projection=Oblique(35.0, 0.6))
        vs_wrong.look_along([0, 1, 0])
        xy_wrong, depth_wrong, _ = vs_wrong.project(pts)
        assert not np.allclose(xy_wrong, expected_xy)
        assert not np.allclose(depth_wrong, expected_depth)


class TestProjectionTypes:
    """Tests for the projection mode value types."""

    def test_orthographic_instances_equal(self):
        assert Orthographic() == Orthographic()

    def test_orthographic_not_equal_to_other_modes(self):
        assert Orthographic() != Perspective()
        assert Orthographic() != Oblique(45.0, 0.5)

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
        for bad in (0.0, -0.5):
            with pytest.raises(ValueError, match="view_distance"):
                Perspective(view_distance=bad)

    def test_modes_are_frozen_with_slots(self):
        # Assigning a new attribute name must fail loudly (slots, no
        # __dict__), but the exception type is a CPython detail: 3.11
        # and 3.12 raise TypeError (the generated frozen __setattr__
        # trips over its stale pre-slots class reference), while 3.13+
        # fix that and raise FrozenInstanceError.  Assigning an
        # existing field raises FrozenInstanceError on all versions.
        # Orthographic has no fields, so only the former applies to it.
        o = Orthographic()
        with pytest.raises((TypeError, FrozenInstanceError)):
            o.anything = 1
        assert not hasattr(o, "__dict__")

        p = Perspective()
        with pytest.raises(FrozenInstanceError):  # frozen prevents field changes
            p.strength = 0.3
        assert not hasattr(p, "__dict__")

        ob = Oblique(45.0, 0.5)
        with pytest.raises(FrozenInstanceError):  # frozen prevents field changes
            ob.angle = 30.0
        assert not hasattr(ob, "__dict__")
