"""Tests for ViewState projection, look_along, slab clipping, and validation."""

import dataclasses
import warnings

import numpy as np
import pytest

from hofmann.model.projection import Orthographic, Perspective, Projection
from hofmann.model.view_state import ViewState


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

    def test_zoom_scales_perspective_positions(self):
        """Zoom must apply under perspective, not only orthographic."""
        coords = np.array([[1.0, 2.0, 3.0]])
        plain = ViewState(projection=Perspective(0.7, 12.0))
        zoomed = ViewState(zoom=2.5, projection=Perspective(0.7, 12.0))
        xy_plain, _, _ = plain.project(coords)
        xy_zoomed, _, _ = zoomed.project(coords)
        np.testing.assert_allclose(xy_zoomed, xy_plain * 2.5)

    def test_zoom_scales_perspective_radii(self):
        coords = np.array([[0.0, 0.0, 1.0]])
        radii = np.array([0.8])
        plain = ViewState(projection=Perspective(0.7, 12.0))
        zoomed = ViewState(zoom=2.5, projection=Perspective(0.7, 12.0))
        _, _, r_plain = plain.project(coords, radii)
        _, _, r_zoomed = zoomed.project(coords, radii)
        np.testing.assert_allclose(r_zoomed, r_plain * 2.5)

    def test_points_at_or_behind_the_eye_plane_warn(self):
        """Behind-eye points draw mirrored and sort frontmost."""
        vs = ViewState(projection=Perspective(1.0, 10.0))
        with pytest.warns(UserWarning, match="eye plane"):
            vs.project(np.array([[1.0, 0.0, 11.0]]))

    def test_no_eye_plane_warning_for_ordinary_depths(self):
        vs = ViewState(projection=Perspective(1.0, 10.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vs.project(np.array([[1.0, 0.0, 2.0]]))

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


class TestViewStateValidation:
    def test_zero_zoom_raises(self):
        with pytest.raises(ValueError, match="zoom"):
            ViewState(zoom=0.0)

    def test_negative_zoom_raises(self):
        with pytest.raises(ValueError, match="zoom"):
            ViewState(zoom=-1.0)

    def test_valid_view_state_accepted(self):
        vs = ViewState(zoom=2.0, projection=Perspective(0.5, 15.0))
        assert vs.zoom == 2.0
        assert vs.projection.view_distance == 15.0

    def test_projection_defaults_to_orthographic(self):
        assert ViewState().projection == Orthographic()


class TestProjectionTypes:
    def test_perspective_defaults(self):
        p = Perspective()
        assert p.strength == 0.5
        assert p.view_distance == 10.0

    @pytest.mark.parametrize(
        "strength", [-0.5, float("nan"), float("inf")],
    )
    def test_invalid_strength_rejected(self, strength):
        with pytest.raises(ValueError, match="strength"):
            Perspective(strength=strength)

    @pytest.mark.parametrize(
        "view_distance", [0.0, -1.0, float("nan"), float("inf")],
    )
    def test_invalid_view_distance_rejected(self, view_distance):
        with pytest.raises(ValueError, match="view_distance"):
            Perspective(view_distance=view_distance)

    def test_zero_strength_directs_the_caller_to_orthographic(self):
        """A parallel projection is a different type, not a zero strength."""
        with pytest.raises(ValueError, match="Orthographic"):
            Perspective(strength=0.0)

    def test_modes_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Perspective().strength = 0.9

    def test_modes_reject_unknown_attributes(self):
        """slots keeps a typo loud rather than setting an inert attribute.

        The exception type is a CPython detail that moved: up to 3.12
        the slots check fires first and raises ``TypeError``, from 3.13
        the frozen check fires first and raises
        ``FrozenInstanceError``.  What matters here is that the write
        is refused and leaves nothing behind, so both are accepted.
        """
        mode = Orthographic()
        with pytest.raises((AttributeError, TypeError)):
            mode.anything = 1
        assert not hasattr(mode, "anything")
        # frozen alone would satisfy the raise above, so pin the slots
        # that make an unknown attribute unstorable in the first place.
        assert not hasattr(mode, "__dict__")
        assert not hasattr(Perspective(), "__dict__")

    def test_setters_return_self_for_chaining(self):
        vs = ViewState()
        assert vs.set_perspective() is vs
        assert vs.set_orthographic() is vs

    def test_setters_select_the_mode(self):
        vs = ViewState().set_perspective(0.8, 15.0)
        assert vs.projection == Perspective(0.8, 15.0)
        assert vs.set_orthographic().projection == Orthographic()

    def test_setter_validates_through_the_type(self):
        with pytest.raises(ValueError, match="strength"):
            ViewState().set_perspective(strength=-1.0)

    def test_modes_compare_by_value(self):
        assert Perspective(0.5, 10.0) == Perspective(0.5, 10.0)
        assert Perspective(0.5, 10.0) != Perspective(0.6, 10.0)
        assert Orthographic() == Orthographic()

    def test_projection_enforces_every_method(self):
        """A variant omitting any one of the five cannot be instantiated."""
        assert Projection.__abstractmethods__ == frozenset({
            "to_screen",
            "silhouette_radius",
            "max_magnification",
            "eye_distance",
            "reaches_eye_plane",
        })

        class Incomplete(Projection):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_max_magnification_passes_small_positive_denominators(self):
        """A denominator in (0, 1e-6) is not clamped.

        Pins the ``denom if denom > 0 else 1e-6`` form against the
        tempting ``max(denom, 1e-6)``, which would flatten this band:
        with strength 1 and view_distance 10, worst_depth 10 - 5e-7
        leaves denom = 5e-7, so the magnification is ~2e7, not the 1e7
        that clamping to 1e-6 would give.
        """
        persp = Perspective(strength=1.0, view_distance=10.0)
        mag = persp.max_magnification(10.0 - 5e-7)
        assert mag == pytest.approx(2e7, rel=1e-3)
        assert mag > 1.5e7  # would be 1e7 if the denominator were clamped

    def test_orthographic_answers_its_contract(self):
        ortho = Orthographic()
        assert ortho.eye_distance == 1e6
        assert ortho.reaches_eye_plane(np.array([1e9])) is False
        np.testing.assert_array_equal(
            ortho.to_screen(np.array([[2.0, 3.0, 5.0]])), [[2.0, 3.0]]
        )
        np.testing.assert_array_equal(
            ortho.silhouette_radius(np.array([5.0]), np.array([1.5])), [1.5]
        )
        assert ortho.max_magnification(100.0) == 1.0

    def test_perspective_answers_its_contract(self):
        persp = Perspective(1.0, 10.0)
        assert persp.eye_distance == 10.0
        assert persp.reaches_eye_plane(np.array([9.0])) is False
        assert persp.reaches_eye_plane(np.array([11.0])) is True
        # depth 5: d = 10 - 5 = 5, scale = 10/5 = 2
        np.testing.assert_allclose(
            persp.to_screen(np.array([[1.0, 0.0, 5.0]])), [[2.0, 0.0]]
        )
        # silhouette: r*D/sqrt(D^2 - r^2) = 10/sqrt(99)
        np.testing.assert_allclose(
            persp.silhouette_radius(np.array([0.0]), np.array([1.0])),
            [10.0 / np.sqrt(99.0)],
        )
        # worst depth 5: D/(D - 5*s) = 10/5 = 2
        assert persp.max_magnification(5.0) == 2.0

    def test_perspective_to_screen_not_clamped_at_or_behind_eye(self):
        """At/behind the eye, positions blow up rather than clamp."""
        persp = Perspective(1.0, 10.0)
        on_plane = persp.to_screen(np.array([[1.0, 0.0, 10.0]]))  # depth = D
        assert not np.all(np.isfinite(on_plane))
        behind = persp.to_screen(np.array([[1.0, 0.0, 11.0]]))  # depth > D
        assert behind[0, 0] < 0  # mirrored through the origin

    def test_silhouette_converges_to_orthographic_as_strength_falls(self):
        depth = np.array([2.0])
        radii = np.array([1.5])
        ortho = Orthographic().silhouette_radius(depth, radii)  # == radii
        r_strong = Perspective(1.0, 10.0).silhouette_radius(depth, radii)
        r_weak = Perspective(0.01, 10.0).silhouette_radius(depth, radii)
        assert abs(r_weak[0] - ortho[0]) < abs(r_strong[0] - ortho[0])
        np.testing.assert_allclose(
            Perspective(1e-6, 10.0).silhouette_radius(depth, radii),
            ortho, rtol=1e-3,
        )

    def test_silhouette_finite_at_huge_view_distance(self):
        r = Perspective(1.0, 1e200).silhouette_radius(
            np.array([0.0]), np.array([1.5])
        )
        assert np.all(np.isfinite(r)) and np.all(r > 0)

    def test_silhouette_finite_for_atoms_behind_the_eye(self):
        r = Perspective(1.0, 10.0).silhouette_radius(
            np.array([20.0]), np.array([1.5])
        )
        assert np.all(np.isfinite(r))

    def test_sphere_containing_the_eye_warns(self):
        with pytest.warns(UserWarning, match="contains the perspective eye"):
            Perspective(1.0, 10.0).silhouette_radius(
                np.array([9.5]), np.array([1.0])
            )

    def test_ordinary_atom_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Perspective(1.0, 10.0).silhouette_radius(
                np.array([0.0]), np.array([1.0])
            )

    def test_eye_distance_is_the_effective_pinhole(self):
        assert Perspective(1.0, 10.0).eye_distance == 10.0
        assert Perspective(0.5, 5.0).eye_distance == 10.0
        assert Perspective(0.5, 10.0).eye_distance == 20.0
