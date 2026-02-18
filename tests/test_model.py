"""Tests for hofmann.model — dataclasses, colour handling, and projection."""

import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
    AxesStyle,
    Bond,
    BondSpec,
    CellEdgeStyle,
    Frame,
    Polyhedron,
    PolyhedronSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    WidgetCorner,
    normalise_colour,
    resolve_atom_colours,
)


# --- normalise_colour ---


class TestNormaliseColour:
    def test_css_name(self):
        assert normalise_colour("red") == (1.0, 0.0, 0.0)

    def test_hex_string(self):
        assert normalise_colour("#00FF00") == pytest.approx((0.0, 1.0, 0.0))

    def test_grey_float_zero(self):
        assert normalise_colour(0.0) == (0.0, 0.0, 0.0)

    def test_grey_float(self):
        assert normalise_colour(0.7) == pytest.approx((0.7, 0.7, 0.7))

    def test_grey_float_one(self):
        assert normalise_colour(1.0) == (1.0, 1.0, 1.0)

    def test_rgb_tuple(self):
        assert normalise_colour((0.5, 0.3, 0.1)) == pytest.approx(
            (0.5, 0.3, 0.1)
        )

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognised colour"):
            normalise_colour("notacolour")

    def test_grey_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Grey value"):
            normalise_colour(1.5)

    def test_rgb_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3 elements"):
            normalise_colour((0.5, 0.3))  # type: ignore[arg-type]

    def test_rgb_out_of_range_raises(self):
        with pytest.raises(ValueError, match="RGB component"):
            normalise_colour((0.5, 1.5, 0.0))

    def test_rgb_list(self):
        assert normalise_colour([0.5, 0.3, 0.1]) == pytest.approx(
            (0.5, 0.3, 0.1)
        )

    def test_rgb_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3 elements"):
            normalise_colour([0.5, 0.3])  # type: ignore[arg-type]


# --- BondSpec.matches ---


class TestBondSpecMatches:
    def _spec(self, sp_a: str, sp_b: str) -> BondSpec:
        """Create a BondSpec with dummy geometry values."""
        return BondSpec(species=(sp_a, sp_b), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0)

    def test_exact_match(self):
        assert self._spec("C", "H").matches("C", "H") is True

    def test_symmetric_match(self):
        assert self._spec("C", "H").matches("H", "C") is True

    def test_no_match(self):
        assert self._spec("C", "H").matches("O", "N") is False

    def test_wildcard_star(self):
        assert self._spec("*", "H").matches("C", "H") is True

    def test_wildcard_star_symmetric(self):
        assert self._spec("*", "H").matches("H", "O") is True

    def test_wildcard_question_mark(self):
        spec = self._spec("C?", "H")
        assert spec.matches("Cu", "H") is True
        assert spec.matches("C", "H") is False  # "C" is 1 char, "C?" needs 2

    def test_both_wildcard(self):
        assert self._spec("*", "*").matches("X", "Y") is True

    def test_species_sorted(self):
        """Species tuple should be stored in sorted order."""
        spec = self._spec("H", "C")
        assert spec.species == ("C", "H")

    def test_species_already_sorted(self):
        """Already-sorted species should be unchanged."""
        spec = self._spec("C", "H")
        assert spec.species == ("C", "H")

    def test_complete_true_raises(self):
        """complete=True is not allowed — must be a species string or '*'."""
        with pytest.raises(ValueError, match="complete=True is not supported"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete=True)

    def test_complete_non_string_truthy_raises(self):
        """Non-string truthy values like 1 are rejected."""
        with pytest.raises(ValueError, match="complete must be"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete=1)  # type: ignore[arg-type]

    def test_complete_string_accepted(self):
        """A species name string is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="C")
        assert spec.complete == "C"

    def test_complete_wildcard_accepted(self):
        """'*' is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="*")
        assert spec.complete == "*"

    def test_complete_false_accepted(self):
        """False (default) is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete=False)
        assert spec.complete is False

    def test_complete_empty_string_raises(self):
        """An empty string is not a valid species name for complete."""
        with pytest.raises(ValueError, match="complete must not be an empty string"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete="")

    def test_complete_species_not_in_pair_raises(self):
        """complete='Zr' on a ('C', 'H') bond spec is rejected."""
        with pytest.raises(ValueError, match="does not match either species"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete="Zr")

    def test_complete_species_matches_one_side(self):
        """complete='Na' on ('Cl', 'Na') is accepted."""
        spec = BondSpec(species=("Na", "Cl"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="Na")
        assert spec.complete == "Na"


# --- BondSpec defaults ---


class TestBondSpecDefaults:
    def test_min_length_defaults_to_zero(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.min_length == 0.0

    def test_radius_defaults_to_class_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.radius == 0.1

    def test_colour_defaults_to_class_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.colour == 0.5

    def test_explicit_radius_overrides_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        assert spec.radius == 0.2

    def test_explicit_colour_overrides_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, colour="red")
        assert spec.colour == "red"

    def test_changing_class_default_radius(self):
        original = BondSpec.default_radius
        try:
            BondSpec.default_radius = 0.15
            spec = BondSpec(species=("C", "H"), max_length=3.4)
            assert spec.radius == 0.15
        finally:
            BondSpec.default_radius = original

    def test_changing_class_default_does_not_affect_explicit(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        original = BondSpec.default_radius
        try:
            BondSpec.default_radius = 0.99
            assert spec.radius == 0.2
        finally:
            BondSpec.default_radius = original

    def test_radius_setter(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        spec.radius = 0.3
        assert spec.radius == 0.3

    def test_radius_setter_none_reverts_to_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.3)
        spec.radius = None
        assert spec.radius == 0.1

    def test_colour_setter(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        spec.colour = "blue"
        assert spec.colour == "blue"


class TestBondSpecRepr:
    def test_default_radius_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "radius=<default 0.1>" in repr(spec)

    def test_default_colour_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "colour=<default 0.5>" in repr(spec)

    def test_explicit_radius_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        assert "radius=0.2" in repr(spec)

    def test_explicit_colour_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, colour="red")
        assert "colour='red'" in repr(spec)

    def test_complete_omitted_when_false(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "complete" not in repr(spec)

    def test_recursive_omitted_when_false(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "recursive" not in repr(spec)

    def test_complete_shown_when_set(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, complete="C")
        assert "complete='C'" in repr(spec)


class TestBondSpecEquality:
    def test_equal_with_defaults(self):
        a = BondSpec(species=("C", "H"), max_length=3.4)
        b = BondSpec(species=("C", "H"), max_length=3.4)
        assert a == b

    def test_equal_with_explicit_values(self):
        a = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2, colour="red")
        b = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2, colour="red")
        assert a == b

    def test_default_not_equal_to_explicit_same_value(self):
        a = BondSpec(species=("C", "H"), max_length=3.4)
        b = BondSpec(species=("C", "H"), max_length=3.4, radius=0.1)
        assert a != b

    def test_not_equal_to_other_type(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec != "not a spec"


# --- Bond frozen ---


class TestBond:
    def test_is_frozen(self):
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=3.0, radius=0.1, colour=1.0)
        bond = Bond(0, 1, 2.0, spec)
        with pytest.raises(AttributeError):
            bond.length = 3.0  # type: ignore[misc]


# --- Frame ---


class TestFrame:
    def test_valid_coords(self):
        coords = np.zeros((5, 3))
        frame = Frame(coords=coords, label="test")
        assert frame.coords.shape == (5, 3)

    def test_invalid_1d_raises(self):
        with pytest.raises(ValueError, match="\\(n_atoms, 3\\)"):
            Frame(coords=np.zeros(6))

    def test_invalid_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="\\(n_atoms, 3\\)"):
            Frame(coords=np.zeros((5, 2)))

    def test_coords_converted_to_float(self):
        coords = np.array([[1, 2, 3]], dtype=int)
        frame = Frame(coords=coords)
        assert frame.coords.dtype == float


# --- ViewState.project ---


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
        vs = ViewState(perspective=1.0, view_distance=10.0)
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
        vs = ViewState(perspective=1.0, view_distance=10.0)
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        _, _, proj_r = vs.project(coords, radii)
        # Silhouette: r * D / sqrt(D^2 - r^2) = 1 * 10 / sqrt(99)
        expected = 10.0 / np.sqrt(99.0)
        np.testing.assert_allclose(proj_r, [expected], rtol=1e-6)

    def test_projected_radii_larger_than_point_scale(self):
        """Silhouette radii should exceed naive r * scale under perspective."""
        vs = ViewState(perspective=1.0, view_distance=10.0)
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
        vs = ViewState(zoom=2.5, perspective=0.8, view_distance=15.0)
        vs.look_along([1, 1, 0])
        assert vs.zoom == 2.5
        assert vs.perspective == 0.8
        assert vs.view_distance == 15.0

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


# --- SlabClipMode ---


class TestSlabClipMode:
    def test_values(self):
        assert SlabClipMode.PER_FACE == "per_face"
        assert SlabClipMode.CLIP_WHOLE == "clip_whole"
        assert SlabClipMode.INCLUDE_WHOLE == "include_whole"

    def test_string_construction(self):
        assert SlabClipMode("per_face") is SlabClipMode.PER_FACE
        assert SlabClipMode("clip_whole") is SlabClipMode.CLIP_WHOLE
        assert SlabClipMode("include_whole") is SlabClipMode.INCLUDE_WHOLE


# --- RenderStyle ---


class TestRenderStyle:
    def test_slab_clip_mode_default(self):
        style = RenderStyle()
        assert style.slab_clip_mode == SlabClipMode.PER_FACE

    def test_slab_clip_mode_string_coercion(self):
        style = RenderStyle(slab_clip_mode="include_whole")
        assert style.slab_clip_mode is SlabClipMode.INCLUDE_WHOLE

    def test_slab_clip_mode_enum_value(self):
        style = RenderStyle(slab_clip_mode=SlabClipMode.CLIP_WHOLE)
        assert style.slab_clip_mode is SlabClipMode.CLIP_WHOLE

    def test_polyhedra_outline_width_default_none(self):
        style = RenderStyle()
        assert style.polyhedra_outline_width is None

    def test_polyhedra_outline_width_override(self):
        style = RenderStyle(polyhedra_outline_width=2.0)
        assert style.polyhedra_outline_width == 2.0

    @pytest.mark.parametrize("field, value", [
        ("atom_scale", 0),
        ("atom_scale", -1.0),
        ("bond_scale", 0),
        ("bond_scale", -0.5),
    ])
    def test_positive_scale_required(self, field, value):
        with pytest.raises(ValueError, match="must be positive"):
            RenderStyle(**{field: value})

    @pytest.mark.parametrize("field, value", [
        ("atom_outline_width", -1.0),
        ("bond_outline_width", -0.1),
        ("polyhedra_outline_width", -0.5),
    ])
    def test_non_negative_width_required(self, field, value):
        with pytest.raises(ValueError, match="must be non-negative"):
            RenderStyle(**{field: value})

    def test_circle_segments_default(self):
        assert RenderStyle().circle_segments == 72

    def test_arc_segments_default(self):
        assert RenderStyle().arc_segments == 12

    def test_circle_segments_minimum(self):
        with pytest.raises(ValueError, match="circle_segments must be >= 3"):
            RenderStyle(circle_segments=2)

    def test_arc_segments_minimum(self):
        with pytest.raises(ValueError, match="arc_segments must be >= 2"):
            RenderStyle(arc_segments=1)

    def test_interactive_circle_segments_default(self):
        assert RenderStyle().interactive_circle_segments == 24

    def test_interactive_arc_segments_default(self):
        assert RenderStyle().interactive_arc_segments == 5

    def test_interactive_circle_segments_minimum(self):
        with pytest.raises(ValueError, match="interactive_circle_segments must be >= 3"):
            RenderStyle(interactive_circle_segments=2)

    def test_interactive_arc_segments_minimum(self):
        with pytest.raises(ValueError, match="interactive_arc_segments must be >= 2"):
            RenderStyle(interactive_arc_segments=1)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_polyhedra_shading_range(self, value):
        with pytest.raises(ValueError, match="polyhedra_shading must be between"):
            RenderStyle(polyhedra_shading=value)

    def test_valid_boundary_values_accepted(self):
        """Edge values at the boundaries should not raise."""
        RenderStyle(
            atom_scale=0.01,
            bond_scale=0.01,
            atom_outline_width=0.0,
            bond_outline_width=0.0,
            polyhedra_shading=0.0,
            polyhedra_outline_width=0.0,
        )
        RenderStyle(polyhedra_shading=1.0)

    def test_show_cell_default_none(self):
        style = RenderStyle()
        assert style.show_cell is None

    def test_cell_style_default(self):
        style = RenderStyle()
        assert isinstance(style.cell_style, CellEdgeStyle)
        assert style.cell_style.linestyle == "solid"

    def test_cell_style_override(self):
        cs = CellEdgeStyle(colour="red", line_width=2.0)
        style = RenderStyle(cell_style=cs)
        assert style.cell_style.colour == "red"

    def test_show_axes_default_none(self):
        style = RenderStyle()
        assert style.show_axes is None

    def test_axes_style_default(self):
        style = RenderStyle()
        assert isinstance(style.axes_style, AxesStyle)
        assert style.axes_style.corner is WidgetCorner.BOTTOM_LEFT

    def test_axes_style_override(self):
        ws = AxesStyle(corner="top_right", font_size=14.0)
        style = RenderStyle(axes_style=ws)
        assert style.axes_style.corner is WidgetCorner.TOP_RIGHT
        assert style.axes_style.font_size == 14.0


# --- StructureScene ---


class TestStructureScene:
    def test_defaults(self):
        coords = np.zeros((2, 3))
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        assert scene.atom_styles == {}
        assert scene.bond_specs == []
        assert scene.polyhedra == []
        assert scene.title == ""
        np.testing.assert_array_equal(scene.view.rotation, np.eye(3))

    def test_centre_on(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        scene.centre_on(1)
        np.testing.assert_array_equal(scene.view.centre, [4.0, 5.0, 6.0])

    def test_centre_on_does_not_alias_coords(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        scene.centre_on(0)
        scene.view.centre[0] = 999.0
        assert scene.frames[0].coords[0, 0] == 1.0

    def test_lattice_default_none(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(species=["A"], frames=[Frame(coords=coords)])
        assert scene.lattice is None

    def test_lattice_accepted(self):
        coords = np.zeros((1, 3))
        lat = np.eye(3) * 5.0
        scene = StructureScene(
            species=["A"], frames=[Frame(coords=coords)], lattice=lat,
        )
        np.testing.assert_array_equal(scene.lattice, lat)

    def test_lattice_bad_shape_raises(self):
        coords = np.zeros((1, 3))
        with pytest.raises(ValueError, match="shape"):
            StructureScene(
                species=["A"], frames=[Frame(coords=coords)],
                lattice=np.eye(2),
            )

    def test_lattice_coerced_to_float(self):
        coords = np.zeros((1, 3))
        lat_int = np.eye(3, dtype=int) * 5
        scene = StructureScene(
            species=["A"], frames=[Frame(coords=coords)], lattice=lat_int,
        )
        assert scene.lattice.dtype == float

    def test_atom_data_default_empty(self):
        coords = np.zeros((2, 3))
        scene = StructureScene(
            species=["A", "B"], frames=[Frame(coords=coords)],
        )
        assert scene.atom_data == {}

    def test_atom_data_constructor_validation(self):
        coords = np.zeros((2, 3))
        with pytest.raises(ValueError, match="atom_data"):
            StructureScene(
                species=["A", "B"], frames=[Frame(coords=coords)],
                atom_data={"charge": np.array([1.0, 2.0, 3.0])},
            )


class TestSetAtomData:
    """Tests for StructureScene.set_atom_data."""

    def _scene(self, n: int = 3) -> StructureScene:
        coords = np.zeros((n, 3))
        return StructureScene(
            species=["A", "B", "C"][:n],
            frames=[Frame(coords=coords)],
        )

    def test_full_array(self):
        scene = self._scene()
        values = np.array([1.0, 2.0, 3.0])
        scene.set_atom_data("charge", values)
        np.testing.assert_array_equal(scene.atom_data["charge"], values)

    def test_full_array_list(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            scene.atom_data["charge"], [1.0, 2.0, 3.0],
        )

    def test_categorical_array(self):
        scene = self._scene()
        scene.set_atom_data("site", np.array(["4a", "8b", "4a"], dtype=object))
        assert list(scene.atom_data["site"]) == ["4a", "8b", "4a"]

    def test_wrong_length_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="length 3"):
            scene.set_atom_data("charge", np.array([1.0, 2.0]))

    def test_sparse_dict_numeric(self):
        scene = self._scene()
        scene.set_atom_data("charge", {0: 1.5, 2: -0.3})
        arr = scene.atom_data["charge"]
        assert arr[0] == pytest.approx(1.5)
        assert np.isnan(arr[1])
        assert arr[2] == pytest.approx(-0.3)

    def test_sparse_dict_string(self):
        scene = self._scene()
        scene.set_atom_data("site", {1: "4a"})
        arr = scene.atom_data["site"]
        assert arr[0] == ""
        assert arr[1] == "4a"
        assert arr[2] == ""

    def test_sparse_dict_out_of_range_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="out of range"):
            scene.set_atom_data("charge", {5: 1.0})

    def test_sparse_dict_empty_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="must not be empty"):
            scene.set_atom_data("charge", {})

    def test_overwrite_existing_key(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.set_atom_data("charge", [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(
            scene.atom_data["charge"], [4.0, 5.0, 6.0],
        )

    def test_multiple_keys(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.set_atom_data("site", np.array(["a", "b", "c"], dtype=object))
        assert "charge" in scene.atom_data
        assert "site" in scene.atom_data

    def test_sparse_dict_mixed_types_raises(self):
        """Dict with mixed string and numeric values raises TypeError."""
        scene = self._scene()
        with pytest.raises(TypeError, match="same type"):
            scene.set_atom_data("bad", {0: 1, 2: "text"})


# --- resolve_atom_colours ---


class TestResolveAtomColours:
    """Tests for resolve_atom_colours."""

    SPECIES = ["C", "H", "O"]
    STYLES: dict = {
        "C": AtomStyle(radius=1.0, colour=(0.4, 0.4, 0.4)),
        "H": AtomStyle(radius=0.7, colour=(1.0, 1.0, 1.0)),
        "O": AtomStyle(radius=0.8, colour=(0.6, 0.0, 0.0)),
    }

    def test_species_fallback(self):
        """colour_by=None returns species-based colours."""
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, {},
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_species_fallback_missing_style(self):
        """Missing species falls back to grey."""
        result = resolve_atom_colours(
            ["X"], {}, {},
        )
        assert result == [(0.5, 0.5, 0.5)]

    def test_numerical_viridis_endpoints(self):
        """Known endpoints of the viridis colourmap."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        expected_0 = cmap(0.0)[:3]
        expected_1 = cmap(1.0)[:3]

        data = {"val": np.array([0.0, 1.0])}
        result = resolve_atom_colours(
            ["A", "B"], self.STYLES, data,
            colour_by="val", cmap="viridis",
        )
        assert result[0] == pytest.approx(expected_0)
        assert result[1] == pytest.approx(expected_1)

    def test_numerical_custom_range(self):
        """Explicit colour_range normalises correctly."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        # value 5 with range (0, 10) -> normalised 0.5
        expected = cmap(0.5)[:3]

        data = {"val": np.array([5.0])}
        result = resolve_atom_colours(
            ["A"], self.STYLES, data,
            colour_by="val", colour_range=(0.0, 10.0),
        )
        assert result[0] == pytest.approx(expected)

    def test_numerical_constant_values(self):
        """All-same values should map to 0.5 (no division by zero)."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        expected = cmap(0.5)[:3]

        data = {"val": np.array([3.0, 3.0, 3.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        for c in result:
            assert c == pytest.approx(expected)

    def test_numerical_nan_falls_back(self):
        """NaN entries get their species colour."""
        data = {"val": np.array([0.0, np.nan, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)

    def test_numerical_all_nan(self):
        """All NaN returns species colours."""
        data = {"val": np.array([np.nan, np.nan, np.nan])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_categorical_distinct_colours(self):
        """Two categories get two different colours."""
        data = {"site": np.array(["4a", "8b", "4a"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Atoms 0 and 2 (both "4a") should have the same colour.
        assert result[0] == result[2]
        # Atom 1 ("8b") should differ.
        assert result[1] != result[0]

    def test_categorical_empty_falls_back(self):
        """Empty string entries get their species colour."""
        data = {"site": np.array(["4a", "", "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H, empty label) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)

    def test_callable_cmap(self):
        """A callable cmap is used directly."""
        def red_blue(val: float) -> tuple[float, float, float]:
            return (val, 0.0, 1.0 - val)

        data = {"val": np.array([0.0, 0.5, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap=red_blue,
        )
        assert result[0] == pytest.approx((0.0, 0.0, 1.0))
        assert result[1] == pytest.approx((0.5, 0.0, 0.5))
        assert result[2] == pytest.approx((1.0, 0.0, 0.0))

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, {},
                colour_by="nonexistent",
            )

    def test_list_colour_by_priority(self):
        """Non-overlapping layers: each atom gets its layer's colour."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[red, blue],
        )
        assert result[0] == (1.0, 0.0, 0.0)  # from layer "a"
        assert result[1] == (0.0, 0.0, 1.0)  # from layer "b"
        # Atom 2 has NaN in both — species fallback (O)
        assert result[2] == (0.6, 0.0, 0.0)

    def test_list_colour_by_first_wins(self):
        """When an atom has data in multiple layers, first wins."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([2.0, 2.0, np.nan]),  # atom 0 in both
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[red, blue],
        )
        # Atom 0: layer "a" has data, so red wins over blue.
        assert result[0] == (1.0, 0.0, 0.0)
        assert result[1] == (0.0, 0.0, 1.0)

    def test_list_colour_by_broadcast_cmap(self):
        """A single cmap string is broadcast to all layers."""
        data = {
            "a": np.array([0.0, np.nan, np.nan]),
            "b": np.array([np.nan, 1.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap="viridis",
        )
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        assert result[0] == pytest.approx(cmap(0.5)[:3])  # constant -> 0.5
        assert result[1] == pytest.approx(cmap(0.5)[:3])

    def test_list_colour_by_all_missing_falls_back(self):
        """Atom with NaN in all layers gets species colour."""
        data = {
            "a": np.array([np.nan, np.nan, np.nan]),
            "b": np.array([np.nan, np.nan, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"],
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_list_colour_by_categorical(self):
        """List colour_by works with categorical layers."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "metal": np.array(["Fe", "", ""], dtype=object),
            "anion": np.array(["", "O", ""], dtype=object),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["metal", "anion"], cmap=[red, blue],
        )
        assert result[0] == (1.0, 0.0, 0.0)  # Fe
        assert result[1] == (0.0, 0.0, 1.0)  # O
        assert result[2] == (0.6, 0.0, 0.0)  # fallback

    def test_numerical_integer_array(self):
        """Integer arrays are coerced to float without error."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]

        data = {"val": np.array([1, 2, 3])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        # 1->0.0, 2->0.5, 3->1.0 after normalisation
        assert result[0] == pytest.approx(cmap(0.0)[:3])
        assert result[1] == pytest.approx(cmap(0.5)[:3])
        assert result[2] == pytest.approx(cmap(1.0)[:3])

    def test_categorical_nan_falls_back(self):
        """np.nan in categorical data is treated as missing."""
        data = {"site": np.array(["4a", np.nan, "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)
        assert result[2] != (0.6, 0.0, 0.0)

    def test_categorical_none_falls_back(self):
        """None in categorical data is treated as missing."""
        data = {"site": np.array(["4a", None, "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)
        assert result[2] != (0.6, 0.0, 0.0)

    def test_list_colour_by_mismatched_cmap_length(self):
        """Mismatched cmap list length raises ValueError."""
        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        with pytest.raises(ValueError, match="cmap"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by=["a", "b"],
                cmap=["viridis", "plasma", "inferno"],
            )

    def test_list_colour_by_mismatched_colour_range_length(self):
        """Mismatched colour_range list length raises ValueError."""
        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        with pytest.raises(ValueError, match="colour_range"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by=["a", "b"],
                colour_range=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
            )

    def test_list_colour_by_fallback_colour_collision(self):
        """Merge uses data masks, not colour equality with fallback.

        If a cmap returns the same RGB as the species fallback, the
        atom should still get the cmap colour (not be skipped).
        """
        # C's species colour is (0.4, 0.4, 0.4).  Create a cmap that
        # always returns exactly (0.4, 0.4, 0.4).
        def grey_cmap(_v: float) -> tuple[float, float, float]:
            return (0.4, 0.4, 0.4)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            # Layer "a" has data for atom 0 (C) — cmap will return same
            # as species colour.
            "a": np.array([1.0, np.nan, np.nan]),
            # Layer "b" has data for atom 1.
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[grey_cmap, blue],
        )
        # Atom 0 has data in layer "a", so it should NOT fall through
        # to layer "b".  With colour-equality merge it would be treated
        # as missing because grey_cmap returns the same as fallback.
        assert result[0] == (0.4, 0.4, 0.4)  # from grey_cmap
        assert result[1] == (0.0, 0.0, 1.0)  # from blue
        assert result[2] == (0.6, 0.0, 0.0)  # fallback (O)

    def test_single_key_list_cmap_raises(self):
        """List cmap with single colour_by string raises ValueError."""
        data = {"val": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="cmap"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by="val", cmap=["viridis"],
            )

    def test_single_key_list_colour_range_raises(self):
        """List colour_range with single colour_by string raises ValueError."""
        data = {"val": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="colour_range"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by="val", colour_range=[(0.0, 1.0)],
            )

    def test_colormap_object_returns_rgb(self):
        """Passing a Colormap object produces 3-tuple (r, g, b) colours."""
        import matplotlib
        cmap_obj = matplotlib.colormaps["viridis"]

        data = {"val": np.array([0.0, 0.5, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap=cmap_obj,
        )
        for colour in result:
            assert len(colour) == 3
        # Check values match the string-based lookup.
        expected = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap="viridis",
        )
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)


# --- CellEdgeStyle ---


class TestCellEdgeStyle:
    def test_defaults(self):
        style = CellEdgeStyle()
        assert style.colour == (0.3, 0.3, 0.3)
        assert style.line_width == 0.8
        assert style.linestyle == "solid"

    def test_custom_values(self):
        style = CellEdgeStyle(colour="blue", line_width=1.5, linestyle="dashed")
        assert style.colour == "blue"
        assert style.line_width == 1.5
        assert style.linestyle == "dashed"

    def test_negative_line_width_raises(self):
        with pytest.raises(ValueError, match="line_width must be non-negative"):
            CellEdgeStyle(line_width=-0.1)

    def test_invalid_linestyle_raises(self):
        with pytest.raises(ValueError, match="linestyle must be one of"):
            CellEdgeStyle(linestyle="wavy")

    def test_is_frozen(self):
        style = CellEdgeStyle()
        with pytest.raises(AttributeError):
            style.line_width = 2.0

    @pytest.mark.parametrize("ls", ["solid", "dashed", "dotted", "dashdot"])
    def test_valid_linestyles(self, ls):
        style = CellEdgeStyle(linestyle=ls)
        assert style.linestyle == ls


# --- WidgetCorner ---


class TestWidgetCorner:
    def test_values(self):
        assert WidgetCorner.BOTTOM_LEFT == "bottom_left"
        assert WidgetCorner.BOTTOM_RIGHT == "bottom_right"
        assert WidgetCorner.TOP_LEFT == "top_left"
        assert WidgetCorner.TOP_RIGHT == "top_right"

    def test_string_construction(self):
        assert WidgetCorner("bottom_left") is WidgetCorner.BOTTOM_LEFT
        assert WidgetCorner("top_right") is WidgetCorner.TOP_RIGHT


# --- AxesStyle ---


class TestAxesStyle:
    def test_defaults(self):
        style = AxesStyle()
        assert len(style.colours) == 3
        assert all(c == (0.3, 0.3, 0.3) for c in style.colours)
        assert style.labels == ("a", "b", "c")
        assert style.font_size == 10.0
        assert style.italic is True
        assert style.arrow_length == 0.12
        assert style.line_width == 1.0
        assert style.corner is WidgetCorner.BOTTOM_LEFT
        assert style.margin == 0.15

    def test_custom_values(self):
        style = AxesStyle(
            colours=("red", "green", "blue"),
            labels=("x", "y", "z"),
            font_size=12.0,
            italic=False,
            arrow_length=0.1,
            corner="top_right",
        )
        assert style.colours == ("red", "green", "blue")
        assert style.labels == ("x", "y", "z")
        assert style.font_size == 12.0
        assert style.italic is False
        assert style.arrow_length == 0.1
        assert style.corner is WidgetCorner.TOP_RIGHT

    def test_is_frozen(self):
        style = AxesStyle()
        with pytest.raises(AttributeError):
            style.font_size = 14.0

    def test_corner_string_coercion(self):
        style = AxesStyle(corner="bottom_right")
        assert style.corner is WidgetCorner.BOTTOM_RIGHT

    def test_negative_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            AxesStyle(font_size=-1.0)

    def test_zero_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            AxesStyle(font_size=0)

    def test_negative_arrow_length_raises(self):
        with pytest.raises(ValueError, match="arrow_length must be positive"):
            AxesStyle(arrow_length=-0.1)

    def test_negative_line_width_raises(self):
        with pytest.raises(ValueError, match="line_width must be non-negative"):
            AxesStyle(line_width=-1.0)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin must be non-negative"):
            AxesStyle(margin=-0.1)

    def test_wrong_number_of_colours_raises(self):
        with pytest.raises(ValueError, match="colours must have exactly 3"):
            AxesStyle(colours=((1, 0, 0), (0, 1, 0)))

    def test_wrong_number_of_labels_raises(self):
        with pytest.raises(ValueError, match="labels must have exactly 3"):
            AxesStyle(labels=("a", "b"))

    def test_corner_tuple(self):
        style = AxesStyle(corner=(0.1, 0.9))
        assert style.corner == (0.1, 0.9)

    def test_corner_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="corner tuple must have 2 elements"):
            AxesStyle(corner=(0.1, 0.2, 0.3))


# --- PolyhedronSpec ---


class TestPolyhedronSpec:
    def test_defaults(self):
        spec = PolyhedronSpec(centre="Ti")
        assert spec.centre == "Ti"
        assert spec.colour is None
        assert spec.alpha == 0.4
        assert spec.edge_colour == (0.15, 0.15, 0.15)
        assert spec.edge_width == 1.0
        assert spec.hide_centre is False
        assert spec.hide_bonds is False
        assert spec.hide_vertices is False
        assert spec.min_vertices is None

    def test_custom_values(self):
        spec = PolyhedronSpec(
            centre="Si",
            colour=(0.2, 0.4, 0.8),
            alpha=0.6,
            edge_colour="black",
            edge_width=1.5,
            hide_centre=True,
            hide_bonds=True,
            hide_vertices=True,
            min_vertices=6,
        )
        assert spec.centre == "Si"
        assert spec.colour == (0.2, 0.4, 0.8)
        assert spec.alpha == 0.6
        assert spec.hide_centre is True
        assert spec.hide_bonds is True
        assert spec.hide_vertices is True
        assert spec.min_vertices == 6


# --- Polyhedron ---


class TestPolyhedron:
    def test_is_frozen(self):
        faces = [np.array([0, 1, 2])]
        spec = PolyhedronSpec(centre="Ti")
        poly = Polyhedron(
            centre_index=0,
            neighbour_indices=(1, 2, 3),
            faces=faces,
            spec=spec,
        )
        with pytest.raises(AttributeError):
            poly.centre_index = 5  # type: ignore[misc]

    def test_fields(self):
        faces = [np.array([0, 1, 2]), np.array([0, 2, 3])]
        spec = PolyhedronSpec(centre="Ti")
        poly = Polyhedron(
            centre_index=0,
            neighbour_indices=(1, 2, 3, 4),
            faces=faces,
            spec=spec,
        )
        assert poly.centre_index == 0
        assert poly.neighbour_indices == (1, 2, 3, 4)
        assert len(poly.faces) == 2
        assert all(len(f) == 3 for f in poly.faces)
        assert poly.spec is spec
