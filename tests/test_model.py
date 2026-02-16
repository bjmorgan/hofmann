"""Tests for hofmann.model — dataclasses, colour handling, and projection."""

import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
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
    normalise_colour,
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

    def test_circle_segments_minimum(self):
        with pytest.raises(ValueError, match="circle_segments must be >= 3"):
            RenderStyle(circle_segments=2)

    def test_arc_segments_minimum(self):
        with pytest.raises(ValueError, match="arc_segments must be >= 2"):
            RenderStyle(arc_segments=1)

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


# --- PolyhedronSpec ---


class TestPolyhedronSpec:
    def test_defaults(self):
        spec = PolyhedronSpec(centre="Ti")
        assert spec.centre == "Ti"
        assert spec.colour is None
        assert spec.alpha == 0.4
        assert spec.edge_colour == (0.15, 0.15, 0.15)
        assert spec.edge_width == 0.8
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
