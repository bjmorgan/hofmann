"""Tests for hofmann.model — dataclasses, colour handling, and projection."""

import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
    Bond,
    BondSpec,
    Frame,
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


# --- BondSpec.matches ---


class TestBondSpecMatches:
    def _spec(self, sp_a: str, sp_b: str) -> BondSpec:
        """Create a BondSpec with dummy geometry values."""
        return BondSpec(sp_a, sp_b, 0.0, 5.0, 0.1, 1.0)

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


# --- Bond frozen ---


class TestBond:
    def test_is_frozen(self):
        spec = BondSpec("C", "H", 0.0, 3.0, 0.1, 1.0)
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
        assert scene.title == ""
        np.testing.assert_array_equal(scene.view.rotation, np.eye(3))
