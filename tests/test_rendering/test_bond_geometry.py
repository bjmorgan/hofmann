"""Tests for bond geometry helpers — clipping, stick polygons, and batch operations."""

import numpy as np

from hofmann.model import BondSpec, ViewState
from hofmann.rendering.bond_geometry import (
    _bond_polygon,
    _bond_polygons_batch,
    _clip_bond_3d,
    _clip_polygon_to_half_plane,
    _half_bond_verts_batch,
    _stick_polygon,
)
from hofmann.rendering.interactive import _rotation_x, _rotation_y


class TestClipBond3d:
    def test_symmetric_clipping(self):
        """Two identical atoms should clip symmetrically."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([4.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        # Offset from each atom: sqrt(1.0 - 0.01) ~ 0.99499
        w = np.sqrt(1.0 - 0.01)
        np.testing.assert_allclose(clip_start, [w, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(clip_end, [4.0 - w, 0.0, 0.0], atol=1e-10)

    def test_overlapping_atoms_returns_none(self):
        """If atoms overlap so much that clipping leaves nothing, return None."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([1.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is None

    def test_coincident_atoms_returns_none(self):
        p_a = np.array([0.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_a.copy(), r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is None

    def test_bond_radius_larger_than_atom(self):
        """When bond radius >= atom radius, offset is zero (no clipping)."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([4.0, 0.0, 0.0])
        result = _clip_bond_3d(p_a, p_b, r_a=0.05, r_b=0.05, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        np.testing.assert_allclose(clip_start, p_a)
        np.testing.assert_allclose(clip_end, p_b)

    def test_diagonal_bond(self):
        """Clipping should work along any direction, not just axes."""
        p_a = np.array([0.0, 0.0, 0.0])
        p_b = np.array([3.0, 4.0, 0.0])  # length = 5
        result = _clip_bond_3d(p_a, p_b, r_a=1.0, r_b=1.0, bond_r=0.1)
        assert result is not None
        clip_start, clip_end = result
        w = np.sqrt(1.0 - 0.01)
        unit = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(clip_start, unit * w, atol=1e-10)
        np.testing.assert_allclose(clip_end, p_b - unit * w, atol=1e-10)


class TestStickPolygon:
    def test_horizontal_bond(self):
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        verts = _stick_polygon(start, end, hw_start=0.5, hw_end=0.5)
        assert verts is not None
        assert verts.shape == (4, 2)

    def test_tapered_bond(self):
        """Different half-widths should produce a trapezoid."""
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        verts = _stick_polygon(start, end, hw_start=0.5, hw_end=0.3)
        assert verts is not None
        # Start end should be wider than end
        start_width = np.linalg.norm(verts[0] - verts[1])
        end_width = np.linalg.norm(verts[2] - verts[3])
        assert start_width > end_width

    def test_zero_length_returns_none(self):
        start = np.array([1.0, 1.0])
        verts = _stick_polygon(start, start.copy(), hw_start=0.5, hw_end=0.5)
        assert verts is None


class TestClipPolygonToHalfPlane:
    def test_square_split_vertically(self):
        """Splitting a unit square along x=0.5 should give two rectangles."""
        square = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        point = np.array([0.5, 0.0])
        normal = np.array([1.0, 0.0])  # keep x >= 0.5
        clipped = _clip_polygon_to_half_plane(square, point, normal)
        assert len(clipped) == 4
        assert np.all(clipped[:, 0] >= 0.5 - 1e-10)

    def test_square_split_gives_complementary_halves(self):
        """The two halves should have equal area."""
        square = np.array([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
        ])
        point = np.array([1.0, 0.0])
        normal = np.array([1.0, 0.0])
        right = _clip_polygon_to_half_plane(square, point, normal)
        left = _clip_polygon_to_half_plane(square, point, -normal)
        # Both halves should be rectangles with area 2.0.
        def shoelace(v):
            x, y = v[:, 0], v[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        np.testing.assert_allclose(shoelace(right), 2.0, atol=1e-10)
        np.testing.assert_allclose(shoelace(left), 2.0, atol=1e-10)

    def test_fully_inside(self):
        """Polygon entirely inside the half-plane is returned unchanged."""
        tri = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]])
        point = np.array([0.0, 0.0])
        normal = np.array([1.0, 0.0])
        clipped = _clip_polygon_to_half_plane(tri, point, normal)
        np.testing.assert_allclose(clipped, tri)

    def test_fully_outside(self):
        """Polygon entirely outside returns empty."""
        tri = np.array([[-2.0, 0.0], [-1.0, 0.0], [-1.5, 1.0]])
        point = np.array([0.0, 0.0])
        normal = np.array([1.0, 0.0])
        clipped = _clip_polygon_to_half_plane(tri, point, normal)
        assert len(clipped) == 0


class TestBondPolygonsBatch:
    """Verify that the vectorised batch matches the scalar _bond_polygon."""

    def _ch4_scene_data(self, view=None):
        """Build CH4 scene data needed by both scalar and batch paths."""
        from hofmann.construction.bonds import compute_bonds

        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [1.155, 1.155, 1.155],
            [-1.155, -1.155, 1.155],
            [1.155, -1.155, -1.155],
            [-1.155, 1.155, -1.155],
        ])
        bond_specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4,
                          radius=0.109, colour=1.0)]
        bonds = compute_bonds(species, coords, bond_specs)

        if view is None:
            view = ViewState()
        atom_scale = 0.5

        radii_3d = np.array([1.0, 0.7, 0.7, 0.7, 0.7])
        xy, depth, screen_radii = view.project(coords, radii_3d * atom_scale)
        rotated = (coords - view.centre) @ view.rotation.T

        bond_ia = np.array([b.index_a for b in bonds])
        bond_ib = np.array([b.index_b for b in bonds])
        bond_radii = np.array([b.spec.radius for b in bonds])

        return {
            "bonds": bonds,
            "rotated": rotated,
            "xy": xy,
            "radii_3d": radii_3d,
            "screen_radii": screen_radii,
            "bond_ia": bond_ia,
            "bond_ib": bond_ib,
            "bond_radii": bond_radii,
            "atom_scale": atom_scale,
            "view": view,
        }

    def test_matches_scalar_identity_view(self):
        """Batch output matches scalar _bond_polygon for identity rotation."""
        d = self._ch4_scene_data()
        view = d["view"]
        atom_scale = d["atom_scale"]

        full_verts, start_2d, end_2d, _bb_a, _bb_b, _bx, _by, valid = (
            _bond_polygons_batch(
                d["rotated"], d["xy"],
                d["radii_3d"] * atom_scale, d["screen_radii"],
                d["bond_ia"], d["bond_ib"], d["bond_radii"],
                view,
            )
        )

        for i, bond in enumerate(d["bonds"]):
            ia, ib = bond.index_a, bond.index_b
            scalar = _bond_polygon(
                d["rotated"][ia], d["rotated"][ib],
                d["radii_3d"][ia] * atom_scale,
                d["radii_3d"][ib] * atom_scale,
                bond.spec.radius,
                d["screen_radii"][ia], d["screen_radii"][ib],
                view,
            )
            assert scalar is not None
            assert valid[i]
            s_verts, s_start, s_end = scalar
            np.testing.assert_allclose(full_verts[i], s_verts, atol=1e-12)
            np.testing.assert_allclose(start_2d[i], s_start, atol=1e-12)
            np.testing.assert_allclose(end_2d[i], s_end, atol=1e-12)

    def test_matches_scalar_rotated_view(self):
        """Batch output matches scalar for a non-trivial rotation."""
        rot = _rotation_y(0.7) @ _rotation_x(0.3)
        view = ViewState(rotation=rot, zoom=1.2)
        d = self._ch4_scene_data(view=view)
        atom_scale = d["atom_scale"]

        full_verts, start_2d, end_2d, *_, valid = _bond_polygons_batch(
            d["rotated"], d["xy"],
            d["radii_3d"] * atom_scale, d["screen_radii"],
            d["bond_ia"], d["bond_ib"], d["bond_radii"],
            view,
        )

        for i, bond in enumerate(d["bonds"]):
            ia, ib = bond.index_a, bond.index_b
            scalar = _bond_polygon(
                d["rotated"][ia], d["rotated"][ib],
                d["radii_3d"][ia] * atom_scale,
                d["radii_3d"][ib] * atom_scale,
                bond.spec.radius,
                d["screen_radii"][ia], d["screen_radii"][ib],
                view,
            )
            assert scalar is not None
            assert valid[i]
            s_verts, s_start, s_end = scalar
            np.testing.assert_allclose(full_verts[i], s_verts, atol=1e-12)

    def test_empty_bonds(self):
        """No bonds produces empty arrays."""
        view = ViewState()
        rotated = np.zeros((2, 3))
        xy = np.zeros((2, 2))
        radii = np.array([1.0, 1.0])
        screen_radii = np.array([0.5, 0.5])

        full_verts, start_2d, end_2d, bb_a, bb_b, bx, by, valid = (
            _bond_polygons_batch(
                rotated, xy, radii, screen_radii,
                np.array([], dtype=int), np.array([], dtype=int),
                np.array([]), view,
            )
        )
        assert full_verts.shape[0] == 0
        assert valid.shape[0] == 0

    def test_occluded_bond_marked_invalid(self):
        """Overlapping atoms produce an invalid bond."""
        view = ViewState()
        # Two atoms very close together with large radii.
        rotated = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        xy = np.array([[0.0, 0.0], [0.5, 0.0]])
        radii = np.array([1.0, 1.0])
        screen_radii = np.array([1.0, 1.0])

        _, _, _, _, _, _, _, valid = _bond_polygons_batch(
            rotated, xy, radii, screen_radii,
            np.array([0]), np.array([1]), np.array([0.1]),
            view,
        )
        assert not valid[0]


class TestHalfBondVertsBatch:
    """Verify direct half-bond construction."""

    def test_each_half_has_seven_vertices(self):
        """Each half-bond polygon should have exactly 7 vertices."""
        from hofmann.rendering.bond_geometry import _N_ARC

        view = ViewState()
        # Simple horizontal bond.
        rotated = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        xy = np.array([[0.0, 0.0], [4.0, 0.0]])
        radii = np.array([1.0, 1.0])
        screen_radii = np.array([1.0, 1.0])

        full_verts, start_2d, end_2d, bb_a, bb_b, bx, by, valid = (
            _bond_polygons_batch(
                rotated, xy, radii, screen_radii,
                np.array([0]), np.array([1]), np.array([0.1]),
                view,
            )
        )
        half_a, half_b = _half_bond_verts_batch(
            full_verts, start_2d, end_2d, bb_a, bb_b, bx, by,
        )
        assert half_a.shape == (1, _N_ARC + 2, 2)
        assert half_b.shape == (1, _N_ARC + 2, 2)

    def test_halves_cover_full_area(self):
        """The two halves together should approximately cover the full polygon area."""

        def shoelace(v):
            x, y = v[:, 0], v[:, 1]
            return 0.5 * abs(
                np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            )

        view = ViewState()
        rotated = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        xy = np.array([[0.0, 0.0], [4.0, 0.0]])
        radii = np.array([1.0, 1.0])
        screen_radii = np.array([1.0, 1.0])

        full_verts, start_2d, end_2d, bb_a, bb_b, bx, by, valid = (
            _bond_polygons_batch(
                rotated, xy, radii, screen_radii,
                np.array([0]), np.array([1]), np.array([0.1]),
                view,
            )
        )
        half_a, half_b = _half_bond_verts_batch(
            full_verts, start_2d, end_2d, bb_a, bb_b, bx, by,
        )

        area_full = shoelace(full_verts[0])
        area_a = shoelace(half_a[0])
        area_b = shoelace(half_b[0])
        # The two halves should sum to approximately the full area.
        # A small tolerance accounts for the straight midpoint cut vs arc.
        np.testing.assert_allclose(area_a + area_b, area_full, rtol=0.05)

    def test_ch4_halves_have_correct_shape(self):
        """All CH4 half-bond polygons should have the expected vertex count."""
        from hofmann.construction.bonds import compute_bonds
        from hofmann.rendering.bond_geometry import _N_ARC

        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [1.155, 1.155, 1.155],
            [-1.155, -1.155, 1.155],
            [1.155, -1.155, -1.155],
            [-1.155, 1.155, -1.155],
        ])
        bond_specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4,
                          radius=0.109, colour=1.0)]
        bonds = compute_bonds(species, coords, bond_specs)
        view = ViewState()
        atom_scale = 0.5

        radii_3d = np.array([1.0, 0.7, 0.7, 0.7, 0.7])
        xy, _, screen_radii = view.project(coords, radii_3d * atom_scale)
        rotated = (coords - view.centre) @ view.rotation.T

        bond_ia = np.array([b.index_a for b in bonds])
        bond_ib = np.array([b.index_b for b in bonds])
        bond_r = np.array([b.spec.radius for b in bonds])

        full_verts, start_2d, end_2d, bb_a, bb_b, bx, by, valid = (
            _bond_polygons_batch(
                rotated, xy, radii_3d * atom_scale, screen_radii,
                bond_ia, bond_ib, bond_r, view,
            )
        )
        half_a, half_b = _half_bond_verts_batch(
            full_verts, start_2d, end_2d, bb_a, bb_b, bx, by,
        )
        n_bonds = len(bonds)
        assert half_a.shape == (n_bonds, _N_ARC + 2, 2)
        assert half_b.shape == (n_bonds, _N_ARC + 2, 2)
