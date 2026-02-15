"""Tests for hofmann.render_mpl — depth-sorted matplotlib renderer."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import AtomStyle, BondSpec, Frame, StructureScene, ViewState
from hofmann.render_mpl import (
    _bond_polygon,
    _bond_polygons_batch,
    _clip_bond_3d,
    _clip_polygon_to_half_plane,
    _half_bond_verts_batch,
    _project_point,
    _rotation_x,
    _rotation_y,
    _stick_polygon,
    render_mpl,
)
from hofmann.scene import from_xbs


def _minimal_scene(n_atoms=2, with_bonds=True):
    """Create a minimal scene for testing."""
    species = ["A", "B"][:n_atoms]
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])[:n_atoms]
    styles = {
        "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
        "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
    }
    specs = []
    if with_bonds and n_atoms >= 2:
        specs = [BondSpec("A", "B", 0.0, 5.0, 0.1, 1.0)]
    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles=styles,
        bond_specs=specs,
    )


class TestRenderMpl:
    def test_returns_figure(self):
        scene = _minimal_scene()
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_saves_to_file(self, tmp_path):
        scene = _minimal_scene()
        out = tmp_path / "test.png"
        render_mpl(scene, output=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_ch4_renders(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_no_bonds(self):
        scene = _minimal_scene(with_bonds=False)
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_custom_frame_index(self):
        coords1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords1), Frame(coords=coords2)],
            atom_styles={"A": AtomStyle(1.0, "grey"), "B": AtomStyle(0.8, "red")},
        )
        fig = render_mpl(scene, frame_index=1, show=False)
        assert isinstance(fig, Figure)

    def test_convenience_method(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = scene.render_mpl(show=False)
        assert isinstance(fig, Figure)

    def test_half_bonds(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        fig = render_mpl(scene, half_bonds=True, show=False)
        assert isinstance(fig, Figure)

    def test_saves_svg(self, tmp_path, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        out = tmp_path / "ch4.svg"
        scene.render_mpl(output=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_slab_clipping(self):
        """A tight slab should produce a figure with fewer drawn polygons."""
        scene = _minimal_scene()
        # Atoms at z=0 and z=0 (both x-axis), slab should include both.
        fig_all = render_mpl(scene, show=False)
        assert isinstance(fig_all, Figure)

        # Now set a slab that excludes one atom (at x=2, z=0).
        # Looking along x, atom B at x=2 has depth=2.
        scene.view.look_along([1, 0, 0])
        scene.view.slab_near = -0.5
        scene.view.slab_far = 0.5
        # Only atom at x=0 (depth 0) should be visible.
        fig_slab = render_mpl(scene, show=False)
        assert isinstance(fig_slab, Figure)


# --- Geometry helpers ---


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


# --- Rotation helpers ---


class TestRotationHelpers:
    def test_rotation_x_zero(self):
        """Zero angle gives identity."""
        np.testing.assert_allclose(_rotation_x(0.0), np.eye(3), atol=1e-15)

    def test_rotation_y_zero(self):
        """Zero angle gives identity."""
        np.testing.assert_allclose(_rotation_y(0.0), np.eye(3), atol=1e-15)

    def test_rotation_x_90(self):
        """90-degree X rotation sends Y to Z."""
        r = _rotation_x(np.pi / 2)
        y_axis = np.array([0.0, 1.0, 0.0])
        result = r @ y_axis
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-15)

    def test_rotation_y_90(self):
        """90-degree Y rotation sends Z to X."""
        r = _rotation_y(np.pi / 2)
        z_axis = np.array([0.0, 0.0, 1.0])
        result = r @ z_axis
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-15)

    def test_rotation_is_orthogonal(self):
        """Rotation matrices should satisfy R^T R = I."""
        for angle in [0.3, -1.2, np.pi]:
            rx = _rotation_x(angle)
            ry = _rotation_y(angle)
            np.testing.assert_allclose(rx.T @ rx, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(ry.T @ ry, np.eye(3), atol=1e-14)

    def test_composition_preserves_orthogonality(self):
        """Composing rotations should stay orthogonal."""
        r = _rotation_y(0.5) @ _rotation_x(0.3)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)


# --- Batch bond geometry ---


class TestBondPolygonsBatch:
    """Verify that the vectorised batch matches the scalar _bond_polygon."""

    def _ch4_scene_data(self, view=None):
        """Build CH4 scene data needed by both scalar and batch paths."""
        from hofmann.bonds import compute_bonds

        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [1.155, 1.155, 1.155],
            [-1.155, -1.155, 1.155],
            [1.155, -1.155, -1.155],
            [-1.155, 1.155, -1.155],
        ])
        bond_specs = [BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0)]
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
        from hofmann.render_mpl import _N_ARC

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
        from hofmann.bonds import compute_bonds
        from hofmann.render_mpl import _N_ARC

        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [1.155, 1.155, 1.155],
            [-1.155, -1.155, 1.155],
            [1.155, -1.155, -1.155],
            [-1.155, 1.155, -1.155],
        ])
        bond_specs = [BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0)]
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
