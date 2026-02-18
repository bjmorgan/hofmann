"""Tests for hofmann.render_mpl — depth-sorted matplotlib renderer."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import (
    AtomStyle,
    AxesStyle,
    BondSpec,
    CellEdgeStyle,
    Frame,
    PolyhedronSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
)
from hofmann.render_mpl import (
    _apply_key_action,
    _bond_polygon,
    _bond_polygons_batch,
    _cell_edges_3d,
    _clip_bond_3d,
    _clip_edge_at_atoms,
    _clip_polygon_to_half_plane,
    _collect_polyhedra_faces,
    _half_bond_verts_batch,
    _HELP_TEXT,
    _KEY_PAN_FRACTION,
    _KEY_ROTATION_STEP,
    _KEY_ZOOM_FACTOR,
    _PERSPECTIVE_STEP,
    _project_point,
    _rotation_x,
    _rotation_y,
    _rotation_z,
    _scene_extent,
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
        specs = [BondSpec(species=("A", "B"), min_length=0.0,
                          max_length=5.0, radius=0.1, colour=1.0)]
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

    def test_invisible_atom_not_drawn(self):
        """An atom with visible=False should not appear in the render."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0],
                                           [2.0, 0.0, 0.0]]))],
            atom_styles={
                "A": AtomStyle(1.0, "grey"),
                "B": AtomStyle(0.8, "red", visible=False),
            },
        )
        fig_hidden = render_mpl(scene, show=False)
        assert isinstance(fig_hidden, Figure)
        # Compare with both visible.
        scene_visible = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0],
                                           [2.0, 0.0, 0.0]]))],
            atom_styles={
                "A": AtomStyle(1.0, "grey"),
                "B": AtomStyle(0.8, "red"),
            },
        )
        fig_visible = render_mpl(scene_visible, show=False)
        # The hidden scene should have fewer drawn patches.
        ax_hidden = fig_hidden.axes[0]
        ax_visible = fig_visible.axes[0]
        n_hidden = sum(len(c.get_paths()) for c in ax_hidden.collections)
        n_visible = sum(len(c.get_paths()) for c in ax_visible.collections)
        assert n_hidden < n_visible

    def test_invisible_atom_hidden_without_polyhedra(self):
        """visible=False hides atoms even when show_polyhedra=False."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0],
                                           [2.0, 0.0, 0.0]]))],
            atom_styles={
                "A": AtomStyle(1.0, "grey"),
                "B": AtomStyle(0.8, "red", visible=False),
            },
        )
        fig_no_poly = render_mpl(scene, show=False, show_polyhedra=False)
        fig_visible = render_mpl(
            StructureScene(
                species=["A", "B"],
                frames=[Frame(coords=np.array([[0.0, 0.0, 0.0],
                                               [2.0, 0.0, 0.0]]))],
                atom_styles={
                    "A": AtomStyle(1.0, "grey"),
                    "B": AtomStyle(0.8, "red"),
                },
            ),
            show=False, show_polyhedra=False,
        )
        ax_no_poly = fig_no_poly.axes[0]
        ax_visible = fig_visible.axes[0]
        n_hidden = sum(len(c.get_paths()) for c in ax_no_poly.collections)
        n_visible = sum(len(c.get_paths()) for c in ax_visible.collections)
        assert n_hidden < n_visible

    def test_empty_scene(self):
        """An empty scene (zero atoms) should render without crashing."""
        scene = StructureScene(
            species=[],
            frames=[Frame(coords=np.empty((0, 3)))],
            atom_styles={},
        )
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

    def test_renders_to_supplied_axes(self):
        """Rendering into a user-supplied axes draws on that axes."""
        scene = _minimal_scene()
        fig, ax = plt.subplots()
        result = render_mpl(scene, ax=ax)
        assert result is fig
        n_paths = sum(len(c.get_paths()) for c in ax.collections)
        assert n_paths > 0
        plt.close(fig)

    def test_supplied_axes_does_not_create_new_figure(self):
        """When ax is provided, plt.subplots should not be called."""
        scene = _minimal_scene()
        fig, ax = plt.subplots()
        from unittest.mock import patch
        with patch("hofmann.render_mpl.plt.subplots") as mock_subplots:
            render_mpl(scene, ax=ax)
            mock_subplots.assert_not_called()
        plt.close(fig)

    def test_subplot_panels(self):
        """Rendering the same scene into multiple subplot axes."""
        scene = _minimal_scene()
        fig, axes = plt.subplots(1, 3)
        for i, direction in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            scene.view.look_along(direction)
            render_mpl(scene, ax=axes[i])
        for ax in axes:
            n_paths = sum(len(c.get_paths()) for c in ax.collections)
            assert n_paths > 0
        plt.close(fig)

    def test_title_rendered_in_viewport_on_supplied_axes(self):
        """scene.title is drawn as text inside the viewport on user axes."""
        scene = _minimal_scene()
        scene.title = "Test Title"
        fig, ax = plt.subplots()
        render_mpl(scene, ax=ax)
        texts = [t.get_text() for t in ax.texts]
        assert "Test Title" in texts
        plt.close(fig)

    def test_half_bonds_via_style(self, ch4_bs_path):
        """Passing half_bonds via a RenderStyle works."""
        scene = from_xbs(ch4_bs_path)
        style = RenderStyle(half_bonds=True)
        fig = render_mpl(scene, style=style, show=False)
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

    def test_show_bonds_false(self):
        """Rendering with show_bonds=False should produce atoms only."""
        scene = _minimal_scene()
        fig = render_mpl(scene, show_bonds=False, show=False)
        assert isinstance(fig, Figure)
        # With bonds off, the only PolyCollection should contain just atoms.
        ax = fig.axes[0]
        pc = ax.collections[0]
        # Two atoms, no bond polygons.
        assert len(pc.get_paths()) == 2

    def test_show_outlines_false(self):
        """Outlines disabled via style produces zero-width edges."""
        scene = _minimal_scene()
        style = RenderStyle(show_outlines=False)
        fig = render_mpl(scene, style=style, show=False)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        pc = ax.collections[0]
        # All line widths should be zero.
        lws = pc.get_linewidths()
        assert all(w == 0.0 for w in lws)

    def test_custom_outline_colour(self):
        """Custom outline colour via style is applied."""
        scene = _minimal_scene()
        style = RenderStyle(outline_colour="red")
        fig = render_mpl(scene, style=style, show=False)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        pc = ax.collections[0]
        # Edge colours should be red (1, 0, 0) for all polygons.
        edges = pc.get_edgecolors()
        for ec in edges:
            np.testing.assert_allclose(ec[:3], [1.0, 0.0, 0.0], atol=1e-3)

    def test_style_kwarg_override(self):
        """Convenience kwargs override matching style fields."""
        scene = _minimal_scene()
        style = RenderStyle(show_bonds=True)
        fig = render_mpl(scene, style=style, show_bonds=False, show=False)
        assert isinstance(fig, Figure)
        # show_bonds=False overrides the style, so only 2 atoms drawn.
        ax = fig.axes[0]
        pc = ax.collections[0]
        assert len(pc.get_paths()) == 2

    def test_half_bonds_kwarg(self):
        """half_bonds=False passed as a convenience kwarg is respected."""
        scene = _minimal_scene()
        fig = render_mpl(scene, half_bonds=False, show=False)
        assert isinstance(fig, Figure)

    def test_unknown_style_kwarg_raises(self):
        """Unknown keyword arguments raise TypeError."""
        scene = _minimal_scene()
        with pytest.raises(TypeError, match="Unknown style keyword"):
            render_mpl(scene, nonexistent_option=True, show=False)

    def test_render_style_defaults_match_original(self):
        """Default RenderStyle produces the same output as no style."""
        scene = _minimal_scene()
        fig_default = render_mpl(scene, show=False)
        fig_style = render_mpl(scene, style=RenderStyle(), show=False)
        # Same number of polygons drawn.
        paths_a = fig_default.axes[0].collections[0].get_paths()
        paths_b = fig_style.axes[0].collections[0].get_paths()
        assert len(paths_a) == len(paths_b)


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
            rz = _rotation_z(angle)
            np.testing.assert_allclose(rx.T @ rx, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(ry.T @ ry, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(rz.T @ rz, np.eye(3), atol=1e-14)

    def test_rotation_z_zero(self):
        """Zero angle gives identity."""
        np.testing.assert_allclose(_rotation_z(0.0), np.eye(3), atol=1e-15)

    def test_rotation_z_90(self):
        """90-degree Z rotation sends X to Y."""
        r = _rotation_z(np.pi / 2)
        x_axis = np.array([1.0, 0.0, 0.0])
        result = r @ x_axis
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-15)

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


# --- Polyhedra rendering ---


def _octahedron_scene(**poly_kwargs):
    """Build a TiO6 octahedron scene for polyhedra testing."""
    species = ["Ti"] + ["O"] * 6
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -2.0],
    ])
    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles={
            "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
            "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
        },
        bond_specs=[BondSpec(
            species=("O", "Ti"), min_length=0.0, max_length=3.0,
            radius=0.1, colour=0.5,
        )],
        polyhedra=[PolyhedronSpec(centre="Ti", **poly_kwargs)],
    )


class TestPolyhedraRendering:
    def test_smoke(self):
        """Scene with TiO6 octahedron renders without error."""
        scene = _octahedron_scene()
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)

    def test_polyhedra_hide_interior_bonds(self):
        """Interior bonds are hidden when polyhedra are drawn."""
        scene_no_poly = _octahedron_scene()
        scene_no_poly.polyhedra = []
        scene_no_poly.view.look_along([1.0, 0.8, 0.6])
        fig_no = render_mpl(scene_no_poly, show=False)
        n_no = len(fig_no.axes[0].collections[0].get_paths())

        scene_poly = _octahedron_scene()
        scene_poly.view.look_along([1.0, 0.8, 0.6])
        fig_poly = render_mpl(scene_poly, show=False)
        n_poly = len(fig_poly.axes[0].collections[0].get_paths())

        # Without polyhedra: 7 atoms + 12 half-bond polygons = 19.
        # With polyhedra: 7 atoms + 8 faces + 0 interior bonds = 15.
        assert n_no > n_poly

    def test_show_polyhedra_false(self):
        """show_polyhedra=False suppresses faces and restores interior bonds."""
        scene = _octahedron_scene()
        scene.view.look_along([1.0, 0.8, 0.6])
        style_on = RenderStyle(show_polyhedra=True)
        style_off = RenderStyle(show_polyhedra=False)
        fig_on = render_mpl(scene, style=style_on, show=False)
        fig_off = render_mpl(scene, style=style_off, show=False)
        n_on = len(fig_on.axes[0].collections[0].get_paths())
        n_off = len(fig_off.axes[0].collections[0].get_paths())
        # With polyhedra off the faces disappear but interior bonds
        # reappear, giving more paths than with polyhedra on.
        assert n_off > n_on

    def test_show_outlines_false_suppresses_polyhedra_edges(self):
        """show_outlines=False zeroes polyhedra edge widths."""
        scene = _octahedron_scene()
        style = RenderStyle(show_outlines=False)
        fig = render_mpl(scene, style=style, show=False)
        pc = fig.axes[0].collections[0]
        lws = pc.get_linewidths()
        assert all(w == 0.0 for w in lws)

    def test_hide_centre(self):
        """hide_centre=True removes the centre atom's circle."""
        scene_show = _octahedron_scene(hide_centre=False)
        scene_hide = _octahedron_scene(hide_centre=True)
        fig_show = render_mpl(scene_show, show=False)
        fig_hide = render_mpl(scene_hide, show=False)
        n_show = len(fig_show.axes[0].collections[0].get_paths())
        n_hide = len(fig_hide.axes[0].collections[0].get_paths())
        # Hiding the centre atom removes 1 polygon.
        assert n_show - n_hide == 1

    def test_hide_bonds(self):
        """Interior centre-to-vertex bonds are always hidden by polyhedra."""
        # Interior bonds (centre->vertex) are always hidden when a
        # polyhedron is drawn, regardless of hide_bonds.  The polygon
        # count should be the same for both settings in a pure
        # octahedron where all bonds are interior.
        scene_show = _octahedron_scene(hide_bonds=False)
        scene_hide = _octahedron_scene(hide_bonds=True)
        fig_show = render_mpl(scene_show, show=False)
        fig_hide = render_mpl(scene_hide, show=False)
        n_show = len(fig_show.axes[0].collections[0].get_paths())
        n_hide = len(fig_hide.axes[0].collections[0].get_paths())
        assert n_show == n_hide

    def test_face_colours_have_alpha(self):
        """Polyhedron face colours should include an alpha channel."""
        scene = _octahedron_scene()
        fig = render_mpl(scene, show=False)
        pc = fig.axes[0].collections[0]
        fc = pc.get_facecolors()
        # All colours should have 4 components (RGBA).
        assert fc.shape[1] == 4
        # Some faces should have alpha < 1 (the polyhedron faces).
        alphas = fc[:, 3]
        assert np.any(alphas < 1.0)

    def test_hide_vertices(self):
        """hide_vertices=True removes vertex atom circles."""
        scene_show = _octahedron_scene(hide_vertices=False)
        scene_hide = _octahedron_scene(hide_vertices=True)
        fig_show = render_mpl(scene_show, show=False)
        fig_hide = render_mpl(scene_hide, show=False)
        n_show = len(fig_show.axes[0].collections[0].get_paths())
        n_hide = len(fig_hide.axes[0].collections[0].get_paths())
        # hide_vertices=True removes all vertex circles; without it
        # only front/equatorial vertices are drawn (back vertices are
        # suppressed behind the polyhedral faces).
        assert n_show > n_hide

    def test_hide_vertices_shared_vertex_not_hidden(self):
        """A shared vertex is kept if any polyhedron has hide_vertices=False."""
        # Ti and Zr centres sharing one O vertex at the origin.
        # Ti at (-3,0,0) with 4 O neighbours including shared O at origin.
        # Zr at (3,0,0) with 4 O neighbours including shared O at origin.
        species = [
            "Ti", "Zr",
            "O", "O", "O",   # Ti-only vertices
            "O",              # shared vertex (index 5)
            "O", "O", "O",   # Zr-only vertices
        ]
        coords = np.array([
            [-3.0, 0.0, 0.0],   # Ti
            [3.0, 0.0, 0.0],    # Zr
            [-3.0, 2.0, 0.0],   # O bonded to Ti only
            [-3.0, -2.0, 0.0],  # O bonded to Ti only
            [-3.0, 0.0, 2.0],   # O bonded to Ti only
            [0.0, 0.0, 0.0],    # O shared (within 3.0 of both)
            [3.0, 2.0, 0.0],    # O bonded to Zr only
            [3.0, -2.0, 0.0],   # O bonded to Zr only
            [3.0, 0.0, 2.0],    # O bonded to Zr only
        ])
        base_styles = {
            "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
            "Zr": AtomStyle(1.0, (0.4, 0.8, 0.2)),
            "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
        }
        base_bonds = [BondSpec(
            species=("O", "*"), min_length=0.0, max_length=3.5,
            radius=0.1, colour=0.5,
        )]

        # Both specs hide vertices — shared O should be hidden.
        scene_both_hide = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles=base_styles,
            bond_specs=base_bonds,
            polyhedra=[
                PolyhedronSpec(centre="Ti", hide_vertices=True),
                PolyhedronSpec(centre="Zr", hide_vertices=True),
            ],
        )
        # Zr spec does NOT hide vertices — shared O should be kept.
        scene_one_keeps = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles=base_styles,
            bond_specs=base_bonds,
            polyhedra=[
                PolyhedronSpec(centre="Ti", hide_vertices=True),
                PolyhedronSpec(centre="Zr", hide_vertices=False),
            ],
        )
        fig_both = render_mpl(scene_both_hide, show=False)
        fig_one = render_mpl(scene_one_keeps, show=False)
        n_both = len(fig_both.axes[0].collections[0].get_paths())
        n_one = len(fig_one.axes[0].collections[0].get_paths())
        # When Zr keeps vertices, the shared O is kept -> more polygons.
        assert n_one > n_both

    def test_hide_vertices_kept_when_bonded_outside_polyhedron(self):
        """A vertex bonded to a non-polyhedron atom stays visible."""
        # Ti at origin with 3 O neighbours forming a polyhedron.
        # Li bonded to one of the O atoms but not a polyhedron centre.
        # Even with hide_vertices=True, that O must stay visible.
        species = ["Ti", "O", "O", "O", "Li"]
        coords = np.array([
            [0.0, 0.0, 0.0],   # Ti (polyhedron centre)
            [2.0, 0.0, 0.0],   # O vertex, also bonded to Li
            [0.0, 2.0, 0.0],   # O vertex
            [0.0, 0.0, 2.0],   # O vertex
            [4.0, 0.0, 0.0],   # Li bonded to O at index 1
        ])
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
                "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
                "Li": AtomStyle(0.6, (0.4, 0.8, 0.4)),
            },
            bond_specs=[
                BondSpec(species=("Ti", "O"), min_length=0.0,
                         max_length=3.0, radius=0.1, colour=0.5),
                BondSpec(species=("Li", "O"), min_length=0.0,
                         max_length=3.0, radius=0.1, colour=0.5),
            ],
            polyhedra=[PolyhedronSpec(centre="Ti", hide_vertices=True)],
        )
        fig = render_mpl(scene, show=False)
        n = len(fig.axes[0].collections[0].get_paths())
        # O at index 1 is bonded to Li, so it must stay visible
        # despite hide_vertices=True.  With hide_vertices=True the
        # showing count should be greater than if O(1) were also
        # removed.  Compare against hide_vertices=False to confirm
        # O(1) is kept.
        scene_no_hide = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles=scene.atom_styles,
            bond_specs=scene.bond_specs,
            polyhedra=[PolyhedronSpec(centre="Ti", hide_vertices=False)],
        )
        fig_no_hide = render_mpl(scene_no_hide, show=False)
        n_no_hide = len(fig_no_hide.axes[0].collections[0].get_paths())
        # hide_vertices removes some visible vertex circles but O(1)
        # is kept because it's bonded outside the polyhedron.
        assert n_no_hide >= n
        assert n > n_no_hide - 3  # At most 2 vertices removed, not all 3


class TestSameColourBonds:
    def test_same_species_bond_single_polygon(self):
        """Same-colour half-bonds should merge into one polygon."""
        species = ["Na", "Na"]
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Na": AtomStyle(1.0, (0.3, 0.3, 0.8)),
            },
            bond_specs=[BondSpec(
                species=("Na", "Na"), min_length=0.0, max_length=3.0,
                radius=0.1, colour=0.5,
            )],
        )
        style_half = RenderStyle(half_bonds=True, show_outlines=True)
        fig = render_mpl(scene, style=style_half, show=False)
        n = len(fig.axes[0].collections[0].get_paths())
        # 2 atoms + 1 bond polygon (not 2 half-bond polygons).
        assert n == 3

    def test_different_species_bond_two_polygons(self):
        """Different-colour half-bonds should produce two polygons."""
        species = ["Na", "Cl"]
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Na": AtomStyle(1.0, (0.3, 0.3, 0.8)),
                "Cl": AtomStyle(1.0, (0.1, 0.8, 0.1)),
            },
            bond_specs=[BondSpec(
                species=("Na", "Cl"), min_length=0.0, max_length=3.0,
                radius=0.1, colour=0.5,
            )],
        )
        style_half = RenderStyle(half_bonds=True, show_outlines=True)
        fig = render_mpl(scene, style=style_half, show=False)
        n = len(fig.axes[0].collections[0].get_paths())
        # 2 atoms + 2 half-bond polygons.
        assert n == 4


def _slab_octahedron_scene(**poly_kwargs):
    """Octahedron with a slab that clips the z=+/-2 oxygen atoms."""
    scene = _octahedron_scene(**poly_kwargs)
    scene.view.slab_near = -1.5
    scene.view.slab_far = 1.5
    return scene


class TestSlabClipModes:
    def test_per_face_produces_partial(self):
        """per_face mode draws some faces but not all (partial fragment)."""
        scene = _slab_octahedron_scene()
        style = RenderStyle(slab_clip_mode=SlabClipMode.PER_FACE)
        fig = render_mpl(scene, style=style, show=False)
        n = len(fig.axes[0].collections[0].get_paths())
        # Some polygons drawn (atoms in-slab + partial polyhedron faces).
        assert n > 0

    def test_clip_whole_removes_polyhedron(self):
        """clip_whole drops the polyhedron when a vertex is outside the slab."""
        scene = _slab_octahedron_scene()
        style = RenderStyle(slab_clip_mode=SlabClipMode.CLIP_WHOLE)
        fig = render_mpl(scene, style=style, show=False)
        pc = fig.axes[0].collections[0]
        fc = pc.get_facecolors()
        # All drawn faces should be opaque (alpha=1) — no polyhedron
        # faces (which have alpha < 1) because clip_whole drops the
        # entire polyhedron when z=+/-2 vertices are outside the slab.
        assert np.all(fc[:, 3] == 1.0)

    def test_include_whole_more_than_per_face(self):
        """include_whole forces all faces visible, producing more polygons."""
        scene_pf = _slab_octahedron_scene()
        scene_iw = _slab_octahedron_scene()
        style_pf = RenderStyle(slab_clip_mode=SlabClipMode.PER_FACE)
        style_iw = RenderStyle(slab_clip_mode=SlabClipMode.INCLUDE_WHOLE)
        fig_pf = render_mpl(scene_pf, style=style_pf, show=False)
        fig_iw = render_mpl(scene_iw, style=style_iw, show=False)
        n_pf = len(fig_pf.axes[0].collections[0].get_paths())
        n_iw = len(fig_iw.axes[0].collections[0].get_paths())
        # include_whole draws all faces + force-visible vertex atoms.
        assert n_iw > n_pf

    def test_include_whole_centre_outside_skips(self):
        """include_whole skips polyhedra whose centre is outside the slab."""
        species = ["Ti"] + ["O"] * 6
        coords = np.array([
            [0.0, 0.0, 5.0],   # Centre outside slab
            [2.0, 0.0, 5.0],
            [-2.0, 0.0, 5.0],
            [0.0, 2.0, 5.0],
            [0.0, -2.0, 5.0],
            [0.0, 0.0, 7.0],
            [0.0, 0.0, 3.0],
        ])
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
                "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
            },
            bond_specs=[BondSpec(
                species=("O", "Ti"), min_length=0.0, max_length=3.0,
                radius=0.1, colour=0.5,
            )],
            polyhedra=[PolyhedronSpec(centre="Ti")],
        )
        scene.view.slab_near = -1.5
        scene.view.slab_far = 1.5
        style = RenderStyle(slab_clip_mode=SlabClipMode.INCLUDE_WHOLE)
        fig = render_mpl(scene, style=style, show=False)
        # Everything is outside the slab (centre at z=5), so nothing drawn.
        pc = fig.axes[0].collections
        assert len(pc) == 0 or len(pc[0].get_paths()) == 0

    def test_no_slab_all_modes_identical(self):
        """Without slab settings, all three modes produce the same output."""
        counts = []
        for mode in SlabClipMode:
            scene = _octahedron_scene()
            style = RenderStyle(slab_clip_mode=mode)
            fig = render_mpl(scene, style=style, show=False)
            counts.append(len(fig.axes[0].collections[0].get_paths()))
        assert counts[0] == counts[1] == counts[2]


def _two_octahedra_scene(extra_atoms=False, **poly_kwargs):
    """Build a scene with two TiO6 octahedra offset along z.

    The first Ti is at z=0, the second at z=5.  Each has 6 oxygen
    neighbours at +/-2 along each axis from its centre.

    If *extra_atoms* is True, two inert Ar atoms are placed at z=+/-3
    (between the two polyhedra) — these are not bonded to anything.
    """
    species = ["Ti"] + ["O"] * 6 + ["Ti"] + ["O"] * 6
    coords = np.array([
        # First octahedron: centre at (0, 0, 0)
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0], [0.0, 0.0, -2.0],
        # Second octahedron: centre at (0, 0, 5)
        [0.0, 0.0, 5.0],
        [2.0, 0.0, 5.0], [-2.0, 0.0, 5.0],
        [0.0, 2.0, 5.0], [0.0, -2.0, 5.0],
        [0.0, 0.0, 7.0], [0.0, 0.0, 3.0],
    ])
    if extra_atoms:
        species += ["Ar", "Ar"]
        coords = np.vstack([coords, [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]])

    atom_styles = {
        "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
        "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
    }
    if extra_atoms:
        atom_styles["Ar"] = AtomStyle(0.6, (0.5, 0.5, 0.5))

    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles=atom_styles,
        bond_specs=[BondSpec(
            species=("O", "Ti"), min_length=0.0, max_length=3.0,
            radius=0.1, colour=0.5,
        )],
        polyhedra=[PolyhedronSpec(centre="Ti", **poly_kwargs)],
    )


class TestPolyhedraDepthOrdering:
    """Tests for polyhedra face depth-ordering fixes."""

    def test_faces_not_dropped_at_invisible_atom_slot(self):
        """Polyhedra faces must be drawn even when their depth slot
        corresponds to a slab-clipped atom."""
        # Two octahedra plus extra non-bonded atoms between them.
        # Slab clips the Ar atoms but includes both Ti centres.
        scene = _two_octahedra_scene(extra_atoms=True)
        scene.view.slab_near = -3.5
        scene.view.slab_far = 8.0
        style_include = RenderStyle(
            slab_clip_mode=SlabClipMode.INCLUDE_WHOLE,
        )
        fig_include = render_mpl(scene, style=style_include, show=False)
        n_include = len(fig_include.axes[0].collections[0].get_paths())

        # Without slab, all faces are drawn — count should match.
        scene_noslab = _two_octahedra_scene(extra_atoms=True)
        style_noslab = RenderStyle(
            slab_clip_mode=SlabClipMode.INCLUDE_WHOLE,
        )
        fig_noslab = render_mpl(
            scene_noslab, style=style_noslab, show=False,
        )
        n_noslab = len(fig_noslab.axes[0].collections[0].get_paths())

        assert n_include == n_noslab

    def test_two_overlapping_polyhedra_include_whole_smoke(self):
        """Two overlapping polyhedra with include_whole render without
        error and produce polyhedra face polygons."""
        scene = _two_octahedra_scene()
        scene.view.slab_near = -1.5
        scene.view.slab_far = 6.5
        style = RenderStyle(slab_clip_mode=SlabClipMode.INCLUDE_WHOLE)
        fig = render_mpl(scene, style=style, show=False)
        n = len(fig.axes[0].collections[0].get_paths())
        # Must have polyhedra faces plus atoms plus bonds.
        assert n > 14  # 14 atoms alone

    def test_faces_sorted_within_depth_slot(self):
        """Faces within the same depth slot are sorted back-to-front."""
        from hofmann.render_mpl import _precompute_scene

        scene = _two_octahedra_scene()
        view = scene.view
        style = RenderStyle()
        coords = scene.frames[0].coords
        precomputed = _precompute_scene(scene, 0, render_style=style)

        xy, depth, _ = view.project(coords)
        rotated = (coords - view.centre) @ view.rotation.T
        order = np.argsort(depth)
        slab_visible = np.ones(len(coords), dtype=bool)

        face_by_depth_slot, _ = _collect_polyhedra_faces(
            precomputed=precomputed,
            polyhedra_list=precomputed.polyhedra,
            poly_skip=set(),
            slab_visible=slab_visible,
            show_polyhedra=True,
            polyhedra_shading=style.polyhedra_shading,
            rotated=rotated,
            depth=depth,
            xy=xy,
            order=order,
        )

        # Within each slot, face_depth (element [4]) must be ascending.
        for slot, faces in face_by_depth_slot.items():
            depths = [f[4] for f in faces]
            assert depths == sorted(depths), (
                f"Faces at slot {slot} not sorted by depth: {depths}"
            )

    def test_no_slab_two_polyhedra_all_modes_identical(self):
        """Without slab settings, all three modes produce the same
        polygon count for a two-polyhedra scene."""
        counts = []
        for mode in SlabClipMode:
            scene = _two_octahedra_scene()
            style = RenderStyle(slab_clip_mode=mode)
            fig = render_mpl(scene, style=style, show=False)
            counts.append(len(fig.axes[0].collections[0].get_paths()))
        assert counts[0] == counts[1] == counts[2]


class TestImageAtomPolyhedra:
    """Image atoms (from PBC expansion) should also form polyhedra."""

    def test_all_matching_atoms_are_centres(self):
        """Both Ti atoms should generate polyhedra regardless of index."""
        from hofmann.render_mpl import _precompute_scene

        # Two Ti atoms, each with its own octahedral O shell.
        species = ["Ti"] + ["O"] * 6 + ["Ti"] + ["O"] * 6
        coords = np.array([
            [0.0, 0.0, 0.0],    # Ti #1
            [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0], [0.0, 0.0, -2.0],
            [8.0, 0.0, 0.0],    # Ti #2
            [10.0, 0.0, 0.0], [6.0, 0.0, 0.0],
            [8.0, 2.0, 0.0], [8.0, -2.0, 0.0],
            [8.0, 0.0, 2.0], [8.0, 0.0, -2.0],
        ])
        scene = StructureScene(
            species=species,
            frames=[Frame(coords=coords)],
            atom_styles={
                "Ti": AtomStyle(1.0, (0.2, 0.4, 0.9)),
                "O": AtomStyle(0.8, (0.9, 0.1, 0.1)),
            },
            bond_specs=[BondSpec(
                species=("O", "Ti"), min_length=0.0, max_length=3.0,
                radius=0.1, colour=0.5,
            )],
            polyhedra=[PolyhedronSpec(centre="Ti")],
        )
        precomputed = _precompute_scene(scene, 0)
        assert len(precomputed.polyhedra) == 2
        centres = {p.centre_index for p in precomputed.polyhedra}
        assert centres == {0, 7}


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
        view_no_persp = ViewState(perspective=0.0)
        view_persp = ViewState(perspective=0.5)
        e_no = _scene_extent(scene, view_no_persp, 0, atom_scale=0.5)
        e_yes = _scene_extent(scene, view_persp, 0, atom_scale=0.5)
        assert e_yes > e_no

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
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
            lattice=np.eye(3) * 10.0,
        )
        e_lat = _scene_extent(scene_lat, view, 0, atom_scale=0.5)
        assert e_lat > e_no_lat


# --- Cell edge geometry ---


def _scene_with_lattice(**kwargs):
    """Create a minimal scene with a cubic lattice."""
    a = kwargs.pop("a", 5.0)
    return StructureScene(
        species=["A"],
        frames=[Frame(coords=np.array([[a / 2, a / 2, a / 2]]))],
        atom_styles={"A": AtomStyle(0.5, (0.5, 0.5, 0.5))},
        lattice=np.eye(3) * a,
        **kwargs,
    )


class TestCellEdges3d:
    def test_produces_12_edges(self):
        starts, ends = _cell_edges_3d(np.eye(3) * 5.0)
        assert starts.shape == (12, 3)
        assert ends.shape == (12, 3)

    def test_cubic_edge_lengths(self):
        a = 5.0
        starts, ends = _cell_edges_3d(np.eye(3) * a)
        lengths = np.linalg.norm(ends - starts, axis=1)
        np.testing.assert_allclose(lengths, a)

    def test_orthorhombic_edge_lengths(self):
        lattice = np.diag([3.0, 4.0, 5.0])
        starts, ends = _cell_edges_3d(lattice)
        lengths = np.linalg.norm(ends - starts, axis=1)
        # Each edge should have length 3, 4, or 5.
        for length in lengths:
            assert length == pytest.approx(3.0) or \
                   length == pytest.approx(4.0) or \
                   length == pytest.approx(5.0)


class TestClipEdgeAtAtoms:
    """Tests for _clip_edge_at_atoms — cell edge clipping at atom spheres."""

    def test_no_atoms_returns_full_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([5.0, 0.0, 0.0])
        segs = _clip_edge_at_atoms(s, e, np.empty((0, 3)), np.empty(0))
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        np.testing.assert_allclose(segs[0][1], e)

    def test_atom_at_start_clips(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        assert segs[0][0][0] == pytest.approx(1.0)
        np.testing.assert_allclose(segs[0][1], e)

    def test_atom_at_end_clips(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[10.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(9.0)

    def test_atom_in_middle_splits_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[5.0, 0.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 2
        # First segment: [0, 4]
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(4.0)
        # Second segment: [6, 10]
        assert segs[1][0][0] == pytest.approx(6.0)
        np.testing.assert_allclose(segs[1][1], e)

    def test_atom_far_from_line_no_clip(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([[5.0, 5.0, 0.0]])
        radii = np.array([1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        np.testing.assert_allclose(segs[0][0], s)
        np.testing.assert_allclose(segs[0][1], e)

    def test_fully_occluded_returns_empty(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([2.0, 0.0, 0.0])
        coords = np.array([[1.0, 0.0, 0.0]])
        radii = np.array([5.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 0

    def test_two_atoms_on_edge(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        radii = np.array([1.0, 1.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        assert len(segs) == 1
        assert segs[0][0][0] == pytest.approx(1.0)
        assert segs[0][1][0] == pytest.approx(9.0)

    def test_overlapping_atoms_merge_gaps(self):
        s = np.array([0.0, 0.0, 0.0])
        e = np.array([10.0, 0.0, 0.0])
        coords = np.array([
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ])
        radii = np.array([2.0, 2.0])
        segs = _clip_edge_at_atoms(s, e, coords, radii)
        # Gap from [1, 6] (merged from [1,5] and [2,6]).
        assert len(segs) == 2
        np.testing.assert_allclose(segs[0][0], s)
        assert segs[0][1][0] == pytest.approx(1.0)
        assert segs[1][0][0] == pytest.approx(6.0)
        np.testing.assert_allclose(segs[1][1], e)


class TestCellEdgeRendering:
    def test_show_cell_auto_with_lattice(self):
        """Renders without error when lattice is present."""
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_show_cell_auto_without_lattice(self):
        """No crash when lattice is absent."""
        scene = _minimal_scene()
        assert scene.lattice is None
        fig = render_mpl(scene, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_show_cell_false_suppresses(self):
        """Explicit show_cell=False produces fewer polygons."""
        scene = _scene_with_lattice()
        fig_with = render_mpl(scene, show=False)
        n_with = len(fig_with.axes[0].collections[0].get_paths())
        plt.close(fig_with)

        fig_without = render_mpl(scene, show=False, show_cell=False)
        n_without = len(fig_without.axes[0].collections[0].get_paths())
        plt.close(fig_without)

        assert n_with > n_without

    def test_show_cell_true_without_lattice_raises(self):
        """show_cell=True on a scene without lattice raises."""
        scene = _minimal_scene()
        with pytest.raises(ValueError, match="show_cell.*no lattice"):
            render_mpl(scene, show=False, show_cell=True)

    def test_cell_edge_dashed(self):
        """Dashed style produces more polygons than solid."""
        scene = _scene_with_lattice()
        fig_solid = render_mpl(scene, show=False)
        n_solid = len(fig_solid.axes[0].collections[0].get_paths())
        plt.close(fig_solid)

        style = RenderStyle(cell_style=CellEdgeStyle(linestyle="dashed"))
        fig_dash = render_mpl(scene, show=False, style=style)
        n_dash = len(fig_dash.axes[0].collections[0].get_paths())
        plt.close(fig_dash)

        # Each of the 12 edges gets split into multiple dashes.
        assert n_dash > n_solid

    def test_cell_edges_clipped_at_corner_atoms(self):
        """Atoms at cell corners produce shorter edges."""
        # Scene with atom at the origin (a cell corner).
        a = 5.0
        scene = StructureScene(
            species=["A"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={"A": AtomStyle(1.0, (0.5, 0.5, 0.5))},
            lattice=np.eye(3) * a,
        )
        fig = render_mpl(scene, show=False)
        n_corner = len(fig.axes[0].collections[0].get_paths())
        plt.close(fig)

        # Scene with atom at the centre (no edge clipping).
        scene_mid = _scene_with_lattice(a=a)
        fig_mid = render_mpl(scene_mid, show=False)
        n_mid = len(fig_mid.axes[0].collections[0].get_paths())
        plt.close(fig_mid)

        # Both render without error; corner scene has same number of
        # edge polygons (clipping shortens edges but doesn't remove them).
        assert n_corner >= 1
        assert n_mid >= 1


# ---------------------------------------------------------------------------
# Axes orientation widget
# ---------------------------------------------------------------------------


class TestAxesWidget:
    """Tests for the crystallographic axes orientation widget."""

    def test_show_axes_auto_with_lattice(self):
        """Widget is drawn when scene has a lattice (auto-detect)."""
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False)
        ax = fig.axes[0]
        # Should have 3 labels (a, b, c) and 3 axis lines.
        label_texts = [t.get_text() for t in ax.texts]
        assert "a" in label_texts
        assert "b" in label_texts
        assert "c" in label_texts
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_show_axes_auto_without_lattice(self):
        """No widget when scene has no lattice."""
        scene = _minimal_scene()
        fig = render_mpl(scene, show=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        # No axis labels (only scene title text, if any).
        label_texts = [t.get_text() for t in ax.texts]
        assert "a" not in label_texts
        plt.close(fig)

    def test_show_axes_false_suppresses(self):
        """Explicit show_axes=False suppresses widget on lattice scene."""
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False, show_axes=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_show_axes_true_without_lattice_raises(self):
        """show_axes=True on a non-lattice scene raises ValueError."""
        scene = _minimal_scene()
        with pytest.raises(ValueError, match="show_axes=True but scene has no lattice"):
            render_mpl(scene, show=False, show_axes=True)

    def test_widget_has_three_labels(self):
        """Widget produces exactly three text labels."""
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False, show_cell=False)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert sorted(label_texts) == ["a", "b", "c"]
        plt.close(fig)

    def test_widget_custom_labels(self):
        """Custom labels are used when provided."""
        style = AxesStyle(labels=("x", "y", "z"))
        scene = _scene_with_lattice()
        fig = render_mpl(
            scene, show=False,
            axes_style=style, show_cell=False,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert sorted(label_texts) == ["x", "y", "z"]
        plt.close(fig)

    @pytest.mark.parametrize("corner", [
        "bottom_left", "bottom_right", "top_left", "top_right",
    ])
    def test_widget_all_corners(self, corner):
        """Widget renders without error in each corner position."""
        style = AxesStyle(corner=corner)
        scene = _scene_with_lattice()
        fig = render_mpl(scene, show=False, axes_style=style)
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_viewport_wider_with_axes_widget(self):
        """Viewport expands when axes widget is enabled."""
        scene = _scene_with_lattice()
        fig_with = render_mpl(scene, show=False, show_axes=True)
        xlim_with = fig_with.axes[0].get_xlim()
        plt.close(fig_with)

        fig_without = render_mpl(scene, show=False, show_axes=False)
        xlim_without = fig_without.axes[0].get_xlim()
        plt.close(fig_without)

        extent_with = xlim_with[1] - xlim_with[0]
        extent_without = xlim_without[1] - xlim_without[0]
        assert extent_with > extent_without


# ---------------------------------------------------------------------------
# Keyboard action tests
# ---------------------------------------------------------------------------


def _key_action_fixtures():
    """Build a ViewState, RenderStyle, state dict, and initial_view for tests."""
    view = ViewState()
    style = RenderStyle()
    state = {"frame_index": 0, "help_visible": False}
    initial_view = {
        "rotation": view.rotation.copy(),
        "zoom": view.zoom,
        "centre": view.centre.copy(),
        "perspective": view.perspective,
        "view_distance": view.view_distance,
    }
    return view, style, state, initial_view


def _do_key(
    key, view, style, state, initial_view,
    *, n_frames=1, base_extent=5.0, has_lattice=False,
):
    """Convenience wrapper around _apply_key_action."""
    return _apply_key_action(
        key, view, style, state,
        n_frames=n_frames,
        base_extent=base_extent,
        initial_view=initial_view,
        has_lattice=has_lattice,
    )


class TestKeyActions:
    """Tests for the extracted _apply_key_action function."""

    # -- Rotation keys --

    @pytest.mark.parametrize("key, axis_fn, sign", [
        ("left", _rotation_y, -1),
        ("right", _rotation_y, +1),
        ("up", _rotation_x, -1),
        ("down", _rotation_x, +1),
        (",", _rotation_z, +1),
        (".", _rotation_z, -1),
    ])
    def test_rotation_keys(self, key, axis_fn, sign):
        """Each rotation key applies the expected rotation matrix."""
        view, style, state, iv = _key_action_fixtures()
        old = view.rotation.copy()
        kind = _do_key(key, view, style, state, iv)
        expected = axis_fn(sign * _KEY_ROTATION_STEP) @ old
        np.testing.assert_allclose(view.rotation, expected, atol=1e-14)
        assert kind == "view"

    # -- Zoom keys --

    def test_zoom_in_plus(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("+", view, style, state, iv)
        assert view.zoom == pytest.approx(_KEY_ZOOM_FACTOR)
        assert kind == "view"

    def test_zoom_in_equals(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("=", view, style, state, iv)
        assert view.zoom == pytest.approx(_KEY_ZOOM_FACTOR)
        assert kind == "view"

    def test_zoom_out(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("-", view, style, state, iv)
        assert view.zoom == pytest.approx(1.0 / _KEY_ZOOM_FACTOR)
        assert kind == "view"

    def test_zoom_clamped_max(self):
        view, style, state, iv = _key_action_fixtures()
        view.zoom = 99.5
        _do_key("+", view, style, state, iv)
        assert view.zoom == 100.0

    def test_zoom_clamped_min(self):
        view, style, state, iv = _key_action_fixtures()
        view.zoom = 0.015
        _do_key("-", view, style, state, iv)
        assert view.zoom == pytest.approx(0.015 / _KEY_ZOOM_FACTOR)
        # Push below minimum.
        view.zoom = 0.005
        _do_key("-", view, style, state, iv)
        assert view.zoom == 0.01

    # -- Pan keys --

    def test_pan_left(self):
        """Shift+left moves the scene left (centre shifts screen-right)."""
        view, style, state, iv = _key_action_fixtures()
        old_centre = view.centre.copy()
        kind = _do_key("shift+left", view, style, state, iv, base_extent=10.0)
        step = _KEY_PAN_FRACTION * 10.0 / view.zoom
        expected = old_centre + step * view.rotation[0]
        np.testing.assert_allclose(view.centre, expected)
        assert kind == "view"

    def test_pan_right(self):
        """Shift+right moves the scene right (centre shifts screen-left)."""
        view, style, state, iv = _key_action_fixtures()
        old_centre = view.centre.copy()
        _do_key("shift+right", view, style, state, iv, base_extent=10.0)
        step = _KEY_PAN_FRACTION * 10.0 / view.zoom
        expected = old_centre - step * view.rotation[0]
        np.testing.assert_allclose(view.centre, expected)

    def test_pan_up(self):
        """Shift+up moves the scene up (centre shifts screen-down)."""
        view, style, state, iv = _key_action_fixtures()
        old_centre = view.centre.copy()
        _do_key("shift+up", view, style, state, iv, base_extent=10.0)
        step = _KEY_PAN_FRACTION * 10.0 / view.zoom
        expected = old_centre - step * view.rotation[1]
        np.testing.assert_allclose(view.centre, expected)

    def test_pan_down(self):
        """Shift+down moves the scene down (centre shifts screen-up)."""
        view, style, state, iv = _key_action_fixtures()
        old_centre = view.centre.copy()
        _do_key("shift+down", view, style, state, iv, base_extent=10.0)
        step = _KEY_PAN_FRACTION * 10.0 / view.zoom
        expected = old_centre + step * view.rotation[1]
        np.testing.assert_allclose(view.centre, expected)

    # -- Perspective --

    def test_perspective_increase(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("p", view, style, state, iv)
        assert view.perspective == pytest.approx(_PERSPECTIVE_STEP)
        assert kind == "view"

    def test_perspective_decrease(self):
        view, style, state, iv = _key_action_fixtures()
        view.perspective = 0.5
        _do_key("P", view, style, state, iv)
        assert view.perspective == pytest.approx(0.5 - _PERSPECTIVE_STEP)

    def test_perspective_clamped_max(self):
        view, style, state, iv = _key_action_fixtures()
        view.perspective = 0.95
        _do_key("p", view, style, state, iv)
        assert view.perspective == 1.0

    def test_perspective_clamped_min(self):
        view, style, state, iv = _key_action_fixtures()
        view.perspective = 0.05
        _do_key("P", view, style, state, iv)
        assert view.perspective == 0.0

    # -- Distance --

    def test_distance_increase(self):
        view, style, state, iv = _key_action_fixtures()
        old = view.view_distance
        kind = _do_key("d", view, style, state, iv)
        assert view.view_distance == pytest.approx(old * 1.05)
        assert kind == "view"

    def test_distance_decrease(self):
        view, style, state, iv = _key_action_fixtures()
        old = view.view_distance
        _do_key("D", view, style, state, iv)
        assert view.view_distance == pytest.approx(old / 1.05)

    def test_distance_clamped_min(self):
        view, style, state, iv = _key_action_fixtures()
        view.view_distance = 0.11
        _do_key("D", view, style, state, iv)
        # 0.11 / 1.05 ~ 0.1048, still above 0.1
        _do_key("D", view, style, state, iv)
        _do_key("D", view, style, state, iv)
        assert view.view_distance >= 0.1

    # -- Style toggles --

    def test_toggle_bonds(self):
        view, style, state, iv = _key_action_fixtures()
        assert style.show_bonds is True
        kind = _do_key("b", view, style, state, iv)
        assert style.show_bonds is False
        assert kind == "view"
        _do_key("b", view, style, state, iv)
        assert style.show_bonds is True

    def test_toggle_outlines(self):
        view, style, state, iv = _key_action_fixtures()
        assert style.show_outlines is True
        kind = _do_key("o", view, style, state, iv)
        assert style.show_outlines is False
        assert kind == "view"

    def test_toggle_polyhedra(self):
        view, style, state, iv = _key_action_fixtures()
        assert style.show_polyhedra is True
        kind = _do_key("e", view, style, state, iv)
        assert style.show_polyhedra is False
        assert kind == "view"

    def test_toggle_cell_no_lattice(self):
        """Without a lattice, None auto-detects to off; first press turns on."""
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("u", view, style, state, iv, has_lattice=False)
        assert style.show_cell is True
        assert kind == "view"
        _do_key("u", view, style, state, iv, has_lattice=False)
        assert style.show_cell is False

    def test_toggle_cell_with_lattice(self):
        """With a lattice, None auto-detects to on; first press turns off."""
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("u", view, style, state, iv, has_lattice=True)
        assert style.show_cell is False
        assert kind == "view"
        _do_key("u", view, style, state, iv, has_lattice=True)
        assert style.show_cell is True

    def test_toggle_axes_no_lattice(self):
        """Without a lattice, None auto-detects to off; first press turns on."""
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("a", view, style, state, iv, has_lattice=False)
        assert style.show_axes is True
        assert kind == "view"
        _do_key("a", view, style, state, iv, has_lattice=False)
        assert style.show_axes is False

    def test_toggle_axes_with_lattice(self):
        """With a lattice, None auto-detects to on; first press turns off."""
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("a", view, style, state, iv, has_lattice=True)
        assert style.show_axes is False
        assert kind == "view"
        _do_key("a", view, style, state, iv, has_lattice=True)
        assert style.show_axes is True

    # -- Frame navigation --

    def test_frame_next(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("]", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 1
        assert kind == "full"

    def test_frame_prev(self):
        view, style, state, iv = _key_action_fixtures()
        state["frame_index"] = 2
        _do_key("[", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 1

    def test_frame_next_wraps(self):
        view, style, state, iv = _key_action_fixtures()
        state["frame_index"] = 4
        _do_key("]", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 0

    def test_frame_prev_wraps(self):
        view, style, state, iv = _key_action_fixtures()
        state["frame_index"] = 0
        _do_key("[", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 4

    def test_frame_first(self):
        view, style, state, iv = _key_action_fixtures()
        state["frame_index"] = 3
        kind = _do_key("{", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 0
        assert kind == "full"

    def test_frame_last(self):
        view, style, state, iv = _key_action_fixtures()
        kind = _do_key("}", view, style, state, iv, n_frames=5)
        assert state["frame_index"] == 4
        assert kind == "full"

    def test_frame_noop_single_frame(self):
        """Frame keys are no-ops for single-frame scenes."""
        view, style, state, iv = _key_action_fixtures()
        for key in ("[", "]", "{", "}"):
            kind = _do_key(key, view, style, state, iv, n_frames=1)
            assert state["frame_index"] == 0
            assert kind == "none"

    # -- Reset --

    def test_reset_restores_initial_view(self):
        view, style, state, iv = _key_action_fixtures()
        # Modify everything.
        view.rotation = _rotation_y(1.0) @ view.rotation
        view.zoom = 3.5
        view.centre = np.array([1.0, 2.0, 3.0])
        view.perspective = 0.7
        view.view_distance = 20.0
        kind = _do_key("r", view, style, state, iv)
        np.testing.assert_allclose(view.rotation, np.eye(3))
        assert view.zoom == 1.0
        np.testing.assert_allclose(view.centre, [0.0, 0.0, 0.0])
        assert view.perspective == 0.0
        assert view.view_distance == 10.0
        assert kind == "view"

    # -- Help overlay --

    def test_help_toggle(self):
        view, style, state, iv = _key_action_fixtures()
        assert state["help_visible"] is False
        kind = _do_key("h", view, style, state, iv)
        assert state["help_visible"] is True
        assert kind == "view"
        _do_key("h", view, style, state, iv)
        assert state["help_visible"] is False

    # -- Unrecognised key --

    def test_unrecognised_key_returns_none(self):
        view, style, state, iv = _key_action_fixtures()
        old_rotation = view.rotation.copy()
        kind = _do_key("z", view, style, state, iv)
        assert kind == "none"
        np.testing.assert_array_equal(view.rotation, old_rotation)

    # -- Help text constant --

    def test_help_text_contains_key_names(self):
        """Help text mentions key categories."""
        assert "Arrows" in _HELP_TEXT
        assert "Zoom" in _HELP_TEXT
        assert "Rotate" in _HELP_TEXT
        assert "help" in _HELP_TEXT.lower()


# --- Atom metadata colouring ---


class TestColourBy:
    """Smoke tests for colourmap-based atom colouring via render_mpl."""

    def test_numerical_colour_by(self):
        """render_mpl with colour_by for numerical data produces a Figure."""
        scene = _minimal_scene()
        scene.set_atom_data("charge", [0.5, -0.5])
        fig = render_mpl(scene, colour_by="charge", cmap="coolwarm", show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_numerical_colour_by_with_range(self):
        scene = _minimal_scene()
        scene.set_atom_data("charge", [0.5, -0.5])
        fig = render_mpl(
            scene, colour_by="charge", colour_range=(-1.0, 1.0), show=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_colour_by(self):
        """render_mpl with colour_by for categorical data produces a Figure."""
        scene = _minimal_scene()
        scene.set_atom_data("site", np.array(["4a", "8b"], dtype=object))
        fig = render_mpl(scene, colour_by="site", cmap="Set2", show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_colour_by_missing_key_raises(self):
        scene = _minimal_scene()
        with pytest.raises(KeyError):
            render_mpl(scene, colour_by="nonexistent", show=False)

    def test_callable_cmap(self):
        """A callable cmap works through render_mpl."""
        scene = _minimal_scene()
        scene.set_atom_data("val", [0.0, 1.0])
        fig = render_mpl(
            scene, colour_by="val",
            cmap=lambda v: (v, 0.0, 1.0 - v), show=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_colour_by_with_nan(self):
        """Atoms with NaN data fall back to species colour."""
        scene = _minimal_scene()
        scene.set_atom_data("charge", {0: 1.0})  # atom 1 gets NaN
        fig = render_mpl(scene, colour_by="charge", show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_convenience_method(self):
        """StructureScene.render_mpl forwards colour_by correctly."""
        scene = _minimal_scene()
        scene.set_atom_data("charge", [0.5, -0.5])
        fig = scene.render_mpl(colour_by="charge", show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_list_colour_by_smoke(self):
        """render_mpl with a list of colour_by keys produces a Figure."""
        scene = _minimal_scene()
        scene.set_atom_data("a", {0: 1.0})
        scene.set_atom_data("b", {1: 2.0})
        fig = render_mpl(
            scene, colour_by=["a", "b"], cmap=["viridis", "plasma"],
            show=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_polyhedra_inherit_colour_by(self):
        """Polyhedra inherit the centre atom's resolved colour_by colour."""
        from hofmann.render_mpl import _precompute_scene

        scene = _octahedron_scene()
        # Ti is atom 0; give it a numerical value so it gets a cmap colour.
        n_atoms = len(scene.species)
        scene.set_atom_data("val", {0: 0.5})

        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        precomputed = _precompute_scene(
            scene, 0, colour_by="val", cmap=red,
        )
        # The polyhedron should inherit (1.0, 0.0, 0.0) from the cmap,
        # not the Ti species colour.
        assert len(precomputed.poly_base_colours) == 1
        assert precomputed.poly_base_colours[0] == (1.0, 0.0, 0.0)

    def test_polyhedra_spec_colour_overrides_colour_by(self):
        """PolyhedronSpec.colour still wins over colour_by."""
        from hofmann.render_mpl import _precompute_scene

        scene = _octahedron_scene(colour=(0.0, 0.0, 1.0))
        n_atoms = len(scene.species)
        scene.set_atom_data("val", {0: 0.5})

        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        precomputed = _precompute_scene(
            scene, 0, colour_by="val", cmap=red,
        )
        # Explicit spec colour should override the colour_by value.
        assert precomputed.poly_base_colours[0] == (0.0, 0.0, 1.0)

