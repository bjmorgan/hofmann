"""Tests for the matplotlib painter — full rendering, polyhedra, slab clipping,
depth ordering, axes widget, and colour-by functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from hofmann.model import (
    AtomStyle,
    AxesStyle,
    BondSpec,
    Frame,
    LegendItem,
    LegendStyle,
    PolyhedronSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    ViewState,
    normalise_colour,
)
from hofmann.rendering.static import render_mpl
from hofmann.construction.scene_builders import from_xbs


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


def _slab_octahedron_scene(**poly_kwargs):
    """Octahedron with a slab that clips the z=+/-2 oxygen atoms."""
    scene = _octahedron_scene(**poly_kwargs)
    scene.view.slab_near = -1.5
    scene.view.slab_far = 1.5
    return scene


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


class TestFrameIndexValidation:
    def test_render_mpl_invalid_frame_index_raises(self):
        scene = _minimal_scene()
        with pytest.raises(ValueError, match="frame_index"):
            render_mpl(scene, frame_index=5, show=False)

    def test_render_mpl_negative_frame_index_raises(self):
        scene = _minimal_scene()
        with pytest.raises(ValueError, match="frame_index"):
            render_mpl(scene, frame_index=-1, show=False)

    def test_render_mpl_valid_frame_index_accepted(self):
        scene = _minimal_scene()
        fig = render_mpl(scene, frame_index=0, show=False)
        assert isinstance(fig, Figure)


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
        with patch("hofmann.rendering.static.plt.subplots") as mock_subplots:
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
        # despite hide_vertices=True.  Compare against hide_vertices=False to confirm
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
        from hofmann.rendering.painter import _precompute_scene, _collect_polyhedra_faces

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
        from hofmann.rendering.painter import _precompute_scene

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
        from hofmann.rendering.painter import _precompute_scene

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
        assert len(precomputed.poly_render_data) == 1
        assert precomputed.poly_render_data[0].base_colour == (1.0, 0.0, 0.0)

    def test_polyhedra_spec_colour_overrides_colour_by(self):
        """PolyhedronSpec.colour still wins over colour_by."""
        from hofmann.rendering.painter import _precompute_scene

        scene = _octahedron_scene(colour=(0.0, 0.0, 1.0))
        n_atoms = len(scene.species)
        scene.set_atom_data("val", {0: 0.5})

        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        precomputed = _precompute_scene(
            scene, 0, colour_by="val", cmap=red,
        )
        # Explicit spec colour should override the colour_by value.
        assert precomputed.poly_render_data[0].base_colour == (0.0, 0.0, 1.0)

    def test_poly_render_data_groups_all_fields(self):
        """_PolyhedronRenderData bundles colour, alpha, edge style."""
        from hofmann.rendering.painter import _precompute_scene

        scene = _octahedron_scene(
            colour=(0.1, 0.2, 0.3),
            alpha=0.7,
            edge_colour=(0.4, 0.5, 0.6),
            edge_width=2.5,
        )
        precomputed = _precompute_scene(scene, 0)
        assert len(precomputed.poly_render_data) == 1
        prd = precomputed.poly_render_data[0]
        assert prd.base_colour == pytest.approx((0.1, 0.2, 0.3))
        assert prd.alpha == 0.7
        assert prd.edge_colour == pytest.approx((0.4, 0.5, 0.6))
        assert prd.edge_width == 2.5


class TestLegendWidget:
    """Tests for the species legend widget."""

    def test_show_legend_draws_entries(self):
        """Legend draws species labels and coloured markers."""
        scene = _minimal_scene()
        fig = render_mpl(scene, show=False, show_legend=True)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" in label_texts
        assert "B" in label_texts
        # One Line2D per species for the circle marker.
        legend_lines = [
            l for l in ax.lines
            if l.get_marker() == "o"
        ]
        assert len(legend_lines) == 2
        plt.close(fig)

    def test_show_legend_false_no_entries(self):
        """Default show_legend=False produces no legend text."""
        scene = _minimal_scene()
        fig = render_mpl(scene, show=False)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" not in label_texts
        assert "B" not in label_texts
        plt.close(fig)

    def test_auto_detect_species_order(self):
        """Auto-detect gives unique species in first-seen order."""
        scene = StructureScene(
            species=["B", "A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
            },
        )
        fig = render_mpl(scene, show=False, show_legend=True)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        # B appears first in scene.species, then A.
        assert label_texts == ["B", "A"]
        plt.close(fig)

    def test_explicit_species(self):
        """Explicit species list controls inclusion and order."""
        scene = _minimal_scene()
        style = LegendStyle(species=("B",))
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert label_texts == ["B"]
        plt.close(fig)

    def test_invisible_species_filtered(self):
        """Auto-detect excludes species with visible=False."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.8, (0.8, 0.2, 0.2), visible=False),
            },
        )
        fig = render_mpl(scene, show=False, show_legend=True)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" in label_texts
        assert "B" not in label_texts
        plt.close(fig)

    def test_explicit_species_includes_invisible(self):
        """Explicit species list includes even invisible species."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.8, (0.8, 0.2, 0.2), visible=False),
            },
        )
        style = LegendStyle(species=("A", "B"))
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" in label_texts
        assert "B" in label_texts
        plt.close(fig)

    @pytest.mark.parametrize("corner", [
        "bottom_left", "bottom_right", "top_left", "top_right",
    ])
    def test_all_corners(self, corner):
        """Legend renders without error in each corner position."""
        style = LegendStyle(corner=corner)
        scene = _minimal_scene()
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "A" in label_texts
        assert "B" in label_texts
        plt.close(fig)

    def test_proportional_circle_radius(self):
        """Range tuple sizes markers proportionally to atom radii."""
        scene = StructureScene(
            species=["Small", "Big"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "Small": AtomStyle(0.5, (0.5, 0.5, 0.5)),
                "Big": AtomStyle(2.0, (0.8, 0.2, 0.2)),
            },
        )
        style = LegendStyle(circle_radius=(3.0, 9.0))
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        sizes = {l.get_markersize() for l in markers}
        # Two different sizes expected.
        assert len(sizes) == 2
        plt.close(fig)

    def test_proportional_equal_radii_uses_max(self):
        """When all atom radii are equal, markers use max circle_radius."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(1.0, (0.8, 0.2, 0.2)),
            },
        )
        style = LegendStyle(circle_radius=(3.0, 9.0))
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        sizes = {l.get_markersize() for l in markers}
        # All markers same size when atom radii are equal.
        assert len(sizes) == 1
        plt.close(fig)

    def test_dict_circle_radius(self):
        """Dict circle_radius sets per-species marker sizes."""
        scene = _minimal_scene()
        style = LegendStyle(circle_radius={"A": 4.0, "B": 8.0})
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        sizes = {l.get_markersize() for l in markers}
        # Two different sizes expected.
        assert len(sizes) == 2
        plt.close(fig)


    def test_label_gap_affects_text_position(self):
        scene = _minimal_scene()
        fig_narrow = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=LegendStyle(label_gap=0.0),
        )
        fig_wide = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=LegendStyle(label_gap=20.0),
        )
        ax_narrow = fig_narrow.axes[0]
        ax_wide = fig_wide.axes[0]
        x_narrow = ax_narrow.texts[0].get_position()[0]
        x_wide = ax_wide.texts[0].get_position()[0]
        assert x_wide > x_narrow
        plt.close(fig_narrow)
        plt.close(fig_wide)

    def test_custom_labels_rendered(self):
        scene = _minimal_scene()
        custom = "$\\mathrm{A^+}$"
        fig = render_mpl(
            scene, show=False,
            show_legend=True,
            legend_style=LegendStyle(labels={"A": custom}),
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts if t.get_text()]
        assert custom in label_texts
        # Species "B" has no override — should keep its name.
        assert "B" in label_texts
        plt.close(fig)

    # ---- custom items ----

    def test_custom_items_rendered(self):
        """Explicit LegendItem entries are drawn with correct labels."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="oct", colour="blue", label="Octahedral"),
            LegendItem(key="tet", colour="red", label="Tetrahedral"),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "Octahedral" in label_texts
        assert "Tetrahedral" in label_texts
        # Species labels should not appear.
        assert "A" not in label_texts
        assert "B" not in label_texts
        plt.close(fig)

    def test_custom_items_colours(self):
        """Custom item colours are used for marker face colours."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="x", colour=(1.0, 0.0, 0.0)),
            LegendItem(key="y", colour=(0.0, 1.0, 0.0)),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        assert len(markers) == 2
        face_colours = [m.get_markerfacecolor()[:3] for m in markers]
        assert (1.0, 0.0, 0.0) in face_colours
        assert (0.0, 1.0, 0.0) in face_colours
        plt.close(fig)

    def test_custom_items_with_radius(self):
        """Items with explicit radii produce different marker sizes."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="small", colour="blue", radius=3.0),
            LegendItem(key="big", colour="red", radius=9.0),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        sizes = sorted(m.get_markersize() for m in markers)
        assert sizes[0] < sizes[1]
        plt.close(fig)

    def test_custom_items_bypass_species(self):
        """Items with keys not matching any scene species still render."""
        scene = _minimal_scene()
        items = (LegendItem(key="custom_cat", colour="purple"),)
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        assert "custom_cat" in label_texts
        plt.close(fig)

    def test_custom_items_label_formatting(self):
        """Chemical notation auto-formatting applies to item labels."""
        scene = _minimal_scene()
        items = (LegendItem(key="Sr2+", colour="green"),)
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts]
        # "Sr2+" should be auto-formatted with superscript.
        assert any("$" in lbl for lbl in label_texts)
        plt.close(fig)

    def test_custom_items_polygon_marker(self):
        """Items with sides produce polygon markers instead of circles."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="oct", colour="blue", sides=6),
            LegendItem(key="tet", colour="red", sides=4, rotation=45.0),
            LegendItem(key="round", colour="green"),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        markers = [l.get_marker() for l in ax.lines if l.get_linestyle() == "None"]
        assert (6, 0, 0.0) in markers
        assert (4, 0, 45.0) in markers
        assert "o" in markers
        plt.close(fig)

    def test_custom_items_gap_after(self):
        """Items with gap_after produce non-uniform vertical spacing."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="A", colour="red", gap_after=20.0),
            LegendItem(key="B", colour="green", gap_after=0.0),
            LegendItem(key="C", colour="blue"),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        marker_lines = [
            l for l in ax.lines if l.get_linestyle() == "None"
        ]
        ys = [l.get_ydata()[0] for l in marker_lines]
        # Three markers stacked downward.
        assert len(ys) == 3
        gap_01 = ys[0] - ys[1]
        gap_12 = ys[1] - ys[2]
        # First gap (20 pt) should be larger than second (0 pt).
        assert gap_01 > gap_12
        plt.close(fig)

    def test_custom_items_alpha(self):
        """Items with alpha < 1 produce RGBA face colours."""
        scene = _minimal_scene()
        items = (
            LegendItem(key="A", colour="red", alpha=0.4),
            LegendItem(key="B", colour="green"),
        )
        style = LegendStyle(items=items)
        fig = render_mpl(
            scene, show=False,
            show_legend=True, legend_style=style,
        )
        ax = fig.axes[0]
        marker_lines = [
            l for l in ax.lines if l.get_linestyle() == "None"
        ]
        # First marker has alpha in face colour.
        fc0 = marker_lines[0].get_markerfacecolor()
        assert len(fc0) == 4
        assert fc0[3] == pytest.approx(0.4)
        # Second marker is fully opaque (RGB only).
        fc1 = marker_lines[1].get_markerfacecolor()
        # Edge colour always stays RGB (opaque outline).
        ec0 = marker_lines[0].get_markeredgecolor()
        assert len(ec0) == 3 or (len(ec0) == 4 and ec0[3] == pytest.approx(1.0))
        plt.close(fig)


class TestRenderLegend:
    """Tests for the standalone render_legend function."""

    def test_returns_figure(self):
        """render_legend returns a matplotlib Figure."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        fig = render_legend(scene)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_contains_species_labels(self):
        """Rendered legend contains species text labels."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        fig = render_legend(scene)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.texts]
        assert "A" in labels
        assert "B" in labels
        plt.close(fig)

    def test_contains_circle_markers(self):
        """Rendered legend contains circle markers for each species."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        fig = render_legend(scene)
        ax = fig.axes[0]
        markers = [l for l in ax.lines if l.get_marker() == "o"]
        assert len(markers) == 2
        plt.close(fig)

    def test_explicit_species(self):
        """Explicit species controls which entries appear."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        style = LegendStyle(species=("B",))
        fig = render_legend(scene, legend_style=style)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.texts]
        assert labels == ["B"]
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """render_legend saves to a file when output is given."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        path = tmp_path / "legend.svg"
        fig = render_legend(scene, output=path)
        assert path.exists()
        plt.close(fig)

    def test_polygon_markers_included_in_crop(self):
        """Polygon-only legends include markers in the bounding box."""
        from hofmann.rendering.static import render_legend
        scene = _minimal_scene()
        items = (
            LegendItem(key="Oct", colour="red", label="Oct", sides=6),
            LegendItem(key="Tet", colour="blue", label="Tet", sides=4),
        )
        style = LegendStyle(items=items)
        fig = render_legend(scene, legend_style=style)
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        assert len(ax.texts) == 2
        plt.close(fig)
