"""Cross-renderer consistency tests for oblique projection.

Every drawn object must obtain screen positions from the shared
camera-to-screen mapping, so cell edges, the axes widget, and bonds
shear identically to the atoms.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hofmann.model import (
    AtomStyle,
    AxesStyle,
    BondSpec,
    CAVALIER,
    Frame,
    Oblique,
    Perspective,
    RenderStyle,
    StructureScene,
    ViewState,
)
from hofmann.rendering.axes_widget import _draw_axes_widget
from hofmann.rendering.static import render_mpl


def _oblique_view() -> ViewState:
    return ViewState().look_along([0, -1, 0]).with_projection(Oblique(35.0, 0.6))


def _make_scene(lattice: np.ndarray) -> StructureScene:
    """Two tiny atoms and one thin bond inside a unit cell.

    Atom radii are small enough that edge clipping at atom spheres is
    inert, so drawn cell-edge segment endpoints reconstruct exactly to
    projected corner positions.
    """
    coords = np.array([[1.0, 1.0, 1.0], [2.0, 1.5, 1.0]])
    return StructureScene(
        species=["A", "B"],
        frames=[Frame(coords=coords, lattice=lattice)],
        atom_styles={
            "A": AtomStyle(0.01, (0.5, 0.5, 0.5)),
            "B": AtomStyle(0.01, (0.8, 0.2, 0.2)),
        },
        bond_specs=[
            BondSpec(species=("A", "B"), min_length=0.0, max_length=5.0,
                     radius=0.005, colour=(0.2, 0.2, 0.2)),
        ],
    )


def _cell_corners(lattice: np.ndarray) -> np.ndarray:
    fracs = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    return fracs @ lattice


def _quad_endpoints(fig) -> np.ndarray:
    """Reconstruct segment endpoints from drawn cell-edge rectangles.

    Cell edges are drawn as 4-vertex rectangles built as
    [start+offset, end+offset, end-offset, start-offset], so the
    midpoints of the two short sides recover start and end exactly
    (the half-width offsets cancel).  In this scene the only 4-vertex
    polygons are cell-edge rectangles: atoms are many-vertex circles
    and bonds are many-vertex arc polygons.
    """
    endpoints = []
    ax = fig.axes[0]
    for pc in ax.collections:
        for path in pc.get_paths():
            v = path.vertices
            if len(v) in (4, 5):  # 5 = closed polygon repeats vertex 0
                q = np.asarray(v[:4])
                endpoints.append((q[0] + q[3]) / 2)
                endpoints.append((q[1] + q[2]) / 2)
    return np.array(endpoints)


class TestCellEdgesShearConsistently:
    def test_corner_endpoints_match_projection(self):
        """Every projected cell corner appears among the drawn
        cell-edge segment endpoints under oblique projection."""
        lattice = np.diag([3.0, 4.0, 5.0])
        scene = _make_scene(lattice)
        scene.view = _oblique_view()
        fig = render_mpl(scene, show=False)
        try:
            drawn = _quad_endpoints(fig)
            assert len(drawn) > 0
            expected_xy, _, _ = scene.view.project(_cell_corners(lattice))
            for corner in expected_xy:
                dists = np.linalg.norm(drawn - corner, axis=1)
                assert dists.min() < 1e-8, (
                    f"projected corner {corner} not found among drawn "
                    f"cell-edge endpoints (closest {dists.min():.3e})"
                )
        finally:
            plt.close(fig)

    def test_corner_endpoints_match_projection_perspective(self):
        """Same agreement property under perspective: pins the
        rewritten cell-edge projection path for perspective views,
        which previously had no integration coverage."""
        lattice = np.diag([3.0, 4.0, 5.0])
        scene = _make_scene(lattice)
        scene.view = ViewState(projection=Perspective(0.5, 30.0))
        fig = render_mpl(scene, show=False)
        try:
            drawn = _quad_endpoints(fig)
            assert len(drawn) > 0
            expected_xy, _, _ = scene.view.project(_cell_corners(lattice))
            for corner in expected_xy:
                dists = np.linalg.norm(drawn - corner, axis=1)
                assert dists.min() < 1e-8, (
                    f"projected corner {corner} not found among drawn "
                    f"cell-edge endpoints (closest {dists.min():.3e})"
                )
        finally:
            plt.close(fig)


class TestAxesWidgetShearsConsistently:
    def test_tips_match_screen_matrix(self):
        """Axis-triad tips must advertise the sheared directions the
        figure actually uses."""
        lattice = np.eye(3) * 3.0
        view = _oblique_view()
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(-10.0, 10.0)
            ax.set_ylim(-10.0, 10.0)
            style = AxesStyle()
            _draw_axes_widget(ax, lattice, view, style)

            pad = 10.0  # half-extent of the square limits above
            arrow_len = style.arrow_length * pad
            directions = lattice / np.linalg.norm(lattice, axis=1)[:, None]
            expected_tips = (
                (directions @ view.rotation.T) @ view.screen_matrix.T
                * arrow_len
            )

            # The widget draws exactly three axis lines, each from the
            # common origin to origin + tip.
            deltas = np.array([
                np.asarray(line.get_xydata())[1]
                - np.asarray(line.get_xydata())[0]
                for line in ax.lines
            ])
            assert len(deltas) == 3
            for tip in expected_tips:
                dists = np.linalg.norm(deltas - tip, axis=1)
                assert dists.min() < 1e-9, (
                    f"expected sheared tip {tip} not drawn "
                    f"(closest delta off by {dists.min():.3e})"
                )
        finally:
            plt.close(fig)

    def test_widget_lines_stay_inside_axes_under_cavalier(self):
        """The corner inset allows for sheared tip reach, so no axis
        line may poke outside the axes limits even at full cavalier
        foreshortening."""
        lattice = np.eye(3) * 3.0
        view = ViewState().look_along([0, -1, 0]).with_projection(CAVALIER)
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(-10.0, 10.0)
            ax.set_ylim(-10.0, 10.0)
            _draw_axes_widget(ax, lattice, view, AxesStyle())
            for line in ax.lines:
                data = np.asarray(line.get_xydata())
                assert data[:, 0].min() >= -10.0 and data[:, 0].max() <= 10.0
                assert data[:, 1].min() >= -10.0 and data[:, 1].max() <= 10.0
        finally:
            plt.close(fig)


class TestWidgetViewportExpansion:
    def test_expansion_grows_with_scale_bound(self):
        """The viewport expansion for the axes widget must allow for
        sheared tip reach: rendering with the widget on expands the
        limits by (margin + 2 * arrow_length * screen_scale_bound) / 2
        per side relative to the widget-off render."""
        lattice = np.diag([3.0, 4.0, 5.0])
        scene = _make_scene(lattice)
        scene.view = _oblique_view()

        fig_on = render_mpl(scene, show=False, show_axes=True)
        fig_off = render_mpl(scene, show=False, show_axes=False)
        try:
            w_on = np.diff(fig_on.axes[0].get_xlim())[0]
            w_off = np.diff(fig_off.axes[0].get_xlim())[0]
            axes_style = RenderStyle().axes_style
            widget_frac = (
                axes_style.margin
                + 2.0 * axes_style.arrow_length
                * scene.view.screen_scale_bound
            )
            np.testing.assert_allclose(w_on / w_off, 1.0 + widget_frac / 2.0)
        finally:
            plt.close(fig_on)
            plt.close(fig_off)


class TestFullSceneConsistency:
    """Property test: drawn geometry agrees with ViewState.project.

    The pairwise tests above check the known screen-mapping sites; this
    checks the property itself, so a site *not* on the known list that
    bypasses project_camera fails here too.
    """

    def test_bond_drawn_along_projected_direction(self):
        """The bond polygon must run between the projected atom
        positions.  Its vertices are tangent- and half-width-offset
        from the centres, but they all lie along the projected
        bond direction — collinearity is exact regardless of offsets."""
        lattice = np.diag([3.0, 4.0, 5.0])
        scene = _make_scene(lattice)
        scene.view = _oblique_view()
        fig = render_mpl(scene, show=False)
        try:
            atoms_xy, _, _ = scene.view.project(scene.frames[0].coords)
            a2d, b2d = atoms_xy
            u = (b2d - a2d) / np.linalg.norm(b2d - a2d)
            perp = np.array([-u[1], u[0]])
            bond_len = np.linalg.norm(b2d - a2d)

            # The bond is the only large many-vertex polygon: atoms
            # are tiny (radius 0.01) circles, cell edges are 4-vertex
            # rectangles.
            bond_polys = []
            ax = fig.axes[0]
            for pc in ax.collections:
                for path in pc.get_paths():
                    v = np.asarray(path.vertices)
                    if len(v) > 5 and np.ptp(v, axis=0).max() > 0.5:
                        bond_polys.append(v)
            assert len(bond_polys) >= 1

            for v in bond_polys:
                rel = v - a2d
                # Perpendicular deviation bounded by the bond's screen
                # half-width (radius 0.005) plus tolerance.
                assert np.abs(rel @ perp).max() < 0.02
                # Vertices span between the two projected centres.
                along = rel @ u
                assert along.min() > -0.02
                assert along.max() < bond_len + 0.02
        finally:
            plt.close(fig)
