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
    Frame,
    Oblique,
    StructureScene,
    ViewState,
)
from hofmann.rendering.axes_widget import _draw_axes_widget
from hofmann.rendering.static import render_mpl


def _oblique_view() -> ViewState:
    return ViewState().look_along([0, -1, 0]).with_oblique(Oblique(35.0, 0.6))


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
