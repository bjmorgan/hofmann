"""Tests for the crystallographic axes widget tip projection."""

import numpy as np

from hofmann.model import Oblique, Orthographic, ViewState
from hofmann.rendering.axes_widget import _axis_tip_offsets


def _rotation() -> np.ndarray:
    """A generic rotation giving every axis a non-zero screen depth."""
    view = ViewState()
    view.look_along([1.0, 0.6, 0.4])
    return view.rotation


class TestAxisTipOffsets:
    def test_orthographic_drops_z(self):
        """Under orthographic the tips are the plain xy drop of the
        rotated axes, scaled by the arrow length."""
        directions = np.eye(3)
        arrow_len = 0.3
        view = ViewState(rotation=_rotation(), projection=Orthographic())

        tips = _axis_tip_offsets(directions, view, arrow_len)

        expected = (directions @ view.rotation.T)[:, :2] * arrow_len
        np.testing.assert_allclose(tips, expected)

    def test_oblique_shears_the_receding_axis(self):
        """Under oblique the tips gain the screen-matrix shear, moving
        away from the plain orthographic drop."""
        directions = np.eye(3)
        arrow_len = 0.3
        rotation = _rotation()
        view = ViewState(rotation=rotation, projection=Oblique(45.0, 0.5))

        tips = _axis_tip_offsets(directions, view, arrow_len)

        projected = directions @ rotation.T
        expected = projected @ view.projection.screen_matrix.T * arrow_len
        np.testing.assert_allclose(tips, expected)

        ortho = projected[:, :2] * arrow_len
        assert not np.allclose(tips, ortho)
