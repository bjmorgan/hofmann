"""Tests for interactive rendering — keyboard actions and rotation helpers."""

import numpy as np
import pytest

from hofmann.model import RenderStyle, ViewState
from hofmann.rendering.interactive import (
    _apply_key_action,
    _HELP_TEXT,
    _KEY_PAN_FRACTION,
    _KEY_ROTATION_STEP,
    _KEY_ZOOM_FACTOR,
    _PERSPECTIVE_STEP,
    _rotation_x,
    _rotation_y,
    _rotation_z,
)


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
