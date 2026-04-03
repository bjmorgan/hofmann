"""Tests for AtomStyle validation."""

import pytest

from hofmann.model.atom_style import AtomStyle


class TestAtomStyleValidation:
    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            AtomStyle(radius=-1.0, colour="red")

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            AtomStyle(radius=0.0, colour="red")

    def test_positive_radius_accepted(self):
        style = AtomStyle(radius=0.5, colour="red")
        assert style.radius == 0.5


class TestAtomStyleConstruction:
    def test_visible_requires_radius_and_colour(self):
        with pytest.raises(ValueError, match="radius"):
            AtomStyle(colour="red")

    def test_visible_requires_colour(self):
        with pytest.raises(ValueError, match="colour"):
            AtomStyle(radius=0.5)

    def test_hidden_without_radius_colour(self):
        style = AtomStyle(visible=False)
        assert style.visible is False
        assert style.radius is None
        assert style.colour is None

    def test_hidden_with_radius_colour(self):
        style = AtomStyle(radius=0.5, colour="red", visible=False)
        assert style.visible is False
        assert style.radius == 0.5

    def test_toggle_visible(self):
        style = AtomStyle(radius=0.5, colour="red", visible=False)
        style.visible = True
        assert style.visible is True
        assert style.radius == 0.5

    def test_toggle_visible_without_style_raises(self):
        style = AtomStyle(visible=False)
        with pytest.raises(ValueError, match="radius and colour"):
            style.visible = True

    def test_positional_args(self):
        """Existing positional API still works."""
        style = AtomStyle(0.5, "red")
        assert style.radius == 0.5

    def test_repr_visible(self):
        style = AtomStyle(0.5, "red")
        r = repr(style)
        assert "0.5" in r
        assert "red" in r

    def test_repr_hidden(self):
        assert "visible=False" in repr(AtomStyle(visible=False))
