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
