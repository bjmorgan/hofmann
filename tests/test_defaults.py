"""Tests for hofmann.defaults — element colours and covalent radii."""

import pytest

from hofmann.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
)


class TestElementColours:
    def test_carbon_is_grey(self):
        r, g, b = ELEMENT_COLOURS["C"]
        assert r == pytest.approx(g)  # Grey means r == g == b
        assert g == pytest.approx(b)

    def test_oxygen_is_red(self):
        r, g, b = ELEMENT_COLOURS["O"]
        assert r > 0.9
        assert g < 0.1
        assert b < 0.1

    def test_hydrogen_is_white(self):
        assert ELEMENT_COLOURS["H"] == (1.0, 1.0, 1.0)


class TestCovalentRadii:
    def test_hydrogen_radius(self):
        assert COVALENT_RADII["H"] == pytest.approx(0.31)

    def test_carbon_radius(self):
        assert COVALENT_RADII["C"] == pytest.approx(0.76)


class TestDefaultAtomStyle:
    def test_known_element(self):
        style = default_atom_style("O")
        assert style.radius == pytest.approx(0.66)
        r, g, b = style.colour
        assert r > 0.9  # Oxygen is red

    def test_unknown_element_fallback(self):
        style = default_atom_style("Xx")
        assert style.radius == 1.0
        assert style.colour == (0.5, 0.5, 0.5)
