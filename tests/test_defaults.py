"""Tests for hofmann.defaults — element colours and covalent radii."""

import pytest

from hofmann.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
)


class TestElementColours:
    def test_carbon_is_dark(self):
        r, g, b = ELEMENT_COLOURS["C"]
        assert max(r, g, b) < 0.5  # Dark grey

    def test_oxygen_is_red(self):
        r, g, b = ELEMENT_COLOURS["O"]
        assert r > g  # Red-dominant
        assert r > b

    def test_hydrogen_is_light(self):
        r, g, b = ELEMENT_COLOURS["H"]
        assert min(r, g, b) > 0.7  # Pale / near-white


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
        assert r > g  # Oxygen is red-dominant

    def test_unknown_element_fallback(self):
        style = default_atom_style("Xx")
        assert style.radius == 1.0
        assert style.colour == (0.5, 0.5, 0.5)
