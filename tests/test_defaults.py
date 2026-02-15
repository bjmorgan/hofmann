"""Tests for hofmann.defaults — element colours and covalent radii."""

import numpy as np
import pytest

from hofmann.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    default_atom_style,
    default_bond_specs,
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


class TestDefaultBondSpecs:
    def test_ch4_produces_all_pairs(self):
        """CH4 species should produce C-C, C-H, and H-H bond specs."""
        specs = default_bond_specs(["C", "H"])
        pairs = {(s.species_a, s.species_b) for s in specs}
        assert ("C", "C") in pairs
        assert ("C", "H") in pairs
        assert ("H", "H") in pairs

    def test_max_length_is_sum_plus_tolerance(self):
        """Max bond length should be r_a + r_b + tolerance."""
        specs = default_bond_specs(["C", "H"])
        ch_spec = next(
            s for s in specs if s.species_a == "C" and s.species_b == "H"
        )
        expected = COVALENT_RADII["C"] + COVALENT_RADII["H"] + 0.4
        assert ch_spec.max_length == pytest.approx(expected)

    def test_custom_tolerance(self):
        """Custom tolerance overrides the default."""
        specs = default_bond_specs(["C", "H"], tolerance=0.6)
        ch_spec = next(
            s for s in specs if s.species_a == "C" and s.species_b == "H"
        )
        expected = COVALENT_RADII["C"] + COVALENT_RADII["H"] + 0.6
        assert ch_spec.max_length == pytest.approx(expected)

    def test_single_species(self):
        """A single species should produce one self-pair spec."""
        specs = default_bond_specs(["O"])
        assert len(specs) == 1
        assert specs[0].species_a == "O"
        assert specs[0].species_b == "O"

    def test_unknown_species_excluded(self):
        """Species not in COVALENT_RADII are skipped."""
        specs = default_bond_specs(["C", "Xx"])
        pairs = {(s.species_a, s.species_b) for s in specs}
        assert ("C", "C") in pairs
        assert ("C", "Xx") not in pairs
        assert ("Xx", "Xx") not in pairs

    def test_finds_ch4_bonds(self):
        """Generated specs should find C-H bonds at realistic bond length."""
        from hofmann.bonds import compute_bonds

        # Realistic CH4: C-H distance ~ 1.09 A (tetrahedral, r = 0.6294).
        r = 1.09 / np.sqrt(3)
        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [r, r, r],
            [-r, -r, r],
            [r, -r, -r],
            [-r, r, -r],
        ])
        specs = default_bond_specs(["C", "H"])
        bonds = compute_bonds(species, coords, specs)
        ch_bonds = [
            b for b in bonds
            if (species[b.index_a] == "C" and species[b.index_b] == "H")
            or (species[b.index_a] == "H" and species[b.index_b] == "C")
        ]
        assert len(ch_bonds) == 4
