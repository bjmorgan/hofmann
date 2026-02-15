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
    def test_ch4_cross_species_only(self):
        """Default should produce only cross-species pairs (no self-bonds)."""
        specs = default_bond_specs(["C", "H"])
        pairs = {s.species for s in specs}
        assert ("C", "H") in pairs
        assert len(specs) == 1

    def test_self_bonds_flag(self):
        """With self_bonds=True, self-pairs are included."""
        specs = default_bond_specs(["C", "H"], self_bonds=True)
        pairs = {s.species for s in specs}
        assert ("C", "C") in pairs
        assert ("C", "H") in pairs
        assert ("H", "H") in pairs
        assert len(specs) == 3

    def test_max_length_is_rounded(self):
        """Max bond length should be rounded to 2 decimal places."""
        specs = default_bond_specs(["C", "H"])
        ch_spec = next(s for s in specs if s.species == ("C", "H"))
        expected = round(COVALENT_RADII["C"] + COVALENT_RADII["H"] + 0.4, 2)
        assert ch_spec.max_length == expected

    def test_custom_tolerance(self):
        """Custom tolerance overrides the default."""
        specs = default_bond_specs(["C", "H"], tolerance=0.6)
        ch_spec = next(s for s in specs if s.species == ("C", "H"))
        expected = round(COVALENT_RADII["C"] + COVALENT_RADII["H"] + 0.6, 2)
        assert ch_spec.max_length == expected

    def test_single_species_no_self_bonds(self):
        """A single species with default self_bonds=False produces no specs."""
        specs = default_bond_specs(["O"])
        assert len(specs) == 0

    def test_single_species_with_self_bonds(self):
        """A single species with self_bonds=True produces one self-pair."""
        specs = default_bond_specs(["O"], self_bonds=True)
        assert len(specs) == 1
        assert specs[0].species == ("O", "O")

    def test_unknown_species_excluded(self):
        """Species not in COVALENT_RADII are skipped."""
        specs = default_bond_specs(["C", "Xx"], self_bonds=True)
        pairs = {s.species for s in specs}
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
