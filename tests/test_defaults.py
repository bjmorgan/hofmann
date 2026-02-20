"""Tests for hofmann.defaults — element colours, radii, and bond defaults."""

import numpy as np
import pytest

from hofmann.construction.defaults import (
    COVALENT_RADII,
    ELEMENT_COLOURS,
    _load_vesta_cutoffs,
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


class TestVestaCutoffs:
    def test_loads_non_empty(self):
        cutoffs = _load_vesta_cutoffs()
        assert len(cutoffs) > 100

    def test_known_pair(self):
        cutoffs = _load_vesta_cutoffs()
        assert ("C", "H") in cutoffs
        assert cutoffs[("C", "H")] == pytest.approx(1.2)

    def test_keys_are_sorted(self):
        """All keys should have elements in alphabetical order."""
        cutoffs = _load_vesta_cutoffs()
        for a, b in cutoffs:
            assert a <= b


class TestDefaultBondSpecs:
    def test_ch4_species(self):
        """C and H should produce C-C and C-H specs from VESTA data."""
        specs = default_bond_specs(["C", "H"])
        pairs = {s.species for s in specs}
        assert ("C", "H") in pairs
        assert ("C", "C") in pairs
        assert len(specs) == 2

    def test_vesta_cutoff_value(self):
        """Max bond length should match the VESTA cutoff."""
        specs = default_bond_specs(["C", "H"])
        ch_spec = next(s for s in specs if s.species == ("C", "H"))
        assert ch_spec.max_length == pytest.approx(1.2)

    def test_self_bond_when_present(self):
        """A single species with a VESTA self-bond entry produces a spec."""
        specs = default_bond_specs(["Si"])
        assert len(specs) == 1
        assert specs[0].species == ("Si", "Si")

    def test_no_self_bond_when_absent(self):
        """A single species without a VESTA entry produces no specs."""
        specs = default_bond_specs(["He"])
        assert len(specs) == 0

    def test_unknown_species_excluded(self):
        """Species not in VESTA cutoffs are skipped."""
        specs = default_bond_specs(["C", "Xx"])
        pairs = {s.species for s in specs}
        assert ("C", "C") in pairs
        assert ("C", "Xx") not in pairs
        assert ("Xx", "Xx") not in pairs

    def test_pair_not_in_vesta(self):
        """A pair absent from VESTA data produces no spec."""
        specs = default_bond_specs(["H"])
        assert len(specs) == 0  # No H-H in VESTA

    def test_custom_bond_appearance(self):
        """bond_radius and bond_colour are passed through."""
        specs = default_bond_specs(
            ["C", "H"], bond_radius=0.2, bond_colour=(1.0, 0.0, 0.0),
        )
        for spec in specs:
            assert spec.radius == 0.2
            assert spec.colour == (1.0, 0.0, 0.0)

    def test_finds_ch4_bonds(self):
        """Generated specs should find C-H bonds at realistic bond length."""
        from hofmann.construction.bonds import compute_bonds

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
