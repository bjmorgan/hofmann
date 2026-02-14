"""Tests for hofmann.bonds — bond computation from BondSpec rules."""

import numpy as np
import pytest

from hofmann.bonds import compute_bonds
from hofmann.model import BondSpec


# CH4 geometry from the test fixture.
CH4_SPECIES = ["C", "H", "H", "H", "H"]
CH4_COORDS = np.array([
    [0.000, 0.000, 0.000],
    [1.155, 1.155, 1.155],
    [-1.155, -1.155, 1.155],
    [1.155, -1.155, -1.155],
    [-1.155, 1.155, -1.155],
])


class TestComputeBonds:
    def test_ch4_c_h_bonds(self):
        specs = [BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 4
        # All bonds should be from atom 0 (C) to atoms 1-4 (H).
        indices = {(b.index_a, b.index_b) for b in bonds}
        assert indices == {(0, 1), (0, 2), (0, 3), (0, 4)}

    def test_ch4_c_h_bond_length(self):
        specs = [BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        expected_length = np.sqrt(3) * 1.155
        for bond in bonds:
            assert bond.length == pytest.approx(expected_length, rel=1e-6)

    def test_ch4_no_hh_bonds(self):
        # H-H distance in this geometry is ~3.267, which exceeds max 2.8.
        specs = [BondSpec("H", "H", 0.0, 2.8, 0.109, 1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 0

    def test_ch4_both_specs(self):
        specs = [
            BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0),
            BondSpec("H", "H", 0.0, 2.8, 0.109, 1.0),
        ]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 4  # Only C-H bonds found.

    def test_empty_specs(self):
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, [])
        assert bonds == []

    def test_empty_coords(self):
        bonds = compute_bonds([], np.zeros((0, 3)), [
            BondSpec("C", "H", 0.0, 3.4, 0.109, 1.0),
        ])
        assert bonds == []

    def test_wildcard_spec(self):
        specs = [BondSpec("*", "*", 0.0, 5.0, 0.1, 1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        # With max 5.0, should find all C-H bonds (dist ~2.0) but
        # H-H distances are ~3.27, so those are also within range.
        assert len(bonds) == 10  # C(4,2) + 4 = 6 H-H + 4 C-H = 10

    def test_first_spec_wins(self):
        spec_a = BondSpec("C", "H", 0.0, 5.0, 0.2, "red")
        spec_b = BondSpec("*", "*", 0.0, 5.0, 0.1, "blue")
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, [spec_a, spec_b])
        # C-H bonds should use spec_a, H-H bonds should use spec_b.
        ch_bonds = [b for b in bonds if b.spec is spec_a]
        hh_bonds = [b for b in bonds if b.spec is spec_b]
        assert len(ch_bonds) == 4
        assert len(hh_bonds) == 6

    def test_min_length_filter(self):
        # Set min_length above actual C-H distance (~2.0) to exclude them.
        specs = [BondSpec("C", "H", 3.0, 5.0, 0.1, 1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 0
