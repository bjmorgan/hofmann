"""Tests for hofmann.bonds — bond computation from BondSpec rules."""

import numpy as np
import pytest

from hofmann.construction.bonds import compute_bonds
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
        specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4, radius=0.109, colour=1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 4
        # All bonds should be from atom 0 (C) to atoms 1-4 (H).
        indices = {(b.index_a, b.index_b) for b in bonds}
        assert indices == {(0, 1), (0, 2), (0, 3), (0, 4)}

    def test_ch4_c_h_bond_length(self):
        specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4, radius=0.109, colour=1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        expected_length = np.sqrt(3) * 1.155
        for bond in bonds:
            assert bond.length == pytest.approx(expected_length, rel=1e-6)

    def test_ch4_no_hh_bonds(self):
        # H-H distance in this geometry is ~3.267, which exceeds max 2.8.
        specs = [BondSpec(species=("H", "H"), min_length=0.0, max_length=2.8, radius=0.109, colour=1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 0

    def test_ch4_both_specs(self):
        specs = [
            BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4, radius=0.109, colour=1.0),
            BondSpec(species=("H", "H"), min_length=0.0, max_length=2.8, radius=0.109, colour=1.0),
        ]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 4  # Only C-H bonds found.

    def test_empty_specs(self):
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, [])
        assert bonds == []

    def test_empty_coords(self):
        bonds = compute_bonds([], np.zeros((0, 3)), [
            BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4, radius=0.109, colour=1.0),
        ])
        assert bonds == []

    def test_wildcard_spec(self):
        specs = [BondSpec(species=("*", "*"), min_length=0.0, max_length=5.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        # With max 5.0, should find all C-H bonds (dist ~2.0) but
        # H-H distances are ~3.27, so those are also within range.
        assert len(bonds) == 10  # C(4,2) + 4 = 6 H-H + 4 C-H = 10

    def test_first_spec_wins(self):
        spec_a = BondSpec(species=("C", "H"), min_length=0.0, max_length=5.0,
                          radius=0.2, colour="red")
        spec_b = BondSpec(species=("*", "*"), min_length=0.0, max_length=5.0,
                          radius=0.1, colour="blue")
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, [spec_a, spec_b])
        # C-H bonds should use spec_a, H-H bonds should use spec_b.
        ch_bonds = [b for b in bonds if b.spec is spec_a]
        hh_bonds = [b for b in bonds if b.spec is spec_b]
        assert len(ch_bonds) == 4
        assert len(hh_bonds) == 6

    def test_min_length_filter(self):
        # Set min_length above actual C-H distance (~2.0) to exclude them.
        specs = [BondSpec(species=("C", "H"), min_length=3.0, max_length=5.0,
                          radius=0.1, colour=1.0)]
        bonds = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        assert len(bonds) == 0

    def test_species_coords_length_mismatch(self):
        specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4,
                          radius=0.1, colour=1.0)]
        wrong_coords = CH4_COORDS[:3]  # 3 rows but 5 species
        with pytest.raises(ValueError, match="species.*5.*coords.*3"):
            compute_bonds(CH4_SPECIES, wrong_coords, specs)

    def test_coords_wrong_columns_raises(self):
        specs = [BondSpec(species=("C", "H"), min_length=0.0, max_length=3.4,
                          radius=0.1, colour=1.0)]
        coords_2d = CH4_COORDS[:, :2]  # (5, 2) instead of (5, 3)
        with pytest.raises(ValueError, match="3 columns"):
            compute_bonds(CH4_SPECIES, coords_2d, specs)


class TestPeriodicBonds:
    """Tests for periodic-aware bond computation with lattice parameter."""

    def test_lattice_none_identical_to_default(self):
        """Passing lattice=None gives identical results to omitting it."""
        specs = [BondSpec(species=("C", "H"), min_length=0.0,
                          max_length=3.4, radius=0.109, colour=1.0)]
        bonds_default = compute_bonds(CH4_SPECIES, CH4_COORDS, specs)
        bonds_none = compute_bonds(CH4_SPECIES, CH4_COORDS, specs,
                                   lattice=None)
        assert bonds_default == bonds_none

    def test_periodic_bond_across_boundary(self):
        """Two atoms on opposite sides of a cell bond through the boundary."""
        species = ["X", "X"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        # Direct distance: 4.0, MIC distance: 1.0
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 1
        bond = bonds[0]
        assert bond.index_a == 0
        assert bond.index_b == 1
        assert bond.length == pytest.approx(1.0)
        assert bond.image == (-1, 0, 0)

    def test_no_bond_when_mic_distance_exceeds_threshold(self):
        """Bond not found when MIC distance exceeds max_length."""
        species = ["X", "X"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=0.5, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 0

    def test_direct_bond_has_zero_image(self):
        """A bond within a large cell has image (0, 0, 0)."""
        species = ["X", "X"]
        lattice = np.diag([10.0, 10.0, 10.0])
        coords = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 1
        assert bonds[0].image == (0, 0, 0)

    def test_image_vector_reconstructs_bond(self):
        """coords[b] + image @ lattice is the bonded position of atom b."""
        species = ["A", "B"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 2.5, 2.5],
            [4.5, 2.5, 2.5],
        ])
        specs = [BondSpec(species=("A", "B"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        bond = bonds[0]
        b_shifted = coords[bond.index_b] + np.array(bond.image) @ lattice
        distance = np.linalg.norm(b_shifted - coords[bond.index_a])
        assert distance == pytest.approx(bond.length)

    def test_symmetric_species_matching_with_lattice(self):
        """Symmetric species matching works for periodic bonds."""
        species = ["A", "B"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        # Spec written as ("B", "A") — should still match.
        specs = [BondSpec(species=("B", "A"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 1

    def test_mixed_direct_and_periodic_bonds(self):
        """Scene with both direct and periodic bonds."""
        species = ["A", "A", "A"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        specs = [BondSpec(species=("A", "A"), min_length=0.0,
                          max_length=1.5, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        bond_dict = {(b.index_a, b.index_b): b for b in bonds}
        # 0-1: direct, distance 1.0
        assert bond_dict[(0, 1)].image == (0, 0, 0)
        assert bond_dict[(0, 1)].length == pytest.approx(1.0)
        # 0-2: periodic through boundary, MIC distance 0.5
        assert bond_dict[(0, 2)].image == (-1, 0, 0)
        assert bond_dict[(0, 2)].length == pytest.approx(0.5)
        # 1-2: periodic, MIC distance 1.5
        assert bond_dict[(1, 2)].image == (-1, 0, 0)
        assert bond_dict[(1, 2)].length == pytest.approx(1.5)

    def test_first_spec_wins_with_lattice(self):
        """First matching spec claims the bond pair with periodic bonds."""
        species = ["A", "B"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ])
        spec_ab = BondSpec(species=("A", "B"), min_length=0.0,
                           max_length=2.0, radius=0.2, colour="red")
        spec_any = BondSpec(species=("*", "*"), min_length=0.0,
                            max_length=2.0, radius=0.1, colour="blue")
        bonds = compute_bonds(species, coords, [spec_ab, spec_any],
                              lattice=lattice)
        assert len(bonds) == 1
        assert bonds[0].spec is spec_ab


class TestSelfBonds:
    """Tests for atoms bonding to their own periodic images."""

    def test_self_bond_across_boundary(self):
        """Atom bonds to its own periodic image along a short lattice vector."""
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.5, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 2
        images = {b.image for b in bonds}
        assert images == {(1, 0, 0), (-1, 0, 0)}
        for b in bonds:
            assert b.index_a == 0
            assert b.index_b == 0
            assert b.length == pytest.approx(2.0)

    def test_no_self_bond_without_lattice(self):
        """Self-bonds are impossible without periodic boundaries."""
        species = ["X"]
        coords = np.array([[0.5, 5.0, 5.0]])
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=100.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=None)
        assert len(bonds) == 0

    def test_self_bond_not_found_when_distance_exceeds_threshold(self):
        """Self-bond not found when image distance exceeds max_length."""
        species = ["X"]
        lattice = np.diag([5.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        specs = [BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        assert len(bonds) == 0

    def test_self_bond_first_spec_wins(self):
        """First matching spec claims the self-bond."""
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        spec_a = BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.5, radius=0.2, colour="red")
        spec_b = BondSpec(species=("*", "*"), min_length=0.0,
                          max_length=2.5, radius=0.1, colour="blue")
        bonds = compute_bonds(species, coords, [spec_a, spec_b],
                              lattice=lattice)
        for b in bonds:
            assert b.spec is spec_a


class TestMultiImageBonds:
    """Tests for structures where the same atom pair bonds through multiple images.

    These are critical for atoms at special positions (cell corners, edges,
    faces) where the same neighbour appears at the same distance through
    more than one periodic image.
    """

    def test_nacl_rock_salt(self):
        """NaCl rock salt: every atom should have 6 nearest-neighbour bonds.

        In the conventional rock salt unit cell (a = 5.64 A), each Na has
        6 Cl neighbours at a/2 = 2.82 A and vice versa.  Several of these
        bonds cross periodic boundaries through different images.
        """
        a = 5.64
        lattice = np.diag([a, a, a])
        # Conventional NaCl unit cell: 4 Na + 4 Cl.
        species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
        coords = np.array([
            [0.0, 0.0, 0.0],      # Na at origin
            [a/2, a/2, 0.0],      # Na face centre
            [a/2, 0.0, a/2],      # Na face centre
            [0.0, a/2, a/2],      # Na face centre
            [a/2, 0.0, 0.0],      # Cl edge centre
            [0.0, a/2, 0.0],      # Cl edge centre
            [0.0, 0.0, a/2],      # Cl edge centre
            [a/2, a/2, a/2],      # Cl body centre
        ])
        specs = [BondSpec(species=("Na", "Cl"), min_length=0.0,
                          max_length=a/2 + 0.1, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)

        # Count bonds per atom.
        bond_count = [0] * 8
        for b in bonds:
            bond_count[b.index_a] += 1
            bond_count[b.index_b] += 1
        for i, count in enumerate(bond_count):
            assert count == 6, (
                f"Atom {i} ({species[i]}) has {count} bonds, expected 6"
            )
        # Total: 8 atoms * 6 bonds / 2 = 24 unique bonds.
        assert len(bonds) == 24

    def test_perovskite_octahedral(self):
        """Perovskite: B-site atom at cell corner should have 6 octahedral bonds.

        In a cubic perovskite ABO3 (e.g. BaTiO3), the B-site cation (Ti)
        sits at the cell corner (origin) and bonds to 6 oxygen atoms at
        a/2 along +/-x, +/-y, +/-z.  The +direction bonds are direct
        (image 0,0,0) while the -direction bonds go through the negative
        image, since the O atom is at the opposite face of the cell.
        """
        a = 4.0
        lattice = np.diag([a, a, a])
        species = ["Ba", "Ti", "O", "O", "O"]
        coords = np.array([
            [a/2, a/2, a/2],    # Ba at body centre
            [0.0, 0.0, 0.0],    # Ti at origin (corner)
            [a/2, 0.0, 0.0],    # O on face (+x from Ti)
            [0.0, a/2, 0.0],    # O on face (+y from Ti)
            [0.0, 0.0, a/2],    # O on face (+z from Ti)
        ])
        specs = [BondSpec(species=("Ti", "O"), min_length=0.0,
                          max_length=a/2 + 0.1, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)

        # Ti should have exactly 6 bonds (3 direct + 3 via negative images).
        ti_bonds = [b for b in bonds
                    if b.index_a == 1 or b.index_b == 1]
        assert len(ti_bonds) == 6, (
            f"Ti has {len(ti_bonds)} bonds, expected 6"
        )

        # Check that we get both direct and image bonds to each O.
        images_by_o = {}
        for b in ti_bonds:
            o_idx = b.index_b if b.index_a == 1 else b.index_a
            images_by_o.setdefault(o_idx, []).append(b.image)
        # Each O should appear twice: once direct, once through image.
        for o_idx, images in images_by_o.items():
            assert len(images) == 2, (
                f"O atom {o_idx} has {len(images)} bonds to Ti, expected 2"
            )

    def test_multi_image_bond_lengths_correct(self):
        """All multi-image bonds have the correct distance."""
        a = 4.0
        lattice = np.diag([a, a, a])
        species = ["A", "B"]
        # A at origin, B at (a/2, 0, 0) — bonds at a/2 via (0,0,0) and (-1,0,0).
        coords = np.array([
            [0.0, 0.0, 0.0],
            [a/2, 0.0, 0.0],
        ])
        specs = [BondSpec(species=("A", "B"), min_length=0.0,
                          max_length=a/2 + 0.1, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        # Should find bonds through multiple images at the same distance.
        for b in bonds:
            assert b.length == pytest.approx(a / 2)

    def test_image_vectors_reconstruct_multi_image_bonds(self):
        """Each image vector correctly reconstructs the bonded position."""
        a = 4.0
        lattice = np.diag([a, a, a])
        species = ["Ti", "O", "O", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [a/2, 0.0, 0.0],
            [0.0, a/2, 0.0],
            [0.0, 0.0, a/2],
        ])
        specs = [BondSpec(species=("Ti", "O"), min_length=0.0,
                          max_length=a/2 + 0.1, radius=0.1, colour=1.0)]
        bonds = compute_bonds(species, coords, specs, lattice=lattice)
        for b in bonds:
            pos_b = coords[b.index_b] + np.array(b.image) @ lattice
            dist = np.linalg.norm(pos_b - coords[b.index_a])
            assert dist == pytest.approx(b.length, rel=1e-10)
