"""Tests for hofmann.construction.rendering_set — rendering set construction."""

import numpy as np
import pytest

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.rendering_set import RenderingSet, build_rendering_set
from hofmann.model import Bond, BondSpec


class TestNonPeriodicPassthrough:
    """When all bonds are direct, the rendering set is a passthrough."""

    def test_no_periodic_bonds(self):
        """All-direct bonds produce output identical to input."""
        species = ["A", "B"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lattice = np.diag([10.0, 10.0, 10.0])
        spec = BondSpec(species=("A", "B"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        assert rset.species == species
        np.testing.assert_array_equal(rset.coords, coords)
        assert len(rset.bonds) == 1
        np.testing.assert_array_equal(rset.source_indices, [0, 1])

    def test_empty_bonds(self):
        """No bonds at all — passthrough."""
        species = ["A"]
        coords = np.array([[0.0, 0.0, 0.0]])
        lattice = np.diag([10.0, 10.0, 10.0])
        rset = build_rendering_set(species, coords, [], [], lattice)
        assert rset.species == species
        assert len(rset.bonds) == 0
        np.testing.assert_array_equal(rset.source_indices, [0])


class TestCompleteExpansion:
    """Single-pass completion via the `complete` parameter."""

    def _two_atom_scene(self):
        """Two atoms on opposite sides of a 5 A cubic cell."""
        species = ["Na", "Cl"]
        coords = np.array([
            [0.5, 2.5, 2.5],  # Na near left face
            [4.5, 2.5, 2.5],  # Cl near right face
        ])
        lattice = np.diag([5.0, 5.0, 5.0])
        return species, coords, lattice

    def test_complete_false_no_expansion(self):
        """complete=False: periodic bonds not in rendering set."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        # Verify a periodic bond was found
        assert any(b.image != (0, 0, 0) for b in bonds)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # No image atoms, no periodic bonds in output
        assert rset.species == species
        assert len(rset.bonds) == 0

    def test_complete_star_both_sides(self):
        """complete='*' materialises image atoms on both sides."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # 2 physical + 2 image atoms (Cl image near Na, Na image near Cl)
        assert len(rset.species) == 4
        assert rset.coords.shape == (4, 3)

    def test_complete_directed(self):
        """complete='Na' only completes Na's coordination shell."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="Na")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Only Na's shell completed: Cl image materialised near Na
        # Na's image NOT materialised (Cl not in complete filter)
        assert len(rset.species) == 3
        # The image atom should be Cl
        assert rset.species[2] == "Cl"

    def test_image_atom_position(self):
        """Image atom coordinates are correct."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="Na")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Image of Cl (index 1) shifted by image vector
        periodic_bond = [b for b in bonds if b.image != (0, 0, 0)][0]
        expected_pos = coords[1] + np.array(periodic_bond.image) @ lattice
        np.testing.assert_allclose(rset.coords[2], expected_pos)

    def test_source_indices(self):
        """source_indices maps image atoms back to physical atoms."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Physical atoms map to themselves
        assert rset.source_indices[0] == 0
        assert rset.source_indices[1] == 1
        # Image atoms map back to their physical originals
        for i in range(2, len(rset.species)):
            src = rset.source_indices[i]
            assert rset.species[i] == species[src]

    def test_rendering_bonds_have_zero_image(self):
        """All bonds in the rendering set have image (0, 0, 0)."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        for b in rset.bonds:
            assert b.image == (0, 0, 0)

    def test_rendering_bond_indices_valid(self):
        """Bond indices reference valid positions in the expanded arrays."""
        species, coords, lattice = self._two_atom_scene()
        spec = BondSpec(species=("Cl", "Na"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        n = len(rset.species)
        for b in rset.bonds:
            assert 0 <= b.index_a < n
            assert 0 <= b.index_b < n


class TestDeduplication:
    """Image atoms requested by multiple bonds are created once."""

    def test_shared_image_atom(self):
        """Two periodic bonds requiring the same image atom → one copy."""
        # Three atoms: A at origin, B and C near the far face.
        # Both B and C bond to A through the boundary, requiring
        # an image of A near B and C.
        species = ["A", "B", "C"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [4.5, 2.5, 2.5],   # A near right face
            [0.5, 2.0, 2.5],   # B near left face
            [0.5, 3.0, 2.5],   # C near left face
        ])
        spec_ab = BondSpec(species=("A", "B"), min_length=0.0,
                           max_length=2.0, radius=0.1, colour=1.0,
                           complete="*")
        spec_ac = BondSpec(species=("A", "C"), min_length=0.0,
                           max_length=2.0, radius=0.1, colour=1.0,
                           complete="*")
        bonds = compute_bonds(species, coords, [spec_ab, spec_ac],
                              lattice=lattice)
        rset = build_rendering_set(species, coords, bonds,
                                   [spec_ab, spec_ac], lattice)
        # Count image atoms of species A
        image_a_count = sum(
            1 for i in range(3, len(rset.species))
            if rset.species[i] == "A"
        )
        # A's image at shift (1,0,0) should appear only once
        assert image_a_count == 1


class TestRecursiveExpansion:
    """Recursive expansion propagates through image atoms."""

    def test_recursive_chain(self):
        """A chain A-B-A across boundaries expands recursively."""
        # A at left, B at right. Bond A-B crosses boundary.
        # With recursive, the image of A at B's side should also
        # have its bond to B's image derived, pulling in more atoms.
        species = ["A", "B"]
        lattice = np.diag([4.0, 10.0, 10.0])
        coords = np.array([
            [0.5, 5.0, 5.0],  # A near left face
            [3.5, 5.0, 5.0],  # B near right face
        ])
        spec = BondSpec(species=("A", "B"), min_length=0.0,
                        max_length=1.5, radius=0.1, colour=1.0,
                        recursive=True)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Recursive expansion should create image atoms beyond
        # the initial single-pass completion.
        assert len(rset.species) > 2

    def test_recursive_false_no_propagation(self):
        """Non-recursive specs don't propagate through image atoms."""
        species = ["A", "B"]
        lattice = np.diag([4.0, 10.0, 10.0])
        coords = np.array([
            [0.5, 5.0, 5.0],
            [3.5, 5.0, 5.0],
        ])
        spec = BondSpec(species=("A", "B"), min_length=0.0,
                        max_length=1.5, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Single-pass: only immediate image atoms, no propagation
        # 2 physical + 2 image (one for each side)
        assert len(rset.species) == 4


class TestSelfBondExpansion:
    """Self-bonds (atom bonding to its own image) create image atoms."""

    def test_self_bond_with_complete(self):
        """Self-bond creates image atoms when complete is set."""
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        spec = BondSpec(species=("X", "X"), min_length=0.0,
                        max_length=2.5, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        # Self-bonds at (+1,0,0) and (-1,0,0) should materialise
        # 2 image atoms
        assert len(rset.species) == 3  # 1 physical + 2 images


class TestMaxRecursiveDepth:
    """Recursive expansion respects the max_recursive_depth parameter."""

    def test_depth_limits_expansion(self):
        """Self-bond chain: depth=1 gives fewer atoms than depth=3."""
        # Single atom in a narrow cell with self-bonds.  Recursive
        # expansion creates an infinite chain; max_recursive_depth
        # controls how far it extends.
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        spec = BondSpec(species=("X", "X"), min_length=0.0,
                        max_length=2.5, radius=0.1, colour=1.0,
                        recursive=True)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset_1 = build_rendering_set(
            species, coords, bonds, [spec], lattice,
            max_recursive_depth=1,
        )
        rset_3 = build_rendering_set(
            species, coords, bonds, [spec], lattice,
            max_recursive_depth=3,
        )
        # depth=1: 1 physical + 2 images = 3
        assert len(rset_1.species) == 3
        # depth=3: 1 physical + 6 images = 7
        assert len(rset_3.species) == 7
        assert len(rset_3.species) > len(rset_1.species)


class TestCompleteRecursiveInteraction:
    """Tests for interaction between complete and recursive flags."""

    def test_complete_skipped_when_recursive(self):
        """Specs with both complete='*' and recursive=True produce the
        same result as recursive=True alone — complete is redundant."""
        species = ["Na", "Cl"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 2.5, 2.5],   # Na near left face
            [4.5, 2.5, 2.5],   # Cl near right face
        ])
        spec_r = BondSpec(species=("Cl", "Na"), min_length=0.0,
                          max_length=2.0, radius=0.1, colour=1.0,
                          recursive=True)
        spec_cr = BondSpec(species=("Cl", "Na"), min_length=0.0,
                           max_length=2.0, radius=0.1, colour=1.0,
                           complete="*", recursive=True)
        bonds_r = compute_bonds(species, coords, [spec_r], lattice=lattice)
        rset_r = build_rendering_set(
            species, coords, bonds_r, [spec_r], lattice)
        bonds_cr = compute_bonds(species, coords, [spec_cr], lattice=lattice)
        rset_cr = build_rendering_set(
            species, coords, bonds_cr, [spec_cr], lattice)
        assert len(rset_r.species) == len(rset_cr.species)

    def test_complete_vs_recursive(self):
        """Recursive expansion produces more atoms than single-pass complete."""
        # Single atom with self-bonds in a narrow cell: complete adds
        # 2 immediate images, recursive follows the chain further.
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        spec_c = BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.5, radius=0.1, colour=1.0,
                          complete="*")
        spec_r = BondSpec(species=("X", "X"), min_length=0.0,
                          max_length=2.5, radius=0.1, colour=1.0,
                          recursive=True)
        bonds_c = compute_bonds(species, coords, [spec_c], lattice=lattice)
        rset_c = build_rendering_set(
            species, coords, bonds_c, [spec_c], lattice)
        bonds_r = compute_bonds(species, coords, [spec_r], lattice=lattice)
        rset_r = build_rendering_set(
            species, coords, bonds_r, [spec_r], lattice)
        # complete: 1 + 2 = 3; recursive (depth 5): 1 + 10 = 11
        assert len(rset_c.species) == 3
        assert len(rset_r.species) > len(rset_c.species)


class TestMixedSpecs:
    """Tests for multiple specs with different complete settings."""

    def test_directed_complete_multiple_specs(self):
        """Each spec's complete setting is processed independently."""
        # Na near origin corner, Cl near far-x face, Br near far-y face.
        # Na-Cl periodic bond on x-axis; Na-Br periodic bond on y-axis.
        species = ["Na", "Cl", "Br"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.5, 0.5, 2.5],   # Na near origin
            [4.5, 0.5, 2.5],   # Cl near far-x face
            [0.5, 4.5, 2.5],   # Br near far-y face
        ])
        # Na-Cl: complete="Cl" → only completes Cl's shell (adds Na image)
        spec_na_cl = BondSpec(species=("Na", "Cl"), min_length=0.0,
                              max_length=2.0, radius=0.1, colour=1.0,
                              complete="Cl")
        # Na-Br: complete="*" → completes both shells
        spec_na_br = BondSpec(species=("Na", "Br"), min_length=0.0,
                              max_length=2.0, radius=0.1, colour=1.0,
                              complete="*")
        all_specs = [spec_na_cl, spec_na_br]
        bonds = compute_bonds(species, coords, all_specs, lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, all_specs, lattice)
        # Na-Cl with complete="Cl": Na image added near Cl.
        # Na-Br with complete="*": Na image added near Br AND Br image
        # added near Na.
        na_count = sum(1 for s in rset.species if s == "Na")
        assert na_count >= 3  # 1 physical + 2 images
        # Cl: no image (complete="Cl" only completes Cl's shell, not Na's)
        cl_count = sum(1 for s in rset.species if s == "Cl")
        assert cl_count == 1
        # Br: 1 image from complete="*"
        br_count = sum(1 for s in rset.species if s == "Br")
        assert br_count == 2
