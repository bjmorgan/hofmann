"""Tests for hofmann.construction.rendering_set — rendering set construction."""

import numpy as np
import pytest

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.rendering_set import (
    RenderingSet, build_rendering_set, deduplicate_molecules,
)
from hofmann.model import Bond, BondSpec, PolyhedronSpec


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


class TestPbcPadding:
    """Geometric cell-face expansion via pbc_padding."""

    def test_atom_near_low_face(self):
        """Atom near frac=0 face gets image at +1 shift."""
        species = ["A"]
        lattice = np.diag([10.0, 10.0, 10.0])
        # frac_x = 0.01, distance to low face = 0.1 A
        coords = np.array([[0.1, 5.0, 5.0]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        # Should have 1 physical + 1 image at shift (+1, 0, 0)
        assert len(rset.species) == 2
        expected_pos = coords[0] + lattice[0]
        np.testing.assert_allclose(rset.coords[1], expected_pos)

    def test_atom_near_high_face(self):
        """Atom near frac=1 face gets image at -1 shift."""
        species = ["A"]
        lattice = np.diag([10.0, 10.0, 10.0])
        # frac_x = 0.99, distance to high face = 0.1 A
        coords = np.array([[9.9, 5.0, 5.0]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        assert len(rset.species) == 2
        expected_pos = coords[0] - lattice[0]
        np.testing.assert_allclose(rset.coords[1], expected_pos)

    def test_corner_atom_seven_images(self):
        """Atom near a corner (3 faces) produces 7 images."""
        species = ["A"]
        lattice = np.diag([10.0, 10.0, 10.0])
        # Near (0, 0, 0) corner: frac ~ (0.01, 0.01, 0.01)
        coords = np.array([[0.1, 0.1, 0.1]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        # 2^3 - 1 = 7 images (all combinations of +1 on each axis)
        assert len(rset.species) == 8

    def test_padding_none_disabled(self):
        """pbc_padding=None disables geometric expansion."""
        species = ["A"]
        lattice = np.diag([10.0, 10.0, 10.0])
        coords = np.array([[0.1, 5.0, 5.0]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=None,
        )
        assert len(rset.species) == 1

    def test_dedup_with_completion(self):
        """Padding atom already created by completion is not duplicated."""
        species = ["Na", "Cl"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.1, 2.5, 2.5],   # Na near low-x face
            [4.9, 2.5, 2.5],   # Cl near high-x face
        ])
        spec = BondSpec(species=("Na", "Cl"), min_length=0.0,
                        max_length=1.0, radius=0.1, colour=1.0,
                        complete="*")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset_with_pad = build_rendering_set(
            species, coords, bonds, [spec], lattice, pbc_padding=0.5,
        )
        rset_without_pad = build_rendering_set(
            species, coords, bonds, [spec], lattice, pbc_padding=None,
        )
        # Completion already creates the same images that padding would.
        # With padding there might be extra images on other axes, but
        # the key point is no duplicates on the x-axis.
        assert len(rset_with_pad.species) >= len(rset_without_pad.species)
        # Check no duplicate (physical_index, shift) combos by verifying
        # source_indices + coords are unique
        for i in range(len(rset_with_pad.species)):
            for j in range(i + 1, len(rset_with_pad.species)):
                assert not np.allclose(
                    rset_with_pad.coords[i], rset_with_pad.coords[j],
                    atol=1e-6,
                )

    def test_bond_to_padding_atom(self):
        """Periodic bond connecting a padding atom to a physical atom."""
        species = ["Na", "Cl"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.1, 2.5, 2.5],   # Na near low-x face
            [4.9, 2.5, 2.5],   # Cl near high-x face
        ])
        # complete=False so completion doesn't create any images.
        spec = BondSpec(species=("Na", "Cl"), min_length=0.0,
                        max_length=1.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, [spec], lattice, pbc_padding=0.5,
        )
        # Padding creates images.  Na near low face → Na image at high face.
        # Cl near high face → Cl image at low face.
        # The periodic bond Na-Cl should appear between the padding images
        # and the physical atoms they are near.
        assert len(rset.bonds) > 0

    def test_source_indices_for_padding(self):
        """Source indices correctly map padding atoms to physical atoms."""
        species = ["A"]
        lattice = np.diag([10.0, 10.0, 10.0])
        coords = np.array([[0.1, 5.0, 5.0]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        # All source indices should map to atom 0
        np.testing.assert_array_equal(rset.source_indices,
                                      np.zeros(len(rset.species), dtype=int))

    def test_triclinic_cell(self):
        """Padding works with non-orthogonal lattice vectors."""
        # Triclinic cell: the perpendicular height along axis 0 differs
        # from the length of vector a.
        lattice = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 10.0],
        ])
        species = ["A"]
        # Place atom at frac (0.5, 0.5, 0.5) — far from all faces.
        frac = np.array([0.5, 0.5, 0.5])
        coords = (frac @ lattice).reshape(1, 3)
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        # Well inside the cell, no images needed.
        assert len(rset.species) == 1


class TestPolyhedraVertexCompletion:
    """Polyhedra vertex completion materialises missing vertex atoms."""

    def test_physical_centre_gets_vertices(self):
        """Physical polyhedron centre with periodic bonds gets all vertices."""
        # Ti at centre, 4 O around it. Two O are across cell boundary.
        # Bond spec has complete=False, so without polyhedra vertex
        # completion those two O images wouldn't be materialised.
        species = ["Ti", "O", "O", "O", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [2.5, 2.5, 2.5],  # Ti at cell centre
            [2.5, 2.5, 0.5],  # O inside cell
            [2.5, 0.5, 2.5],  # O inside cell
            [2.5, 2.5, 4.9],  # O near high-z face
            [2.5, 4.9, 2.5],  # O near high-y face
        ])
        bond_spec = BondSpec(species=("Ti", "O"), min_length=0.0,
                             max_length=3.0, radius=0.1, colour=1.0)
        poly_spec = PolyhedronSpec(centre="Ti")
        bonds = compute_bonds(species, coords, [bond_spec], lattice=lattice)

        # Without polyhedra specs: periodic O not materialised.
        rset_no_poly = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
        )
        # With polyhedra specs: periodic O vertices materialised.
        rset_poly = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
            polyhedra_specs=[poly_spec],
        )
        assert len(rset_poly.species) >= len(rset_no_poly.species)
        # Ti should have bonds to all its neighbours in the expanded set.
        ti_bonds = [b for b in rset_poly.bonds
                    if b.index_a == 0 or b.index_b == 0]
        # At least 4 bonds (the direct ones plus any periodic ones)
        assert len(ti_bonds) >= 4

    def test_image_centre_gets_vertices(self):
        """Image atom that is a polyhedron centre also gets its vertices."""
        # Ti near low-x face, O1 bonded directly, O2 across boundary.
        # complete="O" creates a Ti image at the high-x side.
        # That Ti image needs O1 as a vertex (via vertex completion)
        # since completion only materialised the Ti image, not O1.
        species = ["Ti", "O", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.1, 2.5, 2.5],  # Ti near low-x face
            [1.5, 2.5, 2.5],  # O1 direct bond to Ti
            [4.5, 2.5, 2.5],  # O2 periodic bond to Ti across -x
        ])
        bond_spec = BondSpec(species=("Ti", "O"), min_length=0.0,
                             max_length=2.0, radius=0.1, colour=1.0,
                             complete="O")
        poly_spec = PolyhedronSpec(centre="Ti")
        bonds = compute_bonds(species, coords, [bond_spec], lattice=lattice)

        # Without polyhedra: Ti image exists but missing O1 vertex.
        rset_no_poly = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
        )
        # With polyhedra: O1 vertex materialised for Ti image.
        rset_poly = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
            polyhedra_specs=[poly_spec],
        )
        # Vertex completion should add at least one more atom.
        assert len(rset_poly.species) > len(rset_no_poly.species)
        # The Ti image should have bonds to both O atoms.
        ti_image_indices = [
            i for i in range(3, len(rset_poly.species))
            if rset_poly.species[i] == "Ti"
        ]
        assert len(ti_image_indices) >= 1
        ti_img = ti_image_indices[0]
        ti_img_bonds = [b for b in rset_poly.bonds
                        if b.index_a == ti_img or b.index_b == ti_img]
        assert len(ti_img_bonds) >= 2

    def test_vertex_dedup_with_completion(self):
        """Vertex atom already from completion is not duplicated."""
        species = ["Ti", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [2.5, 2.5, 2.5],
            [2.5, 2.5, 4.9],   # O near high-z face
        ])
        bond_spec = BondSpec(species=("Ti", "O"), min_length=0.0,
                             max_length=3.0, radius=0.1, colour=1.0,
                             complete="Ti")
        poly_spec = PolyhedronSpec(centre="Ti")
        bonds = compute_bonds(species, coords, [bond_spec], lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
            polyhedra_specs=[poly_spec],
        )
        # No duplicate: same image atom from completion and vertex completion.
        coord_set = set()
        for c in rset.coords:
            key = tuple(np.round(c, 6))
            assert key not in coord_set, f"Duplicate coordinate: {key}"
            coord_set.add(key)

    def test_no_polyhedra_specs_no_completion(self):
        """Without polyhedra specs, no vertex completion occurs."""
        species = ["Ti", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [2.5, 2.5, 2.5],
            [2.5, 2.5, 4.9],
        ])
        bond_spec = BondSpec(species=("Ti", "O"), min_length=0.0,
                             max_length=3.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [bond_spec], lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
        )
        # No completion, no padding: only physical atoms.
        assert len(rset.species) == 2


class TestDeduplicateMolecules:
    """Molecule deduplication removes duplicate connected components."""

    def test_molecule_spanning_cell_edge(self):
        """A-B-C where B-C spans the cell: orphan C removed."""
        # A near B; B near left face; C near right face.
        # B-C has a periodic bond across -x (distance 1.0 A).
        # With complete="B", C' image appears near B.
        # After dedup: A-B-C' kept, physical C removed.
        species = ["A", "B", "C"]
        lattice = np.diag([5.0, 10.0, 10.0])
        coords = np.array([
            [1.0, 5.0, 5.0],  # A
            [0.5, 5.0, 5.0],  # B near left face
            [4.5, 5.0, 5.0],  # C near right face (periodic dist to B = 1.0)
        ])
        spec_ab = BondSpec(species=("A", "B"), min_length=0.0,
                           max_length=1.5, radius=0.1, colour=1.0)
        spec_bc = BondSpec(species=("B", "C"), min_length=0.0,
                           max_length=1.5, radius=0.1, colour=1.0,
                           complete="B")
        all_specs = [spec_ab, spec_bc]
        bonds = compute_bonds(species, coords, all_specs, lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, all_specs, lattice,
        )
        # Before dedup: 3 physical + C' image = 4 atoms.
        # C is orphaned (its bond to B is via image, not direct).
        deduped = deduplicate_molecules(rset, lattice)
        # A, B, and C' should survive; physical C removed.
        assert len(deduped.species) == 3
        assert "A" in deduped.species
        assert "B" in deduped.species
        assert "C" in deduped.species
        # 2 bonds: A-B and B-C'
        assert len(deduped.bonds) == 2

    def test_isolated_atoms_physical_wins(self):
        """Physical isolated atom kept over image isolated atom."""
        species = ["A"]
        lattice = np.diag([5.0, 10.0, 10.0])
        coords = np.array([[0.1, 5.0, 5.0]])
        # No bonds, but padding creates an image.
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        assert len(rset.species) == 2  # physical + 1 image
        deduped = deduplicate_molecules(rset, lattice)
        # Physical atom kept, image removed.
        assert len(deduped.species) == 1
        # Source index should be 0 (physical atom).
        assert deduped.source_indices[0] == 0

    def test_extended_structure_no_change(self):
        """Extended structure (one giant component) is unchanged."""
        species = ["X"]
        lattice = np.diag([2.0, 10.0, 10.0])
        coords = np.array([[0.5, 5.0, 5.0]])
        spec = BondSpec(species=("X", "X"), min_length=0.0,
                        max_length=2.5, radius=0.1, colour=1.0,
                        recursive=True)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, [spec], lattice,
            max_recursive_depth=2,
        )
        n_before = len(rset.species)
        deduped = deduplicate_molecules(rset, lattice)
        # One big connected component — nothing removed.
        assert len(deduped.species) == n_before

    def test_source_indices_remapped(self):
        """Source indices are correct after deduplication."""
        species = ["A"]
        lattice = np.diag([5.0, 10.0, 10.0])
        coords = np.array([[0.1, 5.0, 5.0]])
        rset = build_rendering_set(
            species, coords, [], [], lattice, pbc_padding=0.5,
        )
        deduped = deduplicate_molecules(rset, lattice)
        for i, sp in enumerate(deduped.species):
            src = deduped.source_indices[i]
            assert species[src] == sp

    def test_bond_indices_valid_after_dedup(self):
        """Bond indices reference valid positions after deduplication."""
        species = ["A", "B", "C"]
        lattice = np.diag([5.0, 10.0, 10.0])
        coords = np.array([
            [1.0, 5.0, 5.0],
            [0.5, 5.0, 5.0],
            [4.5, 5.0, 5.0],
        ])
        spec_ab = BondSpec(species=("A", "B"), min_length=0.0,
                           max_length=1.5, radius=0.1, colour=1.0)
        spec_bc = BondSpec(species=("B", "C"), min_length=0.0,
                           max_length=1.5, radius=0.1, colour=1.0,
                           complete="B")
        all_specs = [spec_ab, spec_bc]
        bonds = compute_bonds(species, coords, all_specs, lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, all_specs, lattice,
        )
        deduped = deduplicate_molecules(rset, lattice)
        n = len(deduped.species)
        for b in deduped.bonds:
            assert 0 <= b.index_a < n
            assert 0 <= b.index_b < n
