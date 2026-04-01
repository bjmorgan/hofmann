"""Tests for individual pipeline stages in the bond/atom materialisation pipeline."""

import numpy as np
import pytest

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.rendering_set import build_rendering_set
from hofmann.model import BondSpec, PolyhedronSpec


class TestDirectBonds:
    """Step 2: Direct bonds between physical atoms within the cell."""

    def test_direct_bonds_present(self):
        """Atoms well inside cell with direct bonds — all bonds present, no images."""
        species = ["A", "B"]
        lattice = np.diag([10.0, 10.0, 10.0])
        coords = np.array([
            [3.0, 5.0, 5.0],  # A well inside cell
            [4.0, 5.0, 5.0],  # B well inside cell, 1 A away
        ])
        spec = BondSpec(species=("A", "B"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(species, coords, bonds, [spec], lattice)
        assert len(rset.species) == 2  # No image atoms
        assert len(rset.bonds) == 1


class TestPhysicalAtomCompletion:
    """Step 5: Completion materialises missing periodic bond partners for physical atoms."""

    def test_physical_atom_completion(self):
        """Physical Mn with periodic O bond: completion adds the missing O image."""
        species = ["Mn", "O", "O", "O", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.3, 0.3, 2.5],  # Mn near x=0, y=0 corner
            [1.5, 0.3, 2.5],  # O1 direct bond in +x (dist 1.2)
            [0.3, 1.5, 2.5],  # O2 direct bond in +y (dist 1.2)
            [4.7, 0.3, 2.5],  # O3 periodic bond in -x (dist 0.6 via image)
            [0.3, 4.7, 2.5],  # O4 periodic bond in -y (dist 0.6 via image)
        ])
        spec_complete = BondSpec(species=("Mn", "O"), min_length=0.0,
                                 max_length=2.5, radius=0.1, colour=1.0,
                                 complete="Mn")
        spec_no = BondSpec(species=("Mn", "O"), min_length=0.0,
                           max_length=2.5, radius=0.1, colour=1.0)
        bonds_no = compute_bonds(species, coords, [spec_no], lattice=lattice)
        bonds_complete = compute_bonds(
            species, coords, [spec_complete], lattice=lattice
        )
        # Without completion: periodic bonds exist but no images materialised.
        rset_no_complete = build_rendering_set(
            species, coords, bonds_no, [spec_no], lattice,
        )
        # With completion: Mn gets its full coordination shell.
        rset_complete = build_rendering_set(
            species, coords, bonds_complete, [spec_complete], lattice,
        )
        mn_bonds_no = [b for b in rset_no_complete.bonds
                       if b.index_a == 0 or b.index_b == 0]
        mn_bonds_yes = [b for b in rset_complete.bonds
                        if b.index_a == 0 or b.index_b == 0]
        assert len(mn_bonds_no) == 2  # Only direct bonds
        assert len(mn_bonds_yes) == 4  # All bonds (direct + completed)


class TestPaddingBondDiscovery:
    """Step 4: Padding atoms get bonds to already-existing base atoms."""

    def test_padding_atom_bonds_to_physical(self):
        """Padding image bonds to a physical atom that is already in the cell."""
        species = ["Na", "Cl"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [0.1, 2.5, 2.5],  # Na near low-x face
            [4.9, 2.5, 2.5],  # Cl near high-x face
        ])
        # complete=False: no completion images.
        spec = BondSpec(species=("Na", "Cl"), min_length=0.0,
                        max_length=1.0, radius=0.1, colour=1.0)
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        # With padding: Na image appears at high-x, Cl image at low-x.
        # Padding bond discovery should connect Na image to physical Cl
        # and Cl image to physical Na.
        rset = build_rendering_set(
            species, coords, bonds, [spec], lattice, pbc_padding=0.5,
        )
        assert len(rset.bonds) >= 2  # At least Na_img-Cl and Cl_img-Na


class TestPaddingAtomCompletion:
    """Step 5 applied to padding atoms: completion materialises missing
    partners for padding atoms whose species matches a complete filter."""

    def test_padding_atom_gets_completed(self):
        """Mn image from padding needs O neighbours materialised (issue #38)."""
        # Mn near cell face with O neighbours. Padding creates Mn image.
        # The Mn image should get its full coordination shell via completion.
        species = ["Mn", "O", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [4.9, 2.5, 2.5],  # Mn near high-x face
            [3.5, 2.5, 2.5],  # O1 direct bond to Mn
            [0.5, 2.5, 2.5],  # O2 periodic bond to Mn across +x
        ])
        spec = BondSpec(species=("Mn", "O"), min_length=0.0,
                        max_length=2.0, radius=0.1, colour=1.0,
                        complete="Mn")
        bonds = compute_bonds(species, coords, [spec], lattice=lattice)
        rset = build_rendering_set(
            species, coords, bonds, [spec], lattice, pbc_padding=0.5,
        )
        # Find the Mn image atom (padding creates it at shift (-1,0,0)).
        mn_images = [
            i for i in range(3, len(rset.species))
            if rset.species[i] == "Mn"
        ]
        assert len(mn_images) >= 1, "Padding should create Mn image"
        mn_img = mn_images[0]
        # The Mn image should have bonds to BOTH O atoms
        # (one via O1's image, one via physical O2 or its image).
        mn_img_bonds = [b for b in rset.bonds
                        if b.index_a == mn_img or b.index_b == mn_img]
        assert len(mn_img_bonds) >= 2, (
            f"Mn image should have >= 2 bonds but has {len(mn_img_bonds)}"
        )


class TestRecursiveChain:
    """Step 6: Recursive expansion follows molecular chains across boundaries."""

    def test_chain_completed_by_recursion(self):
        """A-B-C chain repeating across cell boundary: completion adds only
        one shell of images; recursion follows the chain through multiple cells."""
        # Three atoms per cell, cell size 3 Angstroms along x.
        # A-B bond (1.1 A) and B-C bond (1.1 A) are direct.
        # C-A bond (0.8 A) is periodic (C near right face, next A across boundary).
        # complete='*' materialises one shell of images for physical atoms.
        # recursive=True iterates, following the chain further.
        species = ["A", "B", "C"]
        lattice = np.diag([3.0, 10.0, 10.0])
        coords = np.array([
            [0.4, 5.0, 5.0],   # A near left face
            [1.5, 5.0, 5.0],   # B at cell centre
            [2.6, 5.0, 5.0],   # C near right face (periodic bond to next A)
        ])
        # With complete only (not recursive): single shell, chain truncated.
        specs_c = [
            BondSpec(species=("A", "B"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, complete="*"),
            BondSpec(species=("B", "C"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, complete="*"),
            BondSpec(species=("C", "A"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, complete="*"),
        ]
        bonds_c = compute_bonds(species, coords, specs_c, lattice=lattice)
        rset_c = build_rendering_set(
            species, coords, bonds_c, specs_c, lattice,
        )
        # With recursive: chain extends across multiple cells.
        specs_r = [
            BondSpec(species=("A", "B"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, recursive=True),
            BondSpec(species=("B", "C"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, recursive=True),
            BondSpec(species=("C", "A"), min_length=0.0,
                     max_length=1.2, radius=0.1, colour=1.0, recursive=True),
        ]
        bonds_r = compute_bonds(species, coords, specs_r, lattice=lattice)
        rset_r = build_rendering_set(
            species, coords, bonds_r, specs_r, lattice,
        )
        # Recursive should produce more atoms than completion alone.
        assert len(rset_r.species) > len(rset_c.species)


class TestPolyhedraVertexCompletion:
    """Step 7: Polyhedra vertex completion materialises missing vertices
    and creates centre-vertex bonds."""

    def test_missing_vertex_materialised(self):
        """Ti centre with a periodic O bond: vertex completion adds the O image."""
        species = ["Ti", "O", "O"]
        lattice = np.diag([5.0, 5.0, 5.0])
        coords = np.array([
            [2.5, 2.5, 2.5],  # Ti at cell centre
            [2.5, 2.5, 0.5],  # O1 inside cell (direct bond)
            [2.5, 2.5, 4.9],  # O2 near high-z face (periodic bond)
        ])
        # complete=False: no completion images.
        bond_spec = BondSpec(species=("Ti", "O"), min_length=0.0,
                             max_length=3.0, radius=0.1, colour=1.0)
        poly_spec = PolyhedronSpec(centre="Ti")
        bonds = compute_bonds(species, coords, [bond_spec], lattice=lattice)
        # Without polyhedra: periodic O not materialised.
        rset_no = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
        )
        # With polyhedra: O2 image materialised as vertex.
        rset_yes = build_rendering_set(
            species, coords, bonds, [bond_spec], lattice,
            polyhedra_specs=[poly_spec],
        )
        assert len(rset_yes.species) > len(rset_no.species)
        # Ti should have bonds to all O atoms.
        ti_bonds = [b for b in rset_yes.bonds
                    if b.index_a == 0 or b.index_b == 0]
        assert len(ti_bonds) >= 2
