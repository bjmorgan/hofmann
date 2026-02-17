"""Tests for hofmann.scene — convenience constructors."""

import numpy as np
import pytest

from hofmann.model import StructureScene
from hofmann.scene import _merge_expansions, from_xbs, from_pymatgen

_has_pymatgen = False
try:
    from pymatgen.core import Lattice, Structure

    _has_pymatgen = True
except ImportError:
    pass


class TestMergeExpansions:
    """Tests for _merge_expansions deduplication."""

    def test_deduplicates_within_extras(self):
        """Two extra atoms at the same position should not both be added."""
        base_species = ["Si"]
        base_coords = np.array([[0.0, 0.0, 0.0]])
        extra_species = ["O", "O"]
        extra_coords = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        merged_sp, merged_co = _merge_expansions(
            base_species, base_coords, extra_species, extra_coords,
        )
        assert len(merged_sp) == 2  # Si + one O, not Si + two O
        assert merged_co.shape[0] == 2


class TestFromXbs:
    def test_ch4_basic(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        assert len(scene.species) == 5
        assert len(scene.frames) == 1
        assert len(scene.atom_styles) == 2
        assert len(scene.bond_specs) == 2
        assert scene.title == "ch4"

    def test_ch4_with_trajectory(self, ch4_bs_path, ch4_mv_path):
        scene = from_xbs(ch4_bs_path, mv_path=ch4_mv_path)
        assert len(scene.frames) == 2
        assert len(scene.species) == 5

    def test_view_centred(self, ch4_bs_path):
        scene = from_xbs(ch4_bs_path)
        expected_centroid = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(scene.view.centre, expected_centroid)

    def test_classmethod(self, ch4_bs_path):
        scene = StructureScene.from_xbs(ch4_bs_path)
        assert isinstance(scene, StructureScene)
        assert len(scene.species) == 5

    def test_no_lattice(self, ch4_bs_path):
        """XBS scenes have no lattice information."""
        scene = from_xbs(ch4_bs_path)
        assert scene.lattice is None


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgen:
    def test_single_structure(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        assert len(scene.species) == 2
        assert len(scene.frames) == 1
        assert "Na" in scene.atom_styles
        assert "Cl" in scene.atom_styles

    def test_trajectory(self):
        lattice = Lattice.cubic(5.0)
        s1 = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        s2 = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.26, 0.26, 0.26]])
        scene = from_pymatgen([s1, s2], pbc=False)
        assert len(scene.frames) == 2

    def test_classmethod(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = StructureScene.from_pymatgen(struct, pbc=False)
        assert isinstance(scene, StructureScene)

    def test_view_centred(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        expected = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(scene.view.centre, expected)

    def test_lattice_stored(self):
        """from_pymatgen stores the lattice matrix."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        assert scene.lattice is not None
        np.testing.assert_allclose(scene.lattice, lattice.matrix)

    def test_lattice_is_copy(self):
        """Modifying scene.lattice does not affect the original structure."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        scene.lattice[0, 0] = 999.0
        np.testing.assert_allclose(struct.lattice.matrix[0, 0], 5.0)

    def test_pbc_on_by_default(self):
        """PBC expansion is enabled by default for pymatgen structures."""
        lattice = Lattice.cubic(10.0)
        # Na at origin gets image atoms from the default pbc_padding=0.1.
        struct = Structure(lattice, ["Na"], [[0.0, 0.5, 0.5]])
        scene = from_pymatgen(struct, bond_specs=[])
        # Na at frac 0.0 is within 0.1 A of the face, so gets a +1 image.
        assert len(scene.species) == 2


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgenPbc:
    def test_atom_near_origin_gets_positive_image(self):
        """An atom near frac=0 gets an image shifted by +1."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        # Atom 0 near origin face -> +1 image.  Atom 1 at centre -> none.
        assert len(scene.species) == 3

    def test_atom_near_far_face_gets_negative_image(self):
        """An atom near frac=1 gets an image shifted by -1."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.99, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        # Near far face on x -> -1 image.
        assert len(scene.species) == 2
        xs = sorted(scene.frames[0].coords[:, 0])
        np.testing.assert_allclose(xs[0], -0.1, atol=1e-10)
        np.testing.assert_allclose(xs[1], 9.9, atol=1e-10)

    def test_centre_atom_no_images(self):
        """An atom far from all faces should get no images."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        assert len(scene.species) == 1

    def test_corner_atom_near_origin(self):
        """An atom near the origin corner gets 7 images (+1 on each axis combo)."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.01, 0.01, 0.01]])
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        # 3 faces + 3 edges + 1 corner = 7 images.
        assert len(scene.species) == 8

    def test_corner_atom_near_far_corner(self):
        """An atom near the far corner gets 7 images (-1 on each axis combo)."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.99, 0.99, 0.99]])
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        assert len(scene.species) == 8

    def test_pbc_false_no_expansion(self):
        """With pbc=False, no images are added."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.01, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        assert len(scene.species) == 1

    def test_image_coords_positive_shift(self):
        """A +1 image should be one lattice vector beyond the far face."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.02, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        coords = scene.frames[0].coords
        xs = sorted(coords[:, 0])
        np.testing.assert_allclose(xs[0], 0.2, atol=1e-10)
        np.testing.assert_allclose(xs[1], 10.2, atol=1e-10)

    def test_pbc_bonds_form_across_boundary(self):
        """Image atoms should enable bonds that cross the boundary."""
        from hofmann.bonds import compute_bonds
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]],
        )
        na_bond = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=3.72, radius=0.1, colour=0.5,
        )
        # Use pbc_padding=0.2 so the geometric expansion captures atoms
        # within 0.2 A of the cell face (frac_cutoff = 0.2/10 = 0.02).
        scene = from_pymatgen(
            struct, bond_specs=[na_bond], pbc=True, pbc_padding=0.2,
        )
        bonds = compute_bonds(
            scene.species,
            scene.frames[0].coords,
            scene.bond_specs,
        )
        # Atom 0 image at 10.1 bonds with atom 1 at 9.9 (0.2 A).
        # Atom 1 image at -0.1 bonds with atom 0 at 0.1 (0.2 A).
        assert len(bonds) > 0

    def test_wrapping_invariance(self):
        """Result should be the same regardless of fractional coordinate wrapping."""
        lattice = Lattice.cubic(10.0)
        # Atom at frac -0.01 is equivalent to frac 0.99.
        struct_neg = Structure(lattice, ["Na"], [[-.01, 0.5, 0.5]])
        struct_pos = Structure(lattice, ["Na"], [[0.99, 0.5, 0.5]])
        scene_neg = from_pymatgen(struct_neg, pbc=True, pbc_padding=1.0)
        scene_pos = from_pymatgen(struct_pos, pbc=True, pbc_padding=1.0)
        assert len(scene_neg.species) == len(scene_pos.species)
        np.testing.assert_allclose(
            np.sort(scene_neg.frames[0].coords, axis=0),
            np.sort(scene_pos.frames[0].coords, axis=0),
            atol=1e-10,
        )

    def test_expansion_preserves_centroid(self):
        """Expansion should be symmetric and preserve the cell centroid."""
        lattice = Lattice.cubic(10.0)
        # Two atoms equidistant from opposite faces.
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.02, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        scene = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        orig_centroid = np.mean(struct.cart_coords, axis=0)
        expanded_centroid = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(expanded_centroid, orig_centroid, atol=1e-10)

    def test_custom_pbc_padding(self):
        """A custom pbc_padding should override the bond-derived default."""
        lattice = Lattice.cubic(10.0)
        # Atom at frac 0.05 (cart 0.5) -- within a 1.0 cutoff, but
        # outside a tight 0.3 cutoff.
        struct = Structure(lattice, ["Na"], [[0.05, 0.5, 0.5]])
        scene_wide = from_pymatgen(struct, pbc=True, pbc_padding=1.0)
        scene_tight = from_pymatgen(struct, pbc=True, pbc_padding=0.3)
        # Wide cutoff includes this atom; tight cutoff excludes it.
        assert len(scene_wide.species) == 2
        assert len(scene_tight.species) == 1


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestPbcPaddingExpansion:
    """Tests for geometric PBC expansion behaviour."""

    def test_tight_padding_excludes_distant_atoms(self):
        """With tight pbc_padding, atoms far from cell faces get no image."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # Na at frac 0.5 — centre of cell, far from any face.
        # Cl at frac 0.95 — 0.25 A from far face.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.95, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        # pbc_padding=0.1 A → frac_cutoff=0.02. Cl at frac 0.95
        # has 1-0.95=0.05 > 0.02, so geometric expansion does not
        # add an image.  Without polyhedra, no neighbour completion
        # runs either.
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True, pbc_padding=0.1,
        )
        assert len(scene.species) == 2  # No images added.

    def test_wider_padding_captures_boundary_atom(self):
        """Increasing pbc_padding captures atoms closer to the face."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.95, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        # pbc_padding=0.3 A → frac_cutoff=0.06. Cl at frac 0.95
        # has 1-0.95=0.05 < 0.06, so an image IS added.
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True, pbc_padding=0.3,
        )
        assert len(scene.species) >= 3  # At least one image.

    def test_no_bond_specs_no_extra_images(self):
        """With no bond specs, no extra images are added."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, bond_specs=[], pbc=True, pbc_padding=0.1)
        assert len(scene.species) == 1

    def test_images_stay_within_one_cell_distance(self):
        """Image atoms should be within one lattice vector of the cell."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(4.0)
        # Two atoms near opposite faces.
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=2.0, radius=0.1, colour=0.5,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True, pbc_padding=0.5,
        )
        coords = scene.frames[0].coords
        # All x coordinates should be within one lattice parameter of [0, a].
        assert np.all(coords[:, 0] > -4.5)
        assert np.all(coords[:, 0] < 8.5)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestNeighbourShellExpansion:
    """Tests for neighbour-shell completion of polyhedron centres."""

    def test_image_centre_gets_vertex_atoms(self):
        """An image atom matching a centre pattern gets its vertex atoms added."""
        from hofmann.model import BondSpec, PolyhedronSpec

        lattice = Lattice.cubic(5.0)
        # Ti at frac (0.02, 0.5, 0.5) — near origin face, will get an image.
        # O at frac (0.5, 0.5, 0.5) — centre of cell, bonds to Ti.
        struct = Structure(
            lattice, ["Ti", "O"],
            [[0.02, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        ti_o_bond = BondSpec(
            species=("Ti", "O"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        # Ti at frac 0.02 gets a +1 image at frac 1.02 (cart 5.1).
        # That image Ti needs its own O vertex atom at cart 5.1 + offset.
        scene = from_pymatgen(
            struct,
            bond_specs=[ti_o_bond],
            polyhedra=[PolyhedronSpec(centre="Ti")],
            pbc=True,
            pbc_padding=1.0,
        )
        # The image Ti should have vertex atoms added for it.
        # Count Ti atoms — should be at least 2 (original + image).
        ti_count = sum(1 for sp in scene.species if sp == "Ti")
        assert ti_count >= 2
        # O atoms should include vertices for the image Ti.
        o_count = sum(1 for sp in scene.species if sp == "O")
        assert o_count >= 2

    def test_no_polyhedra_specs_no_vertex_expansion(self):
        """Without polyhedra specs, no vertex expansion occurs."""
        from hofmann.model import BondSpec, PolyhedronSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Ti", "O"],
            [[0.02, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        ti_o_bond = BondSpec(
            species=("Ti", "O"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        scene_no_poly = from_pymatgen(
            struct, bond_specs=[ti_o_bond], pbc=True, pbc_padding=1.0,
        )
        scene_with_poly = from_pymatgen(
            struct, bond_specs=[ti_o_bond],
            polyhedra=[PolyhedronSpec(centre="Ti")],
            pbc=True, pbc_padding=1.0,
        )
        # With polyhedra, more atoms are added for vertex expansion.
        assert len(scene_with_poly.species) >= len(scene_no_poly.species)

    def test_vertex_expansion_not_recursive(self):
        """Newly added vertex atoms should not themselves get expanded."""
        from hofmann.model import BondSpec, PolyhedronSpec

        lattice = Lattice.cubic(4.0)
        # Ti near origin face, O in centre.
        struct = Structure(
            lattice, ["Ti", "O"],
            [[0.02, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Ti", "O"), min_length=0.0,
            max_length=2.5, radius=0.1, colour=0.5,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond],
            polyhedra=[PolyhedronSpec(centre="Ti")],
            pbc=True, pbc_padding=1.0,
        )
        coords = scene.frames[0].coords
        # All coordinates should be within a reasonable range of the cell,
        # not extending to 2+ lattice vectors away.
        assert np.all(coords[:, 0] > -3.0)
        assert np.all(coords[:, 0] < 7.0)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestCentreAtom:
    """Tests for centre_atom PBC rewrapping."""

    def test_centre_atom_shifts_frac_coords(self):
        """centre_atom shifts the chosen atom to frac (0.5, 0.5, 0.5)."""
        lattice = Lattice.cubic(10.0)
        # Na at corner, Cl in centre.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        scene = from_pymatgen(struct, bond_specs=[], centre_atom=0)
        # Na (index 0) should now be at frac (0.5, 0.5, 0.5) = cart (5, 5, 5).
        np.testing.assert_allclose(
            scene.frames[0].coords[0], [5.0, 5.0, 5.0], atol=1e-10,
        )

    def test_view_centres_on_atom(self):
        """The view should centre on the chosen atom's position."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]],
        )
        scene = from_pymatgen(struct, bond_specs=[], centre_atom=0)
        np.testing.assert_allclose(
            scene.view.centre, scene.frames[0].coords[0], atol=1e-10,
        )

    def test_default_centre_atom_is_centroid(self):
        """Without centre_atom, view centres on the centroid."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        scene = from_pymatgen(struct, bond_specs=[], pbc=False)
        centroid = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(scene.view.centre, centroid, atol=1e-10)


class TestFromPymatgenImportError:
    def test_import_error_when_missing(self, monkeypatch):
        """Verify a helpful ImportError when pymatgen is absent."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymatgen.core":
                raise ImportError("No module named 'pymatgen'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pymatgen is required"):
            from_pymatgen(None)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestMultiFramePbc:
    def test_species_from_first_frame(self):
        """Multi-frame PBC expansion must use the first frame's species."""
        lattice = Lattice.cubic(10.0)
        # Both frames have the same structure — species list must match
        # the first frame regardless of how many frames there are.
        s1 = Structure(lattice, ["Si"], [[0.02, 0.5, 0.5]])
        s2 = Structure(lattice, ["Si"], [[0.02, 0.5, 0.5]])
        scene = from_pymatgen([s1, s2], pbc=True)
        assert len(scene.species) == len(scene.frames[0].coords)
        assert len(scene.species) == len(scene.frames[1].coords)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestRecursiveBondExpansion:
    """Tests for recursive bond search across periodic boundaries."""

    def test_recursive_adds_bonded_atom_across_pbc(self):
        """A dimer spanning the boundary is completed by recursive search."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # A at frac (0.5, 0.5, 0.5), B at frac (0.98, 0.5, 0.5).
        # B is near the far face; its bonded partner would be an image
        # of A at frac (1.5, 0.5, 0.5) — well beyond pbc_padding=0.1.
        # With recursive=True, the bond from B should pull in the A image.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond], pbc=True, pbc_padding=0.1,
        )
        # Without recursive, only the 2 UC atoms and possibly one
        # geometric image of B.  With recursive, the Na image at
        # x ~ 7.5 should appear as a bonded partner to the Cl at x=4.9.
        na_coords = [
            scene.frames[0].coords[i]
            for i, sp in enumerate(scene.species) if sp == "Na"
        ]
        # There should be at least 2 Na atoms (original + image).
        assert len(na_coords) >= 2

    def test_recursive_chain_follows_multiple_hops(self):
        """Multi-hop recursive expansion adds images beyond pbc_padding."""
        from hofmann.model import BondSpec

        # A-B dimer at one end of a large cell (20 A).  A is near the
        # origin face; B is ~2.5 A inside.  With pbc_padding=0.2 only
        # A gets a geometric image at x ~ 20.  Recursive iteration 1
        # should then add a B image bonded to that A image.
        lattice = Lattice.cubic(20.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.005, 0.5, 0.5], [0.125, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond], pbc=True, pbc_padding=0.2,
        )
        # Without recursion, only Na gets a geometric image.  With
        # recursion, a Cl image should appear bonded to the Na image.
        cl_count = sum(1 for sp in scene.species if sp == "Cl")
        assert cl_count >= 2

    def test_recursive_respects_max_depth(self):
        """max_recursive_depth=0 disables recursive expansion."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene_no_recurse = from_pymatgen(
            struct, bond_specs=[bond], pbc=True, pbc_padding=0.1,
            max_recursive_depth=0,
        )
        scene_with_recurse = from_pymatgen(
            struct, bond_specs=[bond], pbc=True, pbc_padding=0.1,
            max_recursive_depth=5,
        )
        # With depth=0, no recursive atoms are added.
        assert len(scene_with_recurse.species) > len(scene_no_recurse.species)

    def test_non_recursive_spec_unchanged(self):
        """A bond spec without recursive=True adds no extra atoms."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond_no_flag = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        bond_with_flag = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene_default = from_pymatgen(
            struct, bond_specs=[bond_no_flag], pbc=True, pbc_padding=0.1,
        )
        scene_recursive = from_pymatgen(
            struct, bond_specs=[bond_with_flag], pbc=True, pbc_padding=0.1,
        )
        # Recursive adds more atoms; default does not.
        assert len(scene_recursive.species) > len(scene_default.species)

    def test_recursive_with_polyhedra(self):
        """Recursive expansion provides atoms that polyhedra can use."""
        from hofmann.model import BondSpec, PolyhedronSpec

        lattice = Lattice.cubic(5.0)
        # Ti near far face, O in centre.  The recursive bond pulls
        # in an O image for the Ti image atom.
        struct = Structure(
            lattice, ["Ti", "O"],
            [[0.98, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Ti", "O"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond],
            polyhedra=[PolyhedronSpec(centre="Ti")],
            pbc=True, pbc_padding=0.1,
        )
        # The recursive expansion + polyhedra completion should give
        # more atoms than just geometric expansion alone.
        ti_count = sum(1 for sp in scene.species if sp == "Ti")
        o_count = sum(1 for sp in scene.species if sp == "O")
        assert ti_count >= 2
        assert o_count >= 2


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestCompleteBondExpansion:
    """Tests for single-pass bond completion across periodic boundaries."""

    def test_complete_adds_bonded_atom_across_pbc(self):
        """complete="*" adds the missing partner in a single pass."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # Na at centre, Cl near the far face.  The Cl's bonded Na image
        # is beyond pbc_padding=0.1, but complete="*" should add it.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*",
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond], pbc=True, pbc_padding=0.1,
        )
        na_coords = [
            scene.frames[0].coords[i]
            for i, sp in enumerate(scene.species) if sp == "Na"
        ]
        # The Na image at x ~ 7.5 should be added.
        assert len(na_coords) >= 2

    def test_complete_is_not_recursive(self):
        """complete="*" only does one pass — it does not follow chains."""
        from hofmann.model import BondSpec

        # 3-atom chain: A at 0.02, B at 0.35, C at 0.68 in a 10 A cell.
        # A is near the boundary; geometric expansion adds an image of A
        # at frac ~1.0.  B is bonded to A (dist ~3.3), C is bonded to B
        # (dist ~3.3), but C is NOT directly bonded to the A image.
        #
        # With complete="*" (single pass): C should NOT gain an image,
        # because C's bonded partner (an image of B) is not already
        # present.
        #
        # With recursive=True: the chain would be followed iteratively.
        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Na", "Na"],
            [[0.02, 0.5, 0.5], [0.35, 0.5, 0.5], [0.68, 0.5, 0.5]],
        )
        bond_complete = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=3.5, radius=0.1, colour=0.5,
            complete="*",
        )
        bond_recursive = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=3.5, radius=0.1, colour=0.5,
            recursive=True,
        )
        scene_complete = from_pymatgen(
            struct, bond_specs=[bond_complete], pbc=True, pbc_padding=0.1,
        )
        scene_recursive = from_pymatgen(
            struct, bond_specs=[bond_recursive], pbc=True, pbc_padding=0.1,
        )
        # Recursive should find more atoms than the single-pass complete.
        assert len(scene_recursive.species) > len(scene_complete.species)

    def test_complete_default_false(self):
        """complete defaults to False and adds no extra atoms."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond_default = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        bond_complete = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*",
        )
        scene_default = from_pymatgen(
            struct, bond_specs=[bond_default], pbc=True, pbc_padding=0.1,
        )
        scene_complete = from_pymatgen(
            struct, bond_specs=[bond_complete], pbc=True, pbc_padding=0.1,
        )
        assert len(scene_complete.species) > len(scene_default.species)

    def test_complete_directed_only_expands_named_species(self):
        """complete='Cl' only adds neighbours around visible Cl atoms."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # Na at centre, Cl near far face.  With complete="*" (both
        # directions), both Na images around Cl AND Cl images around Na
        # are added.  With complete="Cl", only Na images around the
        # visible Cl should be added — no new Cl atoms.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond_directed = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="Cl",
        )
        bond_both = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*",
        )
        scene_directed = from_pymatgen(
            struct, bond_specs=[bond_directed], pbc=True, pbc_padding=0.1,
        )
        scene_both = from_pymatgen(
            struct, bond_specs=[bond_both], pbc=True, pbc_padding=0.1,
        )
        # Directed should add fewer atoms than bidirectional.
        assert len(scene_both.species) > len(scene_directed.species)
        # Directed should still add the Na image for the Cl atom.
        na_count = sum(1 for sp in scene_directed.species if sp == "Na")
        assert na_count >= 2
        # But no new Cl images should be added.
        cl_count = sum(1 for sp in scene_directed.species if sp == "Cl")
        assert cl_count == 1

    def test_complete_mixed_selectors(self):
        """Mixing complete="Cl" and complete="*" respects each spec."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # Na at centre, Cl near far face, Br near far face on y.
        # Two bond specs: Na-Cl with complete="Cl" (only expands
        # around Cl — adding missing Na neighbours for Cl), and
        # Na-Br with complete="*" (expands both ways).
        #
        # Bug: if we merge both specs into a single expansion call
        # with centre_species_only=["Cl"], the "*" spec's expansion
        # around Br is suppressed.  Per-spec processing avoids this.
        struct = Structure(
            lattice, ["Na", "Cl", "Br"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5], [0.5, 0.98, 0.5]],
        )
        bond_na_cl = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="Cl",
        )
        bond_na_br = BondSpec(
            species=("Na", "Br"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*",
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond_na_cl, bond_na_br],
            pbc=True, pbc_padding=0.1,
        )
        # The "*" spec on Na-Br should complete around Br too,
        # adding a Na image as a neighbour of Br.
        na_count = sum(1 for sp in scene.species if sp == "Na")
        assert na_count >= 2
        # Br should also gain image atoms from the "*" expansion.
        br_count = sum(1 for sp in scene.species if sp == "Br")
        assert br_count >= 2

    def test_complete_plus_recursive_no_extra_depth(self):
        """complete + recursive on the same spec does not add an extra pass."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        # recursive-only should give the same result as complete+recursive,
        # because complete is skipped when recursive is set.
        bond_recursive = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            recursive=True,
        )
        bond_both = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*", recursive=True,
        )
        scene_recursive = from_pymatgen(
            struct, bond_specs=[bond_recursive], pbc=True, pbc_padding=0.1,
        )
        scene_both = from_pymatgen(
            struct, bond_specs=[bond_both], pbc=True, pbc_padding=0.1,
        )
        assert len(scene_both.species) == len(scene_recursive.species)

    def test_complete_with_unwrapped_frac_coords(self):
        """Bond completion is consistent regardless of frac coord wrapping.

        Regression: _expand_pbc wraps fractional coordinates to [0, 1),
        but _expand_neighbour_shells used raw structure[i].coords for
        unit-cell atoms, leading to neighbour positions computed in the
        wrong coordinate frame.

        This test verifies that structures with equivalent fractional
        coordinates (one in [0, 1) and one outside) produce the same
        expansion results.
        """
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(10.0)
        # Two equivalent structures: Na at frac -0.01 vs frac 0.99.
        # They are the same physical structure; expansion should give
        # identical results.
        struct_neg = Structure(
            lattice, ["Na", "Cl"],
            [[-0.01, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        struct_pos = Structure(
            lattice, ["Na", "Cl"],
            [[0.99, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=5.5, radius=0.1, colour=0.5,
            complete="*",
        )
        scene_neg = from_pymatgen(
            struct_neg, bond_specs=[bond], pbc=True, pbc_padding=1.5,
        )
        scene_pos = from_pymatgen(
            struct_pos, bond_specs=[bond], pbc=True, pbc_padding=1.5,
        )
        # Same number of atoms.
        assert len(scene_neg.species) == len(scene_pos.species), (
            f"species count mismatch: {len(scene_neg.species)} vs "
            f"{len(scene_pos.species)}"
        )
        # Same sorted coordinates.
        coords_neg = np.sort(scene_neg.frames[0].coords, axis=0)
        coords_pos = np.sort(scene_pos.frames[0].coords, axis=0)
        np.testing.assert_allclose(coords_neg, coords_pos, atol=1e-10)
