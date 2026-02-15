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


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgen:
    def test_single_structure(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct)
        assert len(scene.species) == 2
        assert len(scene.frames) == 1
        assert "Na" in scene.atom_styles
        assert "Cl" in scene.atom_styles

    def test_trajectory(self):
        lattice = Lattice.cubic(5.0)
        s1 = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        s2 = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.26, 0.26, 0.26]])
        scene = from_pymatgen([s1, s2])
        assert len(scene.frames) == 2

    def test_classmethod(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = StructureScene.from_pymatgen(struct)
        assert isinstance(scene, StructureScene)

    def test_view_centred(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct)
        expected = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(scene.view.centre, expected)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgenPbc:
    def test_atom_near_origin_gets_positive_image(self):
        """An atom near frac=0 gets an image shifted by +1."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        # Atom 0 near origin face -> +1 image.  Atom 1 at centre -> none.
        assert len(scene.species) == 3

    def test_atom_near_far_face_gets_negative_image(self):
        """An atom near frac=1 gets an image shifted by -1."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.99, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        # Near far face on x -> -1 image.
        assert len(scene.species) == 2
        xs = sorted(scene.frames[0].coords[:, 0])
        np.testing.assert_allclose(xs[0], -0.1, atol=1e-10)
        np.testing.assert_allclose(xs[1], 9.9, atol=1e-10)

    def test_centre_atom_no_images(self):
        """An atom far from all faces should get no images."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        assert len(scene.species) == 1

    def test_corner_atom_near_origin(self):
        """An atom near the origin corner gets 7 images (+1 on each axis combo)."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.01, 0.01, 0.01]])
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        # 3 faces + 3 edges + 1 corner = 7 images.
        assert len(scene.species) == 8

    def test_corner_atom_near_far_corner(self):
        """An atom near the far corner gets 7 images (-1 on each axis combo)."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.99, 0.99, 0.99]])
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
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
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
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
        scene = from_pymatgen(struct, bond_specs=[na_bond], pbc=True)
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
        scene_neg = from_pymatgen(struct_neg, pbc=True, pbc_cutoff=1.0)
        scene_pos = from_pymatgen(struct_pos, pbc=True, pbc_cutoff=1.0)
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
        scene = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        orig_centroid = np.mean(struct.cart_coords, axis=0)
        expanded_centroid = np.mean(scene.frames[0].coords, axis=0)
        np.testing.assert_allclose(expanded_centroid, orig_centroid, atol=1e-10)

    def test_custom_pbc_cutoff(self):
        """A custom pbc_cutoff should override the bond-derived cutoff."""
        lattice = Lattice.cubic(10.0)
        # Atom at frac 0.05 (cart 0.5) -- within a 1.0 cutoff, but
        # outside a tight 0.3 cutoff.
        struct = Structure(lattice, ["Na"], [[0.05, 0.5, 0.5]])
        scene_wide = from_pymatgen(struct, pbc=True, pbc_cutoff=1.0)
        scene_tight = from_pymatgen(struct, pbc=True, pbc_cutoff=0.3)
        # Wide cutoff includes this atom; tight cutoff excludes it.
        assert len(scene_wide.species) == 2
        assert len(scene_tight.species) == 1


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestExpandBonds:
    """Tests for bond-aware PBC expansion."""

    def test_bonded_image_added(self):
        """An image atom forming a valid bond across the boundary is added."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        # Na at frac (0.5, 0.5, 0.5) — centre of cell.
        # Cl at frac (0.95, 0.5, 0.5) — near far face.
        # Bond cutoff 2.0 A: Cl image at frac (-0.05) = cart -0.25
        # is 2.75 A from Na at cart 2.5 — too far.
        # Use a 3.0 cutoff to catch it.
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.95, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
        )
        # With a tight pbc_cutoff=0.5, _expand_pbc won't add the Cl image
        # (Cl is at frac 0.95, cutoff/face_dist = 0.5/5 = 0.1, and
        # 1 - 0.95 = 0.05 < 0.1, so actually it WILL be added).
        # Use an even tighter cutoff of 0.1 to exclude it.
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True, pbc_cutoff=0.1,
        )
        # _expand_pbc with cutoff=0.1: Cl at frac 0.95 has
        # 1-0.95=0.05, frac_cutoff=0.1/5=0.02. 0.05 > 0.02, so
        # _expand_pbc does NOT add an image.
        # But _expand_bonds should find the Cl image at (-1,0,0)
        # at cart -0.25 which is 2.75 from Na at 2.5 — within 3.0.
        assert len(scene.species) >= 3  # Na, Cl, + at least one image

    def test_no_bond_specs_no_extra_images(self):
        """With no bond specs, no extra images are added."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, bond_specs=[], pbc=True, pbc_cutoff=0.1)
        assert len(scene.species) == 1

    def test_non_recursive(self):
        """Image atoms should not generate further images."""
        from hofmann.bonds import compute_bonds
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(4.0)
        # Two atoms that will bond across boundary.
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=2.0, radius=0.1, colour=0.5,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True, pbc_cutoff=0.5,
        )
        # We should have the 2 unit-cell atoms + their images, but
        # not images-of-images (which would be at frac ~2.0 or ~-1.0).
        coords = scene.frames[0].coords
        # All x coordinates should be within roughly [-cutoff, a+cutoff].
        assert np.all(coords[:, 0] > -2.5)
        assert np.all(coords[:, 0] < 6.5)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestExpandPolyhedraVertices:
    """Tests for vertex expansion of image polyhedron centres."""

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
            pbc_cutoff=1.0,
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
            struct, bond_specs=[ti_o_bond], pbc=True, pbc_cutoff=1.0,
        )
        scene_with_poly = from_pymatgen(
            struct, bond_specs=[ti_o_bond],
            polyhedra=[PolyhedronSpec(centre="Ti")],
            pbc=True, pbc_cutoff=1.0,
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
            pbc=True, pbc_cutoff=1.0,
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
        scene = from_pymatgen(struct, bond_specs=[])
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
