"""Tests for hofmann.scene — convenience constructors."""

import numpy as np
import pytest

from hofmann.model import StructureScene
from hofmann.scene import from_xbs, from_pymatgen

_has_pymatgen = False
try:
    from pymatgen.core import Lattice, Structure

    _has_pymatgen = True
except ImportError:
    pass


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
