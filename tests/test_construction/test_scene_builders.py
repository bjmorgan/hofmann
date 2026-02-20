"""Tests for hofmann.scene — convenience constructors."""

import numpy as np
import pytest

from hofmann.model import AtomStyle, StructureScene, ViewState
from hofmann.construction.scene_builders import from_xbs, from_pymatgen

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
        """PBC is enabled by default for pymatgen structures."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.0, 0.5, 0.5]])
        scene = from_pymatgen(struct, bond_specs=[])
        assert scene.pbc is True
        # Scene stores physical atoms only.
        assert len(scene.species) == 1

    def test_atom_styles_override(self):
        """Custom atom_styles override defaults while preserving others."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        custom_na = AtomStyle(radius=2.0, colour=(1.0, 0.0, 0.0))
        scene = from_pymatgen(
            struct, pbc=False, atom_styles={"Na": custom_na},
        )
        assert scene.atom_styles["Na"] is custom_na
        # Cl should still have the auto-generated default.
        assert "Cl" in scene.atom_styles
        assert scene.atom_styles["Cl"] is not custom_na

    def test_title_passthrough(self):
        """The title parameter is passed through to the scene."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False, title="test title")
        assert scene.title == "test title"

    def test_view_override(self):
        """A custom ViewState overrides the auto-centred default."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        custom_view = ViewState(centre=np.array([1.0, 2.0, 3.0]))
        scene = from_pymatgen(struct, pbc=False, view=custom_view)
        assert scene.view is custom_view

    def test_atom_data_passthrough(self):
        """The atom_data parameter is passed through to the scene."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        data = {"charge": np.array([1.0, -1.0])}
        scene = from_pymatgen(struct, pbc=False, atom_data=data)
        np.testing.assert_array_equal(scene.atom_data["charge"], [1.0, -1.0])

    def test_classmethod_forwards_kwargs(self):
        """The classmethod variant forwards style kwargs."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        custom_na = AtomStyle(radius=2.0, colour="red")
        scene = StructureScene.from_pymatgen(
            struct, pbc=False,
            atom_styles={"Na": custom_na},
            title="via classmethod",
        )
        assert scene.atom_styles["Na"] is custom_na
        assert scene.title == "via classmethod"


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgenPbc:
    def test_scene_stores_physical_atoms_only(self):
        """PBC scenes store only physical atoms; expansion is at render time."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(10.0)
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=5.0, radius=0.1, colour=0.5, complete="*",
        )
        scene = from_pymatgen(struct, bond_specs=[bond], pbc=True)
        assert len(scene.species) == 2  # Physical atoms only.

    def test_pbc_flag_stored(self):
        """from_pymatgen stores the pbc flag on the scene."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene_on = from_pymatgen(struct, bond_specs=[], pbc=True)
        scene_off = from_pymatgen(struct, bond_specs=[], pbc=False)
        assert scene_on.pbc is True
        assert scene_off.pbc is False

    def test_pbc_false_no_expansion(self):
        """With pbc=False, no images are added."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.01, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=False)
        assert len(scene.species) == 1

    def test_wrapping_invariance(self):
        """Wrapped coordinates are the same regardless of input wrapping."""
        lattice = Lattice.cubic(10.0)
        # Atom at frac -0.01 is equivalent to frac 0.99.
        struct_neg = Structure(lattice, ["Na"], [[-.01, 0.5, 0.5]])
        struct_pos = Structure(lattice, ["Na"], [[0.99, 0.5, 0.5]])
        scene_neg = from_pymatgen(struct_neg, pbc=True)
        scene_pos = from_pymatgen(struct_pos, pbc=True)
        assert len(scene_neg.species) == len(scene_pos.species)
        np.testing.assert_allclose(
            np.sort(scene_neg.frames[0].coords, axis=0),
            np.sort(scene_pos.frames[0].coords, axis=0),
            atol=1e-10,
        )

    def test_coords_wrapped_to_unit_cell(self):
        """PBC wraps fractional coordinates to [0, 1)."""
        lattice = Lattice.cubic(10.0)
        struct = Structure(lattice, ["Na"], [[0.02, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True)
        # Physical atom at frac 0.02 → cart 0.2
        np.testing.assert_allclose(
            scene.frames[0].coords[0, 0], 0.2, atol=1e-10,
        )


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestPbcSceneConstruction:
    """Tests for PBC scene construction details."""

    def test_no_bond_specs_physical_atoms_only(self):
        """With no bond specs, scene has physical atoms only."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, bond_specs=[], pbc=True)
        assert len(scene.species) == 1

    def test_physical_coords_in_unit_cell(self):
        """Physical atoms are wrapped into the unit cell."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(4.0)
        struct = Structure(
            lattice, ["Na", "Na"],
            [[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]],
        )
        bond_spec = BondSpec(
            species=("Na", "Na"), min_length=0.0,
            max_length=2.0, radius=0.1, colour=0.5,
        )
        scene = from_pymatgen(
            struct, bond_specs=[bond_spec], pbc=True,
        )
        coords = scene.frames[0].coords
        # All coordinates should be inside the unit cell.
        assert np.all(coords >= 0.0 - 1e-10)
        assert np.all(coords[:, 0] < 4.0 + 1e-10)


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestPolyhedraSpecPreservation:
    """Polyhedra specs are stored on the scene for use at render time."""

    def test_polyhedra_stored(self):
        """from_pymatgen preserves polyhedra specs on the scene."""
        from hofmann.model import PolyhedronSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Ti", "O"],
            [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
        )
        poly = PolyhedronSpec(centre="Ti")
        scene = from_pymatgen(struct, polyhedra=[poly], pbc=True)
        assert scene.polyhedra == [poly]
        assert len(scene.species) == 2  # Physical atoms only.

    def test_default_no_polyhedra(self):
        """Without polyhedra kwarg, scene has empty polyhedra list."""
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.5, 0.5, 0.5]])
        scene = from_pymatgen(struct, pbc=True)
        assert scene.polyhedra == []


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


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestFromPymatgenEmptySequence:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            from_pymatgen([], bond_specs=[])


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestCentreAtomValidation:
    def test_invalid_centre_atom_raises(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="centre_atom"):
            from_pymatgen(struct, bond_specs=[], centre_atom=10)

    def test_negative_centre_atom_raises(self):
        lattice = Lattice.cubic(5.0)
        struct = Structure(lattice, ["Na"], [[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="centre_atom"):
            from_pymatgen(struct, bond_specs=[], centre_atom=-1)


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
class TestRecursiveBondSceneConstruction:
    """Scene construction with recursive bond specs."""

    def test_recursive_spec_preserved(self):
        """Recursive bond specs are preserved on the scene."""
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
        scene = from_pymatgen(struct, bond_specs=[bond], pbc=True)
        assert scene.bond_specs[0].recursive is True
        assert len(scene.species) == 2  # Physical atoms only.

    def test_scene_stores_physical_atoms_with_recursive(self):
        """Scenes with recursive specs store physical atoms only."""
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
        scene = from_pymatgen(struct, bond_specs=[bond], pbc=True)
        assert len(scene.species) == 2
        assert set(scene.species) == {"Na", "Cl"}

    def test_invalid_max_depth_raises(self):
        """max_recursive_depth < 1 raises ValueError."""
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
        with pytest.raises(ValueError, match="max_recursive_depth"):
            from_pymatgen(
                struct, bond_specs=[bond], pbc=True,
                max_recursive_depth=0,
            )
        with pytest.raises(ValueError, match="max_recursive_depth"):
            from_pymatgen(
                struct, bond_specs=[bond], pbc=True,
                max_recursive_depth=-1,
            )


@pytest.mark.skipif(not _has_pymatgen, reason="pymatgen not installed")
class TestCompleteBondSceneConstruction:
    """Scene construction with complete bond specs."""

    def test_complete_spec_preserved(self):
        """Complete bond specs are preserved on the scene."""
        from hofmann.model import BondSpec

        lattice = Lattice.cubic(5.0)
        struct = Structure(
            lattice, ["Na", "Cl"],
            [[0.5, 0.5, 0.5], [0.98, 0.5, 0.5]],
        )
        bond = BondSpec(
            species=("Na", "Cl"), min_length=0.0,
            max_length=3.0, radius=0.1, colour=0.5,
            complete="*",
        )
        scene = from_pymatgen(struct, bond_specs=[bond], pbc=True)
        assert scene.bond_specs[0].complete == "*"
        assert len(scene.species) == 2  # Physical atoms only.

    def test_wrapping_consistency(self):
        """Equivalent fractional coordinates produce identical scenes.

        Structures with fractional coordinates that differ by lattice
        translations (e.g. -0.01 vs 0.99) should produce the same scene
        after wrapping to [0, 1).
        """
        lattice = Lattice.cubic(10.0)
        struct_neg = Structure(
            lattice, ["Na", "Cl"],
            [[-0.01, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        struct_pos = Structure(
            lattice, ["Na", "Cl"],
            [[0.99, 0.5, 0.5], [0.5, 0.5, 0.5]],
        )
        scene_neg = from_pymatgen(struct_neg, pbc=True)
        scene_pos = from_pymatgen(struct_pos, pbc=True)
        assert len(scene_neg.species) == len(scene_pos.species)
        np.testing.assert_allclose(
            np.sort(scene_neg.frames[0].coords, axis=0),
            np.sort(scene_pos.frames[0].coords, axis=0),
            atol=1e-10,
        )
