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
