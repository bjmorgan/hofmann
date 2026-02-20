"""Tests for StructureScene construction, validation, and set_atom_data."""

import numpy as np
import pytest

from hofmann.model.frame import Frame
from hofmann.model.structure_scene import StructureScene


class TestStructureScene:
    def test_defaults(self):
        coords = np.zeros((2, 3))
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        assert scene.atom_styles == {}
        assert scene.bond_specs == []
        assert scene.polyhedra == []
        assert scene.title == ""
        np.testing.assert_array_equal(scene.view.rotation, np.eye(3))

    def test_centre_on(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        scene.centre_on(1)
        np.testing.assert_array_equal(scene.view.centre, [4.0, 5.0, 6.0])

    def test_centre_on_invalid_atom_index_raises(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        with pytest.raises(ValueError, match="atom_index"):
            scene.centre_on(5)

    def test_centre_on_invalid_frame_raises(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        with pytest.raises(ValueError, match="frame"):
            scene.centre_on(0, frame=3)

    def test_centre_on_does_not_alias_coords(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords)],
        )
        scene.centre_on(0)
        scene.view.centre[0] = 999.0
        assert scene.frames[0].coords[0, 0] == 1.0

    def test_lattice_default_none(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(species=["A"], frames=[Frame(coords=coords)])
        assert scene.lattice is None

    def test_lattice_accepted(self):
        coords = np.zeros((1, 3))
        lat = np.eye(3) * 5.0
        scene = StructureScene(
            species=["A"], frames=[Frame(coords=coords)], lattice=lat,
        )
        np.testing.assert_array_equal(scene.lattice, lat)

    def test_lattice_bad_shape_raises(self):
        coords = np.zeros((1, 3))
        with pytest.raises(ValueError, match="shape"):
            StructureScene(
                species=["A"], frames=[Frame(coords=coords)],
                lattice=np.eye(2),
            )

    def test_lattice_coerced_to_float(self):
        coords = np.zeros((1, 3))
        lat_int = np.eye(3, dtype=int) * 5
        scene = StructureScene(
            species=["A"], frames=[Frame(coords=coords)], lattice=lat_int,
        )
        assert scene.lattice.dtype == float

    def test_atom_data_default_empty(self):
        coords = np.zeros((2, 3))
        scene = StructureScene(
            species=["A", "B"], frames=[Frame(coords=coords)],
        )
        assert scene.atom_data == {}

    def test_atom_data_constructor_validation(self):
        coords = np.zeros((2, 3))
        with pytest.raises(ValueError, match="atom_data"):
            StructureScene(
                species=["A", "B"], frames=[Frame(coords=coords)],
                atom_data={"charge": np.array([1.0, 2.0, 3.0])},
            )

    def test_frame_atom_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="3 atoms.*frame 0.*2"):
            StructureScene(
                species=["A", "B", "C"],
                frames=[Frame(coords=np.zeros((2, 3)))],
            )

    def test_later_frame_atom_count_mismatch_raises(self):
        good = Frame(coords=np.zeros((2, 3)))
        bad = Frame(coords=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="2 atoms.*frame 1.*3"):
            StructureScene(
                species=["A", "B"],
                frames=[good, bad],
            )

    def test_matching_frame_sizes_accepted(self):
        scene = StructureScene(
            species=["A", "B"],
            frames=[
                Frame(coords=np.zeros((2, 3))),
                Frame(coords=np.ones((2, 3))),
            ],
        )
        assert len(scene.frames) == 2


class TestSetAtomData:
    """Tests for StructureScene.set_atom_data."""

    def _scene(self, n: int = 3) -> StructureScene:
        coords = np.zeros((n, 3))
        return StructureScene(
            species=["A", "B", "C"][:n],
            frames=[Frame(coords=coords)],
        )

    def test_full_array(self):
        scene = self._scene()
        values = np.array([1.0, 2.0, 3.0])
        scene.set_atom_data("charge", values)
        np.testing.assert_array_equal(scene.atom_data["charge"], values)

    def test_full_array_list(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            scene.atom_data["charge"], [1.0, 2.0, 3.0],
        )

    def test_categorical_array(self):
        scene = self._scene()
        scene.set_atom_data("site", np.array(["4a", "8b", "4a"], dtype=object))
        assert list(scene.atom_data["site"]) == ["4a", "8b", "4a"]

    def test_wrong_length_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="length 3"):
            scene.set_atom_data("charge", np.array([1.0, 2.0]))

    def test_sparse_dict_numeric(self):
        scene = self._scene()
        scene.set_atom_data("charge", {0: 1.5, 2: -0.3})
        arr = scene.atom_data["charge"]
        assert arr[0] == pytest.approx(1.5)
        assert np.isnan(arr[1])
        assert arr[2] == pytest.approx(-0.3)

    def test_sparse_dict_string(self):
        scene = self._scene()
        scene.set_atom_data("site", {1: "4a"})
        arr = scene.atom_data["site"]
        assert arr[0] == ""
        assert arr[1] == "4a"
        assert arr[2] == ""

    def test_sparse_dict_out_of_range_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="out of range"):
            scene.set_atom_data("charge", {5: 1.0})

    def test_sparse_dict_empty_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="must not be empty"):
            scene.set_atom_data("charge", {})

    def test_overwrite_existing_key(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.set_atom_data("charge", [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(
            scene.atom_data["charge"], [4.0, 5.0, 6.0],
        )

    def test_multiple_keys(self):
        scene = self._scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.set_atom_data("site", np.array(["a", "b", "c"], dtype=object))
        assert "charge" in scene.atom_data
        assert "site" in scene.atom_data

    def test_sparse_dict_mixed_types_raises(self):
        """Dict with mixed string and numeric values raises TypeError."""
        scene = self._scene()
        with pytest.raises(TypeError, match="same type"):
            scene.set_atom_data("bad", {0: 1, 2: "text"})
