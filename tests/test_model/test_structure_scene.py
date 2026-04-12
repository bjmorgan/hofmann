"""Tests for StructureScene construction, validation, and set_atom_data."""

import numpy as np
import pytest

from hofmann.construction.defaults import default_atom_style
from hofmann.model.frame import Frame
from hofmann.model.structure_scene import StructureScene
from hofmann.model.view_state import ViewState


def _make_scene(
    *,
    n_atoms: int = 3,
    n_frames: int = 3,
) -> StructureScene:
    """Build a minimal scene with *n_atoms* carbons across *n_frames* frames."""
    species = ["C"] * n_atoms
    frames = [
        Frame(coords=np.zeros((n_atoms, 3)))
        for _ in range(n_frames)
    ]
    return StructureScene(
        species=species,
        frames=frames,
        atom_styles={"C": default_atom_style("C")},
    )


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

    def test_lattice_property_returns_first_frame(self):
        coords = np.zeros((1, 3))
        lat = np.eye(3) * 5.0
        scene = StructureScene(
            species=["A"],
            frames=[Frame(coords=coords, lattice=lat)],
        )
        np.testing.assert_array_equal(scene.lattice, lat)

    def test_lattice_assignment_raises(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(
            species=["A"],
            frames=[Frame(coords=coords, lattice=np.eye(3))],
        )
        with pytest.raises(AttributeError, match="read-only"):
            scene.lattice = np.eye(3) * 10.0

    def test_mixed_lattice_frames_raises(self):
        coords = np.zeros((1, 3))
        with pytest.raises(ValueError, match="all frames must have a lattice"):
            StructureScene(
                species=["A"],
                frames=[
                    Frame(coords=coords, lattice=np.eye(3)),
                    Frame(coords=coords),
                ],
            )

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

    def test_view_rejects_non_viewstate(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(species=["A"], frames=[Frame(coords=coords)])
        with pytest.raises(TypeError, match="ViewState"):
            scene.view = (ViewState(), "not a style")

    def test_view_rejects_tuple_with_hint(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(species=["A"], frames=[Frame(coords=coords)])
        with pytest.raises(TypeError, match="unpack"):
            scene.view = (ViewState(), "style")

    def test_view_accepts_viewstate(self):
        coords = np.zeros((1, 3))
        scene = StructureScene(species=["A"], frames=[Frame(coords=coords)])
        new_view = ViewState(zoom=2.0)
        scene.view = new_view
        assert scene.view.zoom == 2.0

    def test_matching_frame_sizes_accepted(self):
        scene = StructureScene(
            species=["A", "B"],
            frames=[
                Frame(coords=np.zeros((2, 3))),
                Frame(coords=np.ones((2, 3))),
            ],
        )
        assert len(scene.frames) == 2

    def test_atom_data_2d_constructor_accepted(self):
        coords = np.zeros((2, 3))
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=coords), Frame(coords=coords + 1)],
            atom_data={"q": np.array([[1.0, 2.0], [3.0, 4.0]])},
        )
        assert scene.atom_data["q"].shape == (2, 2)

    def test_atom_data_2d_wrong_frames_raises(self):
        coords = np.zeros((2, 3))
        with pytest.raises(ValueError, match="atom_data"):
            StructureScene(
                species=["A", "B"],
                frames=[Frame(coords=coords)],
                atom_data={"q": np.array([[1.0, 2.0], [3.0, 4.0]])},
            )

    def test_atom_data_2d_wrong_atoms_raises(self):
        coords = np.zeros((2, 3))
        with pytest.raises(ValueError, match="atom_data"):
            StructureScene(
                species=["A", "B"],
                frames=[Frame(coords=coords), Frame(coords=coords)],
                atom_data={"q": np.array([[1.0], [2.0]])},
            )


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

    def test_2d_numeric_array(self):
        """A (n_frames, n_atoms) numeric array is accepted."""
        coords = np.zeros((3, 3))
        scene = StructureScene(
            species=["A", "B", "C"],
            frames=[Frame(coords=coords), Frame(coords=coords)],
        )
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scene.set_atom_data("charge", values)
        np.testing.assert_array_equal(scene.atom_data["charge"], values)

    def test_2d_wrong_n_frames_raises(self):
        """A 2D array with wrong frame count is rejected."""
        coords = np.zeros((3, 3))
        scene = StructureScene(
            species=["A", "B", "C"],
            frames=[Frame(coords=coords), Frame(coords=coords)],
        )
        with pytest.raises(ValueError, match="2 frames"):
            scene.set_atom_data("charge", np.array([[1.0, 2.0, 3.0]]))

    def test_2d_wrong_n_atoms_raises(self):
        """A 2D array with wrong atom count is rejected."""
        coords = np.zeros((3, 3))
        scene = StructureScene(
            species=["A", "B", "C"],
            frames=[Frame(coords=coords), Frame(coords=coords)],
        )
        with pytest.raises(ValueError, match="3"):
            scene.set_atom_data("charge", np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_2d_single_frame_accepted(self):
        """A (1, n_atoms) array on a 1-frame scene is accepted."""
        scene = self._scene()  # 1 frame, 3 atoms
        values = np.array([[1.0, 2.0, 3.0]])
        scene.set_atom_data("charge", values)
        assert scene.atom_data["charge"].shape == (1, 3)

    def test_2d_categorical_array(self):
        """A (n_frames, n_atoms) object-dtype array is accepted."""
        coords = np.zeros((3, 3))
        scene = StructureScene(
            species=["A", "B", "C"],
            frames=[Frame(coords=coords), Frame(coords=coords)],
        )
        values = np.array([["4a", "8b", "4a"], ["8b", "4a", "8b"]], dtype=object)
        scene.set_atom_data("site", values)
        assert scene.atom_data["site"].shape == (2, 3)
        assert scene.atom_data["site"][0, 1] == "8b"

    def test_by_index_numeric_scalar(self):
        scene = self._scene()
        scene.set_atom_data("charge", by_index={0: 1.5, 2: -0.3})
        arr = scene.atom_data["charge"]
        assert arr[0] == pytest.approx(1.5)
        assert np.isnan(arr[1])
        assert arr[2] == pytest.approx(-0.3)

    def test_by_index_out_of_range_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="out of range"):
            scene.set_atom_data("charge", by_index={5: 1.0})

    def test_by_index_negative_index_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="out of range"):
            scene.set_atom_data("charge", by_index={-1: 1.0})

    def test_values_and_by_index_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError, match="cannot mix"):
            scene.set_atom_data("charge", [1.0, 2.0, 3.0], by_index={0: 1.0})

    def test_all_omitted_raises(self):
        scene = self._scene()
        with pytest.raises(ValueError):
            scene.set_atom_data("charge")

    def test_by_index_categorical(self):
        scene = self._scene()
        scene.set_atom_data("site", by_index={1: "4a"})
        arr = scene.atom_data["site"]
        assert arr[0] is None
        assert arr[1] == "4a"
        assert arr[2] is None
        assert arr.dtype == object

    def test_by_index_mixed_types_raises(self):
        scene = self._scene()
        with pytest.raises(TypeError, match="same type"):
            scene.set_atom_data("bad", by_index={0: 1.0, 2: "text"})


class TestAtomDataWriteMethods:
    """Tests for del_atom_data, clear_2d_atom_data, setter removal."""

    # del_atom_data

    def test_del_atom_data_removes_entry(self):
        scene = _make_scene()
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.del_atom_data("charge")
        assert "charge" not in scene.atom_data

    def test_del_atom_data_missing_key_raises(self):
        scene = _make_scene()
        with pytest.raises(KeyError):
            scene.del_atom_data("missing")

    # clear_2d_atom_data

    def test_clear_2d_atom_data(self):
        scene = _make_scene(n_frames=5)
        scene.set_atom_data("charge", [1.0, 2.0, 3.0])
        scene.set_atom_data("energy", np.zeros((5, 3)))
        scene.set_atom_data("forces", np.ones((5, 3)))
        scene.clear_2d_atom_data()
        # 2-D entries gone
        assert "energy" not in scene.atom_data
        assert "forces" not in scene.atom_data
        # 1-D entry preserved
        np.testing.assert_array_equal(
            scene.atom_data["charge"], [1.0, 2.0, 3.0]
        )

    # Setter removal

    def test_atom_data_setter_removed(self):
        scene = _make_scene()
        with pytest.raises(AttributeError, match="has no setter"):
            scene.atom_data = None  # type: ignore[misc]


class TestValidateForRender:
    """Unit and integration tests for scene-level render-time validation."""

    def _stale_scene(self) -> StructureScene:
        scene = _make_scene(n_frames=3)
        scene.set_atom_data("energy", np.zeros((3, 3)))
        scene.frames.append(Frame(coords=np.zeros((3, 3))))
        return scene

    # --- Unit test directly on the helper ---

    def test_validate_for_render_raises_on_stale_2d(self):
        """Direct unit test on the helper, bypassing the render
        stack.  Faster and narrower than the end-to-end integration
        tests, and guards against a refactor that reshuffles how
        ``render_*`` call the helper without changing the helper
        itself."""
        scene = self._stale_scene()
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'energy'.*3 frames.*4 frames were expected",
        ):
            scene._validate_for_render()

    # --- End-to-end integration tests, one per render method ---

    def test_render_mpl_raises_on_stale_2d(self, tmp_path):
        scene = self._stale_scene()
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'energy'.*3 frames.*4 frames were expected",
        ):
            scene.render_mpl(output=tmp_path / "out.svg")

    def test_render_mpl_interactive_raises_on_stale_2d(self):
        scene = self._stale_scene()
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'energy'.*3 frames.*4 frames were expected",
        ):
            scene.render_mpl_interactive()

    def test_render_animation_raises_on_stale_2d(self, tmp_path):
        scene = self._stale_scene()
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'energy'.*3 frames.*4 frames were expected",
        ):
            scene.render_animation(output=tmp_path / "out.gif")

    # --- Recovery workflow end-to-end test ---

    def test_render_mpl_succeeds_after_recovery(self, tmp_path):
        scene = self._stale_scene()
        scene.clear_2d_atom_data()
        scene.set_atom_data("energy", np.zeros((4, 3)))
        scene.render_mpl(output=tmp_path / "out.svg")

    def test_render_mpl_succeeds_after_in_place_reassign(self, tmp_path):
        """End-to-end: single 2-D entry can be re-set in place at a
        new shape after extending scene.frames, without
        clear_2d_atom_data."""
        scene = _make_scene(n_frames=3)
        scene.set_atom_data("energy", np.zeros((3, 3)))
        scene.frames.append(Frame(coords=np.zeros((3, 3))))
        # No clear_2d_atom_data -- the single entry is overridden by
        # the pending write and replaced atomically.
        scene.set_atom_data("energy", np.ones((4, 3)))
        scene.render_mpl(output=tmp_path / "out.svg")
        assert scene.atom_data["energy"].shape == (4, 3)

    def test_in_place_reassign_names_other_stale_key(self):
        """When a second 2-D entry is stale, the error names that
        entry -- not the one being reassigned -- so the user knows
        which key still needs attention."""
        scene = _make_scene(n_frames=3)
        scene.set_atom_data("energy", np.zeros((3, 3)))
        scene.set_atom_data("forces", np.zeros((3, 3)))
        scene.frames.append(Frame(coords=np.zeros((3, 3))))
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'forces'.*3 frames.*4 frames were expected",
        ):
            scene.set_atom_data("energy", np.ones((4, 3)))
