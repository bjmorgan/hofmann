"""Tests for the _AtomData validated container."""

import numpy as np
import pytest

from hofmann.model.atom_data import _AtomData


def _make_atom_data(*, n_atoms: int, n_frames: int) -> _AtomData:
    """Build an _AtomData with a dummy frames list of the given length."""
    frames = [None] * n_frames  # type: ignore[list-item]
    return _AtomData(n_atoms=n_atoms, frames=frames)


class TestAtomData:

    def test_setitem_1d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_setitem_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ad._set("charge", vals)
        np.testing.assert_array_equal(ad["charge"], vals)

    def test_setitem_converts_list(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("val", [1.0, 2.0])
        assert isinstance(ad["val"], np.ndarray)

    def test_setitem_wrong_length_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(ValueError, match="3"):
            ad._set("charge", np.array([1.0, 2.0]))

    def test_setitem_2d_wrong_frames_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        with pytest.raises(ValueError, match="2 frames"):
            ad._set("charge", np.array([[1.0, 2.0, 3.0]]))

    def test_setitem_2d_wrong_atoms_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        with pytest.raises(ValueError, match="3"):
            ad._set("charge", np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_setitem_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        vals = np.array([["a", "b"], ["c", "d"]], dtype=object)
        ad._set("site", vals)
        assert ad["site"].shape == (2, 2)

    def test_getitem_missing_raises(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        with pytest.raises(KeyError):
            ad["missing"]

    def test_delitem(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("val", [1.0, 2.0])
        ad._del("val")
        assert "val" not in ad

    def test_single_frame_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        vals = np.array([[1.0, 2.0, 3.0]])
        ad._set("q", vals)
        assert ad["q"].shape == (1, 3)

    def test_3d_rejected(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        with pytest.raises(ValueError):
            ad._set("bad", np.ones((2, 2, 2)))

    def test_ranges_2d_numeric(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [0.4, 0.6]]))
        assert ad.ranges["val"] == (0.0, 1.0)

    def test_ranges_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("val", np.array([0.0, 1.0]))
        assert ad.ranges["val"] is None

    def test_ranges_categorical_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("site", np.array([["a", "b"], ["c", "d"]], dtype=object))
        assert ad.ranges["site"] is None

    def test_ranges_all_nan_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.full((2, 2), np.nan))
        assert ad.ranges["val"] is None

    def test_ranges_updates_on_reassignment(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert ad.ranges["val"] == (0.0, 3.0)
        ad._set("val", np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert ad.ranges["val"] == (10.0, 40.0)

    def test_ranges_recomputed_after_delete_and_reassign(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert ad.ranges["val"] == (0.0, 3.0)
        ad._del("val")
        ad._set("val", np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert ad.ranges["val"] == (10.0, 40.0)

    def test_labels_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("site", np.array([["alpha", "beta"], ["beta", "gamma"]], dtype=object))
        assert ad.labels["site"] == ("alpha", "beta", "gamma")

    def test_labels_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("site", np.array(["alpha", "beta"], dtype=object))
        assert ad.labels["site"] is None

    def test_labels_numeric_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert ad.labels["val"] is None

    def test_labels_excludes_missing(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad._set(
            "site",
            np.array(
                [["alpha", None, "beta"], [None, "alpha", None]],
                dtype=object,
            ),
        )
        assert ad.labels["site"] == ("alpha", "beta")

    def test_labels_updates_on_reassignment(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("site", np.array([["a", "b"], ["b", "a"]], dtype=object))
        assert ad.labels["site"] == ("a", "b")
        ad._set("site", np.array([["x", "y"], ["y", "x"]], dtype=object))
        assert ad.labels["site"] == ("x", "y")

    def test_ranges_partial_nan(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad._set("val", np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]]))
        assert ad.ranges["val"] == (1.0, 5.0)

    def test_ranges_is_read_only(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        with pytest.raises(TypeError):
            ad.ranges["val"] = (0.0, 99.0)
        with pytest.raises(TypeError):
            del ad.ranges["val"]

    def test_ranges_attribute_cannot_be_rebound(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        with pytest.raises(AttributeError):
            ad.ranges = {}  # type: ignore[misc]

    def test_ranges_captured_reference_sees_later_updates(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        captured = ad.ranges
        ad._set("new", np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert "new" in captured
        assert captured["new"] == (0.0, 3.0)

    def test_ranges_missing_key_raises(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        with pytest.raises(KeyError):
            ad.ranges["nonexistent"]

    def test_ranges_empty_2d_returns_none(self):
        ad = _make_atom_data(n_atoms=0, n_frames=2)
        ad._set("val", np.empty((2, 0)))
        assert ad.ranges["val"] is None

    @pytest.mark.parametrize(
        "array",
        [
            np.array([1 + 2j, 3 + 4j]),
            np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]),
            np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")]),
            np.array([b"foo", b"bar"]),
            np.array([(1.0, "a"), (2.0, "b")], dtype=[("x", "f8"), ("y", "U5")]),
        ],
        ids=["complex", "datetime", "timedelta", "bytes", "void"],
    )
    def test_setitem_unsupported_dtype_raises(self, array):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        with pytest.raises(ValueError, match="unsupported dtype"):
            ad._set("key", array)

    def test_ranges_contains_every_stored_key(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        ad._set("site", np.array([["a", "b"], ["c", "d"]], dtype=object))
        ad._set("flat", np.array([0.0, 1.0]))
        assert set(ad.ranges) == set(ad)
        ad._del("site")
        assert set(ad.ranges) == set(ad)

    def test_labels_is_read_only(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("site", np.array([["a", "b"], ["c", "d"]], dtype=object))
        with pytest.raises(TypeError):
            ad.labels["site"] = ("x", "y")
        with pytest.raises(TypeError):
            del ad.labels["site"]

    def test_labels_attribute_cannot_be_rebound(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        with pytest.raises(AttributeError):
            ad.labels = {}  # type: ignore[misc]

    def test_labels_missing_key_raises(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("site", np.array([["a", "b"], ["c", "d"]], dtype=object))
        with pytest.raises(KeyError):
            ad.labels["nonexistent"]

    def test_labels_contains_every_stored_key(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad._set("val", np.array([[0.0, 1.0], [2.0, 3.0]]))
        ad._set("site", np.array([["a", "b"], ["c", "d"]], dtype=object))
        ad._set("flat", np.array([0.0, 1.0]))
        assert set(ad.labels) == set(ad)
        ad._del("val")
        assert set(ad.labels) == set(ad)

    def test_negative_n_atoms_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_atom_data(n_atoms=-1, n_frames=1)

    def test_dynamic_frame_count(self):
        """_AtomData tracks the live frame list length."""
        frames: list = [None]  # type: ignore[list-item]
        ad = _AtomData(n_atoms=2, frames=frames)
        assert ad.n_frames == 1
        frames.append(None)
        assert ad.n_frames == 2
        # 2-D array with 2 rows now valid.
        ad._set("val", np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert ad["val"].shape == (2, 2)

    def test_setitem_1d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0] = 99.0

    def test_setitem_2d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad._set("charge", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0, 0] = 99.0

    def test_setitem_local_reference_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        arr = ad["charge"]
        with pytest.raises(ValueError, match="read-only"):
            arr[...] = 0.0

    def test_setitem_does_not_freeze_caller_source(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        src = np.array([1.0, 2.0, 3.0])
        ad._set("charge", src)
        # Source is still writable.
        assert src.flags.writeable is True
        # Mutating the source does not affect the stored array.
        src[0] = 99.0
        assert ad["charge"][0] == 1.0

    def test_setitem_reassignment_replaces_values(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        ad._set("charge", np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(ad["charge"], [4.0, 5.0, 6.0])

    def test_setitem_accepts_already_read_only_ndarray(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        src = np.array([1.0, 2.0, 3.0])
        src.flags.writeable = False
        ad._set("charge", src)
        assert ad["charge"].flags.writeable is False
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_setitem_object_dtype_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("site", np.array(["alpha", "beta"], dtype=object))
        with pytest.raises(ValueError, match="read-only"):
            ad["site"][0] = "gamma"

    def test_setitem_list_input_stored_read_only(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad._set("val", [1.0, 2.0])
        with pytest.raises(ValueError, match="read-only"):
            ad["val"][0] = 99.0

    def test_stored_array_in_place_mutation_raises(self):
        ad = _AtomData(n_atoms=3, frames=[None])  # type: ignore[list-item]
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        arr = ad["charge"]
        with pytest.raises(ValueError, match="read-only"):
            arr += 1.0

    def test_setitem_view_of_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        view = ad["charge"][1:]
        with pytest.raises(ValueError, match="read-only"):
            view[0] = 99.0

    def test_repr_empty(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        assert repr(ad) == "AtomData()"

    def test_repr_one_1d_entry(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        assert repr(ad) == "AtomData({'charge': (3,)})"

    def test_repr_one_2d_entry(self):
        ad = _make_atom_data(n_atoms=3, n_frames=5)
        ad._set("energy", np.zeros((5, 3)))
        assert repr(ad) == "AtomData({'energy': (5, 3)})"

    def test_repr_mixed_entries(self):
        ad = _make_atom_data(n_atoms=3, n_frames=5)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        ad._set("energy", np.zeros((5, 3)))
        assert repr(ad) == "AtomData({'charge': (3,), 'energy': (5, 3)})"

    def test_bracket_assignment_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(TypeError, match="does not support item assignment"):
            ad["charge"] = np.array([1.0, 2.0, 3.0])  # type: ignore[index]

    def test_bracket_delete_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad._set("charge", np.array([1.0, 2.0, 3.0]))
        with pytest.raises(TypeError):
            del ad["charge"]  # type: ignore[attr-defined]

    def test_pop_not_available(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(AttributeError):
            ad.pop("charge")  # type: ignore[attr-defined]

    def test_popitem_not_available(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(AttributeError):
            ad.popitem()  # type: ignore[attr-defined]

    def test_setdefault_not_available(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(AttributeError):
            ad.setdefault("charge", np.array([0.0, 0.0, 0.0]))  # type: ignore[attr-defined]

    def test_update_not_available(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(AttributeError):
            ad.update({"charge": np.array([1.0, 2.0, 3.0])})  # type: ignore[attr-defined]

    def test_clear_not_available(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(AttributeError):
            ad.clear()  # type: ignore[attr-defined]
