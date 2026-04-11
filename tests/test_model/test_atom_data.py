"""Tests for the AtomData validated container."""

import numpy as np
import pytest

from hofmann.model.atom_data import AtomData


def _make_atom_data(*, n_atoms: int) -> AtomData:
    """Build an AtomData container for the given number of atoms."""
    return AtomData(n_atoms=n_atoms)


class TestAtomData:

    def test_setitem_1d_accepted(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_setitem_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3)
        vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ad._set("charge", vals, expected_frames=2)
        np.testing.assert_array_equal(ad["charge"], vals)

    def test_setitem_converts_list(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set("val", [1.0, 2.0], expected_frames=1)
        assert isinstance(ad["val"], np.ndarray)

    def test_setitem_wrong_length_raises(self):
        ad = _make_atom_data(n_atoms=3)
        with pytest.raises(ValueError, match="3"):
            ad._set("charge", np.array([1.0, 2.0]), expected_frames=1)

    def test_setitem_2d_wrong_atoms_raises(self):
        ad = _make_atom_data(n_atoms=3)
        with pytest.raises(ValueError, match="3"):
            ad._set(
                "charge",
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                expected_frames=2,
            )

    def test_setitem_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2)
        vals = np.array([["a", "b"], ["c", "d"]], dtype=object)
        ad._set("site", vals, expected_frames=2)
        assert ad["site"].shape == (2, 2)

    def test_getitem_missing_raises(self):
        ad = _make_atom_data(n_atoms=2)
        with pytest.raises(KeyError):
            ad["missing"]

    def test_delitem(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set("val", [1.0, 2.0], expected_frames=1)
        ad._del("val")
        assert "val" not in ad

    def test_single_frame_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3)
        vals = np.array([[1.0, 2.0, 3.0]])
        ad._set("q", vals, expected_frames=1)
        assert ad["q"].shape == (1, 3)

    def test_3d_rejected(self):
        ad = _make_atom_data(n_atoms=2)
        with pytest.raises(ValueError):
            ad._set("bad", np.ones((2, 2, 2)), expected_frames=2)

    def test_ranges_2d_numeric(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val", np.array([[0.0, 1.0], [0.4, 0.6]]), expected_frames=2
        )
        assert ad.ranges["val"] == (0.0, 1.0)

    def test_ranges_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set("val", np.array([0.0, 1.0]), expected_frames=1)
        assert ad.ranges["val"] is None

    def test_ranges_categorical_returns_none(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array([["a", "b"], ["c", "d"]], dtype=object),
            expected_frames=2,
        )
        assert ad.ranges["site"] is None

    def test_ranges_all_nan_returns_none(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set("val", np.full((2, 2), np.nan), expected_frames=2)
        assert ad.ranges["val"] is None

    def test_ranges_updates_on_reassignment(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        assert ad.ranges["val"] == (0.0, 3.0)
        ad._set(
            "val",
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            expected_frames=2,
        )
        assert ad.ranges["val"] == (10.0, 40.0)

    def test_ranges_recomputed_after_delete_and_reassign(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        assert ad.ranges["val"] == (0.0, 3.0)
        ad._del("val")
        ad._set(
            "val",
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            expected_frames=2,
        )
        assert ad.ranges["val"] == (10.0, 40.0)

    def test_labels_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array([["alpha", "beta"], ["beta", "gamma"]], dtype=object),
            expected_frames=2,
        )
        assert ad.labels["site"] == ("alpha", "beta", "gamma")

    def test_labels_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array(["alpha", "beta"], dtype=object),
            expected_frames=1,
        )
        assert ad.labels["site"] is None

    def test_labels_numeric_returns_none(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            expected_frames=2,
        )
        assert ad.labels["val"] is None

    def test_labels_excludes_missing(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set(
            "site",
            np.array(
                [["alpha", None, "beta"], [None, "alpha", None]],
                dtype=object,
            ),
            expected_frames=2,
        )
        assert ad.labels["site"] == ("alpha", "beta")

    def test_labels_updates_on_reassignment(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array([["a", "b"], ["b", "a"]], dtype=object),
            expected_frames=2,
        )
        assert ad.labels["site"] == ("a", "b")
        ad._set(
            "site",
            np.array([["x", "y"], ["y", "x"]], dtype=object),
            expected_frames=2,
        )
        assert ad.labels["site"] == ("x", "y")

    def test_ranges_partial_nan(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set(
            "val",
            np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]]),
            expected_frames=2,
        )
        assert ad.ranges["val"] == (1.0, 5.0)

    def test_ranges_is_read_only(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        with pytest.raises(TypeError):
            ad.ranges["val"] = (0.0, 99.0)
        with pytest.raises(TypeError):
            del ad.ranges["val"]

    def test_ranges_attribute_cannot_be_rebound(self):
        ad = _make_atom_data(n_atoms=2)
        with pytest.raises(AttributeError):
            ad.ranges = {}  # type: ignore[misc]

    def test_ranges_captured_reference_sees_later_updates(self):
        ad = _make_atom_data(n_atoms=2)
        captured = ad.ranges
        ad._set(
            "new",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        assert "new" in captured
        assert captured["new"] == (0.0, 3.0)

    def test_ranges_missing_key_raises(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        with pytest.raises(KeyError):
            ad.ranges["nonexistent"]

    def test_ranges_empty_2d_returns_none(self):
        ad = _make_atom_data(n_atoms=0)
        ad._set("val", np.empty((2, 0)), expected_frames=2)
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
        ad = _make_atom_data(n_atoms=2)
        with pytest.raises(ValueError, match="unsupported dtype"):
            ad._set("key", array, expected_frames=1)

    def test_ranges_contains_every_stored_key(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        ad._set(
            "site",
            np.array([["a", "b"], ["c", "d"]], dtype=object),
            expected_frames=2,
        )
        ad._set("flat", np.array([0.0, 1.0]), expected_frames=2)
        assert set(ad.ranges) == set(ad)
        ad._del("site")
        assert set(ad.ranges) == set(ad)

    def test_labels_is_read_only(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array([["a", "b"], ["c", "d"]], dtype=object),
            expected_frames=2,
        )
        with pytest.raises(TypeError):
            ad.labels["site"] = ("x", "y")
        with pytest.raises(TypeError):
            del ad.labels["site"]

    def test_labels_attribute_cannot_be_rebound(self):
        ad = _make_atom_data(n_atoms=2)
        with pytest.raises(AttributeError):
            ad.labels = {}  # type: ignore[misc]

    def test_labels_missing_key_raises(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array([["a", "b"], ["c", "d"]], dtype=object),
            expected_frames=2,
        )
        with pytest.raises(KeyError):
            ad.labels["nonexistent"]

    def test_labels_contains_every_stored_key(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "val",
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            expected_frames=2,
        )
        ad._set(
            "site",
            np.array([["a", "b"], ["c", "d"]], dtype=object),
            expected_frames=2,
        )
        ad._set("flat", np.array([0.0, 1.0]), expected_frames=2)
        assert set(ad.labels) == set(ad)
        ad._del("val")
        assert set(ad.labels) == set(ad)

    def test_negative_n_atoms_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_atom_data(n_atoms=-1)

    def test_setitem_1d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0] = 99.0

    def test_setitem_2d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set(
            "charge",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            expected_frames=2,
        )
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0, 0] = 99.0

    def test_setitem_local_reference_is_read_only(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        arr = ad["charge"]
        with pytest.raises(ValueError, match="read-only"):
            arr[...] = 0.0

    def test_setitem_does_not_freeze_caller_source(self):
        ad = _make_atom_data(n_atoms=3)
        src = np.array([1.0, 2.0, 3.0])
        ad._set("charge", src, expected_frames=1)
        # Source is still writable.
        assert src.flags.writeable is True
        # Mutating the source does not affect the stored array.
        src[0] = 99.0
        assert ad["charge"][0] == 1.0

    def test_setitem_reassignment_replaces_values(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        ad._set("charge", np.array([4.0, 5.0, 6.0]), expected_frames=1)
        np.testing.assert_array_equal(ad["charge"], [4.0, 5.0, 6.0])

    def test_setitem_accepts_already_read_only_ndarray(self):
        ad = _make_atom_data(n_atoms=3)
        src = np.array([1.0, 2.0, 3.0])
        src.flags.writeable = False
        ad._set("charge", src, expected_frames=1)
        assert ad["charge"].flags.writeable is False
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_setitem_object_dtype_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set(
            "site",
            np.array(["alpha", "beta"], dtype=object),
            expected_frames=1,
        )
        with pytest.raises(ValueError, match="read-only"):
            ad["site"][0] = "gamma"

    def test_setitem_list_input_stored_read_only(self):
        ad = _make_atom_data(n_atoms=2)
        ad._set("val", [1.0, 2.0], expected_frames=1)
        with pytest.raises(ValueError, match="read-only"):
            ad["val"][0] = 99.0

    def test_stored_array_in_place_mutation_raises(self):
        ad = AtomData(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        arr = ad["charge"]
        with pytest.raises(ValueError, match="read-only"):
            arr += 1.0

    def test_setitem_view_of_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        view = ad["charge"][1:]
        with pytest.raises(ValueError, match="read-only"):
            view[0] = 99.0

    def test_repr_empty(self):
        ad = _make_atom_data(n_atoms=3)
        assert repr(ad) == "AtomData()"

    def test_repr_mixed_entries(self):
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=5)
        ad._set("energy", np.zeros((5, 3)), expected_frames=5)
        assert repr(ad) == "AtomData({'charge': (3,), 'energy': (5, 3)})"

    def test_bracket_assignment_raises(self):
        """``ad[key] = value`` raises TypeError at the bytecode
        level because ``Mapping`` defines no ``__setitem__``."""
        ad = _make_atom_data(n_atoms=3)
        with pytest.raises(TypeError, match="does not support item assignment"):
            ad["charge"] = np.array([1.0, 2.0, 3.0])  # type: ignore[index]

    def test_bracket_del_raises(self):
        """``del ad[key]`` raises TypeError at the bytecode level
        because ``Mapping`` defines no ``__delitem__``."""
        ad = _make_atom_data(n_atoms=3)
        ad._set("charge", np.array([1.0, 2.0, 3.0]), expected_frames=1)
        with pytest.raises(TypeError, match="does not support item deletion"):
            del ad["charge"]  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "method_name",
        ["__setitem__", "__delitem__", "pop", "popitem",
         "setdefault", "update", "clear"],
    )
    def test_mutable_mapping_methods_absent(self, method_name):
        """Pin the concrete absence of each ``MutableMapping`` entry
        point.  A future refactor that accidentally added any of
        these would silently reopen the mutation surface; the
        ``isinstance(_, MutableMapping)`` check alone would not
        catch an ad-hoc method added without inheriting the mixin.
        """
        ad = _make_atom_data(n_atoms=3)
        assert not hasattr(ad, method_name)

    def test_inherits_from_mapping_not_mutable_mapping(self):
        """Pin the base-class switch directly.  Per-method absence
        is covered by ``test_mutable_mapping_methods_absent``; this
        pins the structural decision at the isinstance level.
        """
        from collections.abc import Mapping, MutableMapping
        ad = AtomData(n_atoms=3)
        assert isinstance(ad, Mapping)
        assert not isinstance(ad, MutableMapping)

    def test_init_rejects_frames_kwarg(self):
        """Passing frames= raises TypeError (API removed)."""
        with pytest.raises(TypeError):
            AtomData(n_atoms=3, frames=[])  # type: ignore[call-arg]

    def test_atom_data_not_importable_from_top_level(self):
        """The class is deliberately kept out of ``hofmann.__all__``
        and ``hofmann.model.__all__`` so users cannot construct one
        at the top-level import path; the supported way to obtain an
        instance is via :attr:`StructureScene.atom_data`.
        """
        import hofmann
        import hofmann.model
        assert "AtomData" not in getattr(hofmann, "__all__", [])
        assert "AtomData" not in getattr(hofmann.model, "__all__", [])
        assert not hasattr(hofmann, "AtomData")
        assert not hasattr(hofmann.model, "AtomData")

    def test_set_rejects_2d_wrong_new_shape(self):
        """New 2-D array with shape[0] != expected_frames raises."""
        ad = AtomData(n_atoms=3)
        with pytest.raises(
            ValueError,
            match=r"atom_data\['foo'\] has 4 rows but 5 frames were expected",
        ):
            ad._set("foo", np.ones((4, 3)), expected_frames=5)

    def test_set_accepts_consistent_2d_across_entries(self):
        """Two 2-D entries with the same shape and expected_frames succeed."""
        ad = AtomData(n_atoms=3)
        ad._set("foo", np.zeros((5, 3)), expected_frames=5)
        ad._set("bar", np.ones((5, 3)), expected_frames=5)
        assert ad["bar"].shape == (5, 3)

    def test_set_detects_stale_stored_entry(self):
        """A stored 2-D entry stale relative to expected_frames raises.

        Exercises the cross-entry branch of ``_set``: the new array
        matches *expected_frames*, but an already-stored 2-D entry
        does not.  This is the post-``scene.frames.append(...)``
        scenario from #69.  The error message pins the key name,
        both frame counts, and the recovery-advice tail pointing at
        ``clear_2d_atom_data`` so a future reword does not silently
        drop the user-actionable part of the message.
        """
        ad = AtomData(n_atoms=3)
        ad._set("foo", np.zeros((3, 3)), expected_frames=3)
        with pytest.raises(ValueError) as exc_info:
            ad._set("bar", np.ones((4, 3)), expected_frames=4)
        message = str(exc_info.value)
        assert "stale 2-D entry 'foo'" in message
        assert "3 frames" in message
        assert "4 frames were expected" in message
        assert "clear_2d_atom_data" in message

    def test_del_last_2d_releases_shape_constraint(self):
        """After deleting the only 2-D entry, a new 2-D with any
        matching expected_frames is accepted."""
        ad = AtomData(n_atoms=3)
        ad._set("foo", np.zeros((5, 3)), expected_frames=5)
        ad._del("foo")
        ad._set("foo", np.ones((4, 3)), expected_frames=4)
        assert ad["foo"].shape == (4, 3)

    def test_del_2d_preserves_constraint_when_others_remain(self):
        """Deleting one 2-D entry does not release the cross-entry
        constraint when others remain; a subsequent 2-D set with a
        non-matching ``expected_frames`` still raises."""
        ad = AtomData(n_atoms=3)
        ad._set("energy", np.zeros((5, 3)), expected_frames=5)
        ad._set("forces", np.zeros((5, 3)), expected_frames=5)
        ad._del("energy")
        # "forces" still stored at shape (5, 3); a new 2-D at a
        # different expected_frames must fail on the stored entry.
        with pytest.raises(
            ValueError, match=r"stale 2-D entry 'forces'"
        ):
            ad._set("new", np.ones((4, 3)), expected_frames=4)

    def test_set_2d_nested_list_coerced(self):
        """2-D nested-list input is coerced to an ndarray before
        shape inspection.  Catches refactors that move shape checks
        ahead of the ``np.array(value)`` coercion."""
        ad = AtomData(n_atoms=3)
        ad._set(
            "val", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], expected_frames=2
        )
        assert ad["val"].shape == (2, 3)
        np.testing.assert_array_equal(
            ad["val"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

    def test_in_place_reassign_same_key_new_shape_succeeds(self):
        """Reassigning a 2-D entry in place with a new shape is
        accepted when it is the only 2-D entry.

        The pending override replaces the stored entry for the same
        key, so the stale stored version is not counted against the
        cross-entry invariant.  Post-operation, the container holds
        exactly the new shape and the invariant is preserved.
        """
        ad = AtomData(n_atoms=3)
        ad._set("energy", np.zeros((3, 3)), expected_frames=3)
        # Scene has extended its frames list to 4.  Reassign in place.
        ad._set("energy", np.ones((4, 3)), expected_frames=4)
        assert ad["energy"].shape == (4, 3)
        np.testing.assert_array_equal(ad["energy"], np.ones((4, 3)))

    def test_set_bad_dtype_fires_before_stale_stored_check(self):
        """Input validation (dtype) runs before state validation.

        A 2-D write with an unsupported dtype must raise the dtype
        error immediately, even when the container holds a stored
        entry stale relative to *expected_frames*.  Firing the
        stale-stored check first would send the user on an
        unnecessary ``clear_2d_atom_data()`` recovery for a write
        that was never going to land regardless.
        """
        ad = AtomData(n_atoms=3)
        ad._set("energy", np.zeros((3, 3)), expected_frames=3)
        # Scene has been extended to 4 frames.  User tries to set a
        # new 2-D entry with an unsupported (complex) dtype.  The
        # stored "energy" is stale (3 vs 4), but the dtype is the
        # actionable issue.
        bad = np.ones((4, 3), dtype=np.complex64)
        with pytest.raises(ValueError, match="unsupported dtype"):
            ad._set("new_key", bad, expected_frames=4)

    def test_in_place_reassign_fails_when_other_2d_stale(self):
        """Reassigning one 2-D entry in place fails when an *other*
        stored 2-D entry is stale relative to the new frame count.

        The error names the stale stored entry, not the one being
        reassigned: that is the key the user still needs to fix.
        """
        ad = AtomData(n_atoms=3)
        ad._set("energy", np.zeros((3, 3)), expected_frames=3)
        ad._set("forces", np.zeros((3, 3)), expected_frames=3)
        with pytest.raises(
            ValueError,
            match=r"stale 2-D entry 'forces' sized for 3 frames",
        ):
            ad._set("energy", np.ones((4, 3)), expected_frames=4)

    def test_clear_2d(self):
        ad = AtomData(n_atoms=3)
        ad._set(
            "charge", np.array([1.0, 2.0, 3.0]), expected_frames=5
        )  # 1-D
        ad._set("energy", np.zeros((5, 3)), expected_frames=5)  # 2-D
        ad._set("forces", np.ones((5, 3)), expected_frames=5)  # 2-D
        ad._clear_2d()
        # 2-D entries gone
        assert "energy" not in ad
        assert "forces" not in ad
        # 1-D entry preserved
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_clear_2d_allows_new_shape(self):
        ad = AtomData(n_atoms=3)
        ad._set("energy", np.zeros((5, 3)), expected_frames=5)
        ad._clear_2d()
        # Constraint released; a differently-shaped 2-D is now accepted
        ad._set("forces", np.ones((7, 3)), expected_frames=7)
        assert ad["forces"].shape == (7, 3)
