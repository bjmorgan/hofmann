"""Tests for the AtomData validated container."""

import numpy as np
import pytest

from hofmann.model.atom_data import AtomData


def _make_atom_data(*, n_atoms: int, n_frames: int) -> AtomData:
    """Build an AtomData with a dummy frames list of the given length."""
    frames = [None] * n_frames  # type: ignore[list-item]
    return AtomData(n_atoms=n_atoms, frames=frames)


class TestAtomData:

    def test_setitem_1d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad["charge"] = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ad["charge"], [1.0, 2.0, 3.0])

    def test_setitem_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ad["charge"] = vals
        np.testing.assert_array_equal(ad["charge"], vals)

    def test_setitem_converts_list(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["val"] = [1.0, 2.0]
        assert isinstance(ad["val"], np.ndarray)

    def test_setitem_wrong_length_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        with pytest.raises(ValueError, match="3"):
            ad["charge"] = np.array([1.0, 2.0])

    def test_setitem_2d_wrong_frames_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        with pytest.raises(ValueError, match="2 frames"):
            ad["charge"] = np.array([[1.0, 2.0, 3.0]])

    def test_setitem_2d_wrong_atoms_raises(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        with pytest.raises(ValueError, match="3"):
            ad["charge"] = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_setitem_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        vals = np.array([["a", "b"], ["c", "d"]], dtype=object)
        ad["site"] = vals
        assert ad["site"].shape == (2, 2)

    def test_getitem_missing_raises(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        with pytest.raises(KeyError):
            ad["missing"]

    def test_delitem(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["val"] = [1.0, 2.0]
        del ad["val"]
        assert "val" not in ad

    def test_single_frame_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        vals = np.array([[1.0, 2.0, 3.0]])
        ad["q"] = vals
        assert ad["q"].shape == (1, 3)

    def test_3d_rejected(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        with pytest.raises(ValueError):
            ad["bad"] = np.ones((2, 2, 2))

    def test_global_range_2d_numeric(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.array([[0.0, 1.0], [0.4, 0.6]])
        assert ad.global_range("val") == (0.0, 1.0)

    def test_global_range_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["val"] = np.array([0.0, 1.0])
        assert ad.global_range("val") is None

    def test_global_range_categorical_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["site"] = np.array([["a", "b"], ["c", "d"]], dtype=object)
        assert ad.global_range("site") is None

    def test_global_range_all_nan_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.full((2, 2), np.nan)
        assert ad.global_range("val") is None

    def test_global_range_cached(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.array([[0.0, 1.0], [2.0, 3.0]])
        r1 = ad.global_range("val")
        r2 = ad.global_range("val")
        assert r1 is r2  # same object from cache

    def test_global_range_invalidated_on_set(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert ad.global_range("val") == (0.0, 3.0)
        ad["val"] = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert ad.global_range("val") == (10.0, 40.0)

    def test_global_range_invalidated_on_delete(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert ad.global_range("val") == (0.0, 3.0)
        del ad["val"]
        ad["val"] = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert ad.global_range("val") == (10.0, 40.0)

    def test_global_labels_2d_categorical(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["site"] = np.array([["alpha", "beta"], ["beta", "gamma"]], dtype=object)
        assert ad.global_labels("site") == ["alpha", "beta", "gamma"]

    def test_global_labels_1d_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["site"] = np.array(["alpha", "beta"], dtype=object)
        assert ad.global_labels("site") is None

    def test_global_labels_numeric_returns_none(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["val"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert ad.global_labels("val") is None

    def test_global_labels_excludes_missing(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad["site"] = np.array([["alpha", None, "beta"],
                                [None, "alpha", None]], dtype=object)
        assert ad.global_labels("site") == ["alpha", "beta"]

    def test_global_labels_cached(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["site"] = np.array([["a", "b"], ["b", "a"]], dtype=object)
        r1 = ad.global_labels("site")
        r2 = ad.global_labels("site")
        assert r1 is r2

    def test_global_labels_invalidated_on_set(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        ad["site"] = np.array([["a", "b"], ["b", "a"]], dtype=object)
        assert ad.global_labels("site") == ["a", "b"]
        ad["site"] = np.array([["x", "y"], ["y", "x"]], dtype=object)
        assert ad.global_labels("site") == ["x", "y"]

    def test_global_range_partial_nan(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad["val"] = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])
        assert ad.global_range("val") == (1.0, 5.0)

    def test_negative_n_atoms_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_atom_data(n_atoms=-1, n_frames=1)

    def test_dynamic_frame_count(self):
        """AtomData tracks the live frame list length."""
        frames: list = [None]  # type: ignore[list-item]
        ad = AtomData(n_atoms=2, frames=frames)
        assert ad.n_frames == 1
        frames.append(None)
        assert ad.n_frames == 2
        # 2-D array with 2 rows now valid.
        ad["val"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert ad["val"].shape == (2, 2)

    def test_setitem_1d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad["charge"] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0] = 99.0

    def test_setitem_2d_stored_array_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=2)
        ad["charge"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="read-only"):
            ad["charge"][0, 0] = 99.0

    def test_setitem_local_reference_is_read_only(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad["charge"] = np.array([1.0, 2.0, 3.0])
        arr = ad["charge"]
        with pytest.raises(ValueError, match="read-only"):
            arr[...] = 0.0

    def test_setitem_does_not_freeze_caller_source(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        src = np.array([1.0, 2.0, 3.0])
        ad["charge"] = src
        # Source is still writable.
        assert src.flags.writeable is True
        # Mutating the source does not affect the stored array.
        src[0] = 99.0
        assert ad["charge"][0] == 1.0

    def test_setitem_reassignment_replaces_values(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        ad["charge"] = np.array([1.0, 2.0, 3.0])
        ad["charge"] = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(ad["charge"], [4.0, 5.0, 6.0])
