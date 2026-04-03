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

    def test_len(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        assert len(ad) == 0
        ad["a"] = [1.0, 2.0]
        ad["b"] = [3.0, 4.0]
        assert len(ad) == 2

    def test_iter(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["a"] = [1.0, 2.0]
        ad["b"] = [3.0, 4.0]
        assert set(ad) == {"a", "b"}

    def test_items(self):
        ad = _make_atom_data(n_atoms=2, n_frames=1)
        ad["val"] = [1.0, 2.0]
        keys = [k for k, v in ad.items()]
        assert keys == ["val"]

    def test_single_frame_2d_accepted(self):
        ad = _make_atom_data(n_atoms=3, n_frames=1)
        vals = np.array([[1.0, 2.0, 3.0]])
        ad["q"] = vals
        assert ad["q"].shape == (1, 3)

    def test_3d_rejected(self):
        ad = _make_atom_data(n_atoms=2, n_frames=2)
        with pytest.raises(ValueError):
            ad["bad"] = np.ones((2, 2, 2))

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
