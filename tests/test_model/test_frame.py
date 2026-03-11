"""Tests for Frame coordinate validation and dtype coercion."""

import numpy as np
import pytest

from hofmann.model.frame import Frame


class TestFrame:
    def test_valid_coords(self):
        coords = np.zeros((5, 3))
        frame = Frame(coords=coords, label="test")
        assert frame.coords.shape == (5, 3)

    def test_invalid_1d_raises(self):
        with pytest.raises(ValueError, match="\\(n_atoms, 3\\)"):
            Frame(coords=np.zeros(6))

    def test_invalid_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="\\(n_atoms, 3\\)"):
            Frame(coords=np.zeros((5, 2)))

    def test_coords_converted_to_float(self):
        coords = np.array([[1, 2, 3]], dtype=int)
        frame = Frame(coords=coords)
        assert frame.coords.dtype == float

    def test_lattice_default_none(self):
        frame = Frame(coords=np.zeros((1, 3)))
        assert frame.lattice is None

    def test_lattice_accepted(self):
        lat = np.eye(3) * 5.0
        frame = Frame(coords=np.zeros((1, 3)), lattice=lat)
        np.testing.assert_array_equal(frame.lattice, lat)

    def test_lattice_bad_shape_raises(self):
        with pytest.raises(ValueError, match="\\(3, 3\\)"):
            Frame(coords=np.zeros((1, 3)), lattice=np.eye(2))

    def test_lattice_coerced_to_float(self):
        lat_int = np.eye(3, dtype=int) * 5
        frame = Frame(coords=np.zeros((1, 3)), lattice=lat_int)
        assert frame.lattice.dtype == float
