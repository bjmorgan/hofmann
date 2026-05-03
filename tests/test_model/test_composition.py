"""Tests for the Composition value type."""

import pytest

from hofmann.model.composition import Composition


class TestCompositionBasic:
    def test_construct_from_dict(self):
        c = Composition({"Fe": 0.7, "Mn": 0.3})
        assert c["Fe"] == 0.7
        assert c["Mn"] == 0.3

    def test_len(self):
        c = Composition({"Fe": 0.7, "Mn": 0.3})
        assert len(c) == 2

    def test_iter_yields_species(self):
        c = Composition({"Fe": 0.7, "Mn": 0.3})
        assert set(iter(c)) == {"Fe", "Mn"}

    def test_contains(self):
        c = Composition({"Fe": 0.7, "Mn": 0.3})
        assert "Fe" in c
        assert "Cu" not in c

    def test_is_frozen(self):
        c = Composition({"Fe": 1.0})
        with pytest.raises(TypeError):
            c["Fe"] = 0.5  # type: ignore[index]
