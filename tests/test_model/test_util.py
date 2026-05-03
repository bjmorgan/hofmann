"""Tests for hofmann.model._util helpers."""

from hofmann.model._util import _site_species
from hofmann.model.composition import Composition


class TestSiteSpecies:
    def test_string_returns_singleton_frozenset(self):
        result = _site_species("Fe")
        assert result == frozenset({"Fe"})
        assert isinstance(result, frozenset)

    def test_pure_composition(self):
        result = _site_species(Composition({"Fe": 1.0}))
        assert result == frozenset({"Fe"})

    def test_mixed_composition(self):
        result = _site_species(Composition({"Fe": 0.7, "Mn": 0.3}))
        assert result == frozenset({"Fe", "Mn"})

    def test_partial_composition_excludes_vacancy(self):
        result = _site_species(Composition({"Fe": 0.7}))
        assert result == frozenset({"Fe"})
