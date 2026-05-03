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

    def test_dataclass_attribute_is_frozen(self):
        import dataclasses
        c = Composition({"Fe": 1.0})
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.occupancies = {}  # type: ignore[misc]


class TestCompositionValidation:
    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Composition({"Fe": -0.1})

    def test_value_above_one_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            Composition({"Fe": 1.5})

    def test_sum_above_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            Composition({"Fe": 0.7, "Mn": 0.6})

    def test_sum_at_one_within_tolerance_ok(self):
        c = Composition({"Fe": 0.5000000001, "Mn": 0.5})
        assert pytest.approx(sum(c.values()), abs=1e-9) == 1.0

    def test_zero_values_dropped(self):
        c = Composition({"Fe": 0.7, "Mn": 0.0})
        assert "Mn" not in c
        assert dict(c) == {"Fe": 0.7}

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Composition({})

    def test_all_zero_after_drop_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Composition({"Fe": 0.0, "Mn": 0.0})

    def test_non_string_key_raises(self):
        with pytest.raises(TypeError, match="string"):
            Composition({26: 1.0})  # type: ignore[dict-item]

    def test_nan_value_raises(self):
        with pytest.raises(ValueError, match="finite"):
            Composition({"Fe": float("nan")})

    def test_inf_value_raises(self):
        with pytest.raises(ValueError, match="finite"):
            Composition({"Fe": float("inf")})

    def test_empty_string_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Composition({"": 1.0})

    def test_whitespace_only_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Composition({"   ": 1.0})

    def test_tab_only_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Composition({"\t": 1.0})


class TestCompositionEqualityAndOrder:
    def test_equality_independent_of_insertion_order(self):
        a = Composition({"Fe": 0.7, "Mn": 0.3})
        b = Composition({"Mn": 0.3, "Fe": 0.7})
        assert a == b
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        a = Composition({"Fe": 0.7, "Mn": 0.3})
        d = {a: "value"}
        assert d[Composition({"Mn": 0.3, "Fe": 0.7})] == "value"

    def test_iter_order_descending_occupancy(self):
        c = Composition({"Mn": 0.3, "Fe": 0.7})
        assert list(c) == ["Fe", "Mn"]

    def test_iter_order_alphabetical_tiebreak(self):
        c = Composition({"Mn": 0.5, "Fe": 0.5})
        assert list(c) == ["Fe", "Mn"]

    def test_iter_order_three_species(self):
        c = Composition({"O": 0.2, "Fe": 0.3, "Mn": 0.5})
        assert list(c) == ["Mn", "Fe", "O"]


class TestCompositionAccessors:
    def test_species_returns_frozenset(self):
        c = Composition({"Fe": 0.7, "Mn": 0.3})
        assert c.species == frozenset({"Fe", "Mn"})
        assert isinstance(c.species, frozenset)

    def test_dominant_species_max_occupancy(self):
        assert Composition({"Fe": 0.7, "Mn": 0.3}).dominant_species == "Fe"

    def test_dominant_species_alphabetical_tiebreak(self):
        assert Composition({"Mn": 0.5, "Fe": 0.5}).dominant_species == "Fe"

    def test_vacancy_zero_when_full(self):
        assert Composition({"Fe": 1.0}).vacancy == pytest.approx(0.0)

    def test_vacancy_partial(self):
        assert Composition({"Fe": 0.7}).vacancy == pytest.approx(0.3)

    def test_vacancy_clamped_to_zero_within_tolerance(self):
        c = Composition({"Fe": 0.5, "Mn": 0.5000000001})
        assert c.vacancy == pytest.approx(0.0, abs=1e-9)
