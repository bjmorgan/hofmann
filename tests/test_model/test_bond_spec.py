"""Tests for BondSpec matching, defaults, repr, equality, validation, and Bond."""

import pytest

from hofmann.model.bond_spec import Bond, BondSpec


class TestBondSpecMatches:
    def _spec(self, sp_a: str, sp_b: str) -> BondSpec:
        """Create a BondSpec with dummy geometry values."""
        return BondSpec(species=(sp_a, sp_b), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0)

    def test_exact_match(self):
        assert self._spec("C", "H").matches("C", "H") is True

    def test_symmetric_match(self):
        assert self._spec("C", "H").matches("H", "C") is True

    def test_no_match(self):
        assert self._spec("C", "H").matches("O", "N") is False

    def test_wildcard_star(self):
        assert self._spec("*", "H").matches("C", "H") is True

    def test_wildcard_star_symmetric(self):
        assert self._spec("*", "H").matches("H", "O") is True

    def test_wildcard_question_mark(self):
        spec = self._spec("C?", "H")
        assert spec.matches("Cu", "H") is True
        assert spec.matches("C", "H") is False  # "C" is 1 char, "C?" needs 2

    def test_both_wildcard(self):
        assert self._spec("*", "*").matches("X", "Y") is True

    def test_species_sorted(self):
        """Species tuple should be stored in sorted order."""
        spec = self._spec("H", "C")
        assert spec.species == ("C", "H")

    def test_species_already_sorted(self):
        """Already-sorted species should be unchanged."""
        spec = self._spec("C", "H")
        assert spec.species == ("C", "H")

    def test_complete_true_raises(self):
        """complete=True is not allowed -- must be a species string or '*'."""
        with pytest.raises(ValueError, match="complete=True is not supported"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete=True)

    def test_complete_non_string_truthy_raises(self):
        """Non-string truthy values like 1 are rejected."""
        with pytest.raises(ValueError, match="complete must be"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete=1)  # type: ignore[arg-type]

    def test_complete_string_accepted(self):
        """A species name string is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="C")
        assert spec.complete == "C"

    def test_complete_wildcard_accepted(self):
        """'*' is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="*")
        assert spec.complete == "*"

    def test_complete_false_accepted(self):
        """False (default) is valid for complete."""
        spec = BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete=False)
        assert spec.complete is False

    def test_complete_empty_string_raises(self):
        """An empty string is not a valid species name for complete."""
        with pytest.raises(ValueError, match="complete must not be an empty string"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete="")

    def test_complete_species_not_in_pair_raises(self):
        """complete='Zr' on a ('C', 'H') bond spec is rejected."""
        with pytest.raises(ValueError, match="does not match either species"):
            BondSpec(species=("C", "H"), min_length=0.0,
                     max_length=5.0, radius=0.1, colour=1.0,
                     complete="Zr")

    def test_complete_species_matches_one_side(self):
        """complete='Na' on ('Cl', 'Na') is accepted."""
        spec = BondSpec(species=("Na", "Cl"), min_length=0.0,
                        max_length=5.0, radius=0.1, colour=1.0,
                        complete="Na")
        assert spec.complete == "Na"


class TestBondSpecDefaults:
    def test_min_length_defaults_to_zero(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.min_length == 0.0

    def test_radius_defaults_to_class_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.radius == 0.1

    def test_colour_defaults_to_class_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec.colour == 0.5

    def test_explicit_radius_overrides_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        assert spec.radius == 0.2

    def test_explicit_colour_overrides_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, colour="red")
        assert spec.colour == "red"

    def test_changing_class_default_radius(self):
        original = BondSpec.default_radius
        try:
            BondSpec.default_radius = 0.15
            spec = BondSpec(species=("C", "H"), max_length=3.4)
            assert spec.radius == 0.15
        finally:
            BondSpec.default_radius = original

    def test_changing_class_default_does_not_affect_explicit(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        original = BondSpec.default_radius
        try:
            BondSpec.default_radius = 0.99
            assert spec.radius == 0.2
        finally:
            BondSpec.default_radius = original

    def test_radius_setter(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        spec.radius = 0.3
        assert spec.radius == 0.3

    def test_radius_setter_none_reverts_to_default(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.3)
        spec.radius = None
        assert spec.radius == 0.1

    def test_radius_setter_rejects_negative(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        with pytest.raises(ValueError, match="radius"):
            spec.radius = -0.1

    def test_colour_setter(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        spec.colour = "blue"
        assert spec.colour == "blue"

    def test_colour_setter_rejects_invalid(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        with pytest.raises(ValueError, match="colour"):
            spec.colour = "not_a_colour"


class TestBondSpecRepr:
    def test_default_radius_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "radius=<default 0.1>" in repr(spec)

    def test_default_colour_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "colour=<default 0.5>" in repr(spec)

    def test_explicit_radius_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2)
        assert "radius=0.2" in repr(spec)

    def test_explicit_colour_shown(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, colour="red")
        assert "colour='red'" in repr(spec)

    def test_complete_omitted_when_false(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "complete" not in repr(spec)

    def test_recursive_omitted_when_false(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert "recursive" not in repr(spec)

    def test_complete_shown_when_set(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4, complete="C")
        assert "complete='C'" in repr(spec)


class TestBondSpecEquality:
    def test_equal_with_defaults(self):
        a = BondSpec(species=("C", "H"), max_length=3.4)
        b = BondSpec(species=("C", "H"), max_length=3.4)
        assert a == b

    def test_equal_with_explicit_values(self):
        a = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2, colour="red")
        b = BondSpec(species=("C", "H"), max_length=3.4, radius=0.2, colour="red")
        assert a == b

    def test_default_not_equal_to_explicit_same_value(self):
        a = BondSpec(species=("C", "H"), max_length=3.4)
        b = BondSpec(species=("C", "H"), max_length=3.4, radius=0.1)
        assert a != b

    def test_not_equal_to_other_type(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        assert spec != "not a spec"

    def test_unhashable(self):
        spec = BondSpec(species=("C", "H"), max_length=3.4)
        with pytest.raises(TypeError, match="unhashable"):
            hash(spec)


class TestBondSpecValidation:
    def test_negative_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            BondSpec(species=("C", "H"), max_length=-1.0)

    def test_zero_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            BondSpec(species=("C", "H"), max_length=0.0)

    def test_negative_min_length_raises(self):
        with pytest.raises(ValueError, match="min_length"):
            BondSpec(species=("C", "H"), max_length=3.0, min_length=-0.5)

    def test_min_length_exceeds_max_length_raises(self):
        with pytest.raises(ValueError, match="min_length"):
            BondSpec(species=("C", "H"), max_length=2.0, min_length=3.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            BondSpec(species=("C", "H"), max_length=3.0, radius=-0.1)

    def test_invalid_colour_raises(self):
        with pytest.raises(ValueError, match="colour"):
            BondSpec(species=("C", "H"), max_length=3.0, colour="not_a_colour")

    def test_valid_spec_accepted(self):
        spec = BondSpec(species=("C", "H"), max_length=3.0, min_length=0.5, radius=0.1)
        assert spec.max_length == 3.0


class TestBond:
    def _spec(self) -> BondSpec:
        return BondSpec(species=("C", "H"), min_length=0.0,
                        max_length=3.0, radius=0.1, colour=1.0)

    def test_is_frozen(self):
        bond = Bond(0, 1, 2.0, self._spec())
        with pytest.raises(AttributeError):
            bond.length = 3.0  # type: ignore[misc]

    def test_default_image_is_origin(self):
        bond = Bond(0, 1, 2.0, self._spec())
        assert bond.image == (0, 0, 0)

    def test_explicit_image(self):
        bond = Bond(0, 1, 2.0, self._spec(), image=(1, 0, 0))
        assert bond.image == (1, 0, 0)

    def test_equality_includes_image(self):
        spec = self._spec()
        bond_a = Bond(0, 1, 2.0, spec, image=(0, 0, 0))
        bond_b = Bond(0, 1, 2.0, spec, image=(1, 0, 0))
        assert bond_a != bond_b

    def test_equality_same_image(self):
        spec = self._spec()
        bond_a = Bond(0, 1, 2.0, spec, image=(1, 0, 0))
        bond_b = Bond(0, 1, 2.0, spec, image=(1, 0, 0))
        assert bond_a == bond_b

    def test_hash_includes_image(self):
        spec = self._spec()
        bond_a = Bond(0, 1, 2.0, spec, image=(0, 0, 0))
        bond_b = Bond(0, 1, 2.0, spec, image=(1, 0, 0))
        assert hash(bond_a) != hash(bond_b)

    def test_hashable(self):
        bond = Bond(0, 1, 2.0, self._spec(), image=(1, 0, 0))
        {bond}  # should not raise
