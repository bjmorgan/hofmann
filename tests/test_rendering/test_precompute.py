"""Unit tests for helpers in ``hofmann.rendering.precompute``."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hofmann._constants import DEFAULT_ATOM_RADIUS
from hofmann.model import AtomStyle
from hofmann.model.composition import Composition
from hofmann.rendering.precompute import (
    _compute_atom_radii,
    _warn_missing_atom_styles,
)


class TestComputeAtomRadii:
    """Vectorised per-atom radius lookup from ``scene.atom_styles``."""

    def test_all_species_have_styles(self):
        """Every atom's radius is taken from its species' ``AtomStyle``."""
        species = ["Na", "Cl", "Na", "Cl"]
        atom_styles = {
            "Na": AtomStyle(radius=1.2, colour="purple"),
            "Cl": AtomStyle(radius=0.9, colour="green"),
        }

        radii = _compute_atom_radii(species, atom_styles)

        assert radii.shape == (4,)
        np.testing.assert_array_equal(radii, [1.2, 0.9, 1.2, 0.9])

    def test_unknown_species_fall_back_to_default(self):
        """Atoms whose species has no ``AtomStyle`` use ``DEFAULT_ATOM_RADIUS``."""
        species = ["Na", "Xx", "Cl"]
        atom_styles = {
            "Na": AtomStyle(radius=1.2, colour="purple"),
            "Cl": AtomStyle(radius=0.9, colour="green"),
        }

        radii = _compute_atom_radii(species, atom_styles)

        np.testing.assert_array_equal(radii, [1.2, DEFAULT_ATOM_RADIUS, 0.9])

    def test_all_species_unknown(self):
        """When no species has a style, every atom uses the default."""
        species = ["Aa", "Bb", "Cc"]
        atom_styles: dict[str, AtomStyle] = {}

        radii = _compute_atom_radii(species, atom_styles)

        np.testing.assert_array_equal(
            radii, [DEFAULT_ATOM_RADIUS, DEFAULT_ATOM_RADIUS, DEFAULT_ATOM_RADIUS]
        )

    def test_empty_species_list(self):
        """An empty input yields an empty array, not an error."""
        radii = _compute_atom_radii([], {})
        assert radii.shape == (0,)

    def test_returns_ndarray(self):
        """The return value is a numpy array with ``float64`` dtype."""
        radii = _compute_atom_radii(
            ["Na"], {"Na": AtomStyle(radius=1.2, colour="purple")}
        )
        assert isinstance(radii, np.ndarray)
        assert radii.dtype == np.float64

    def test_single_unique_species_many_atoms(self):
        """A structure with one species type: all atoms share the same radius."""
        species = ["Si"] * 8
        atom_styles = {"Si": AtomStyle(radius=1.11, colour="grey")}

        radii = _compute_atom_radii(species, atom_styles)

        np.testing.assert_array_equal(radii, np.full(8, 1.11))


class TestWarnMissingAtomStyles:
    """Batched warning for species with no ``AtomStyle``."""

    def test_no_warning_when_all_species_have_styles(self):
        """No warning is emitted when every species has an ``AtomStyle``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles(
                ["Na", "Cl"],
                {
                    "Na": AtomStyle(radius=1.2, colour="purple"),
                    "Cl": AtomStyle(radius=0.9, colour="green"),
                },
            )
        assert len(caught) == 0

    def test_single_missing_species(self):
        """One missing species produces one warning naming that species."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles(
                ["Na", "Fe"],
                {"Na": AtomStyle(radius=1.2, colour="purple")},
            )
        assert len(caught) == 1
        assert "'Fe'" in str(caught[0].message)
        assert caught[0].category is UserWarning

    def test_multiple_missing_species_batched_alphabetically(self):
        """Multiple missing species produce a single warning in alphabetical order."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles(
                ["Sb", "Fe", "O", "Na"],
                {"Na": AtomStyle(radius=1.2, colour="purple")},
            )
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "'Fe'" in msg
        assert "'O'" in msg
        assert "'Sb'" in msg
        # Alphabetical ordering.
        assert msg.index("'Fe'") < msg.index("'O'") < msg.index("'Sb'")

    def test_no_warning_for_empty_species(self):
        """An empty species list produces no warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles([], {})
        assert len(caught) == 0

    def test_duplicate_species_listed_once(self):
        """A species repeated in the atom list appears once in the warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles(
                ["Fe", "Fe", "Fe"],
                {},
            )
        assert len(caught) == 1
        assert str(caught[0].message).count("'Fe'") == 1


class TestWarnMissingAtomStylesWithMixed:
    def test_constituent_species_with_style_no_warning(self):
        species = ["Fe", Composition({"Fe": 0.7, "Mn": 0.3})]
        styles = {
            "Fe": AtomStyle(radius=1.0, colour="red"),
            "Mn": AtomStyle(radius=1.0, colour="purple"),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_missing_atom_styles(species, styles)
        assert not any(
            "AtomStyle" in str(w.message) for w in caught
        )

    def test_constituent_species_missing_style_warns(self):
        species = ["Fe", Composition({"Fe": 0.7, "Mn": 0.3})]
        styles = {"Fe": AtomStyle(radius=1.0, colour="red")}
        with pytest.warns(UserWarning, match="'Mn'"):
            _warn_missing_atom_styles(species, styles)


class TestComputeAtomRadiiWithMixed:
    def test_pure_string_unchanged(self):
        species = ["Fe", "O"]
        styles = {
            "Fe": AtomStyle(radius=1.4, colour="red"),
            "O": AtomStyle(radius=0.7, colour="red"),
        }
        radii = _compute_atom_radii(species, styles)
        assert radii[0] == 1.4
        assert radii[1] == 0.7

    def test_mixed_site_weighted_average(self):
        species = [Composition({"Fe": 0.7, "Mn": 0.3})]
        styles = {
            "Fe": AtomStyle(radius=1.4, colour="red"),
            "Mn": AtomStyle(radius=1.0, colour="purple"),
        }
        radii = _compute_atom_radii(species, styles)
        # 0.7 * 1.4 + 0.3 * 1.0 = 1.28; divided by sum_occ (1.0).
        assert radii[0] == pytest.approx(1.28)

    def test_partial_site_normalised_by_occupancy_sum(self):
        # 70% Fe + 30% vacancy: site is drawn at full Fe radius.
        species = [Composition({"Fe": 0.7})]
        styles = {"Fe": AtomStyle(radius=1.4, colour="red")}
        radii = _compute_atom_radii(species, styles)
        assert radii[0] == pytest.approx(1.4)

    def test_missing_constituent_style_falls_back_to_default(self):
        species = [Composition({"Fe": 0.5, "Mn": 0.5})]
        styles = {"Fe": AtomStyle(radius=1.4, colour="red")}
        expected = (0.5 * 1.4 + 0.5 * DEFAULT_ATOM_RADIUS) / 1.0
        radii = _compute_atom_radii(species, styles)
        assert radii[0] == pytest.approx(expected)
