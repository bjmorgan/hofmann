"""Unit tests for helpers in ``hofmann.rendering.precompute``."""

from __future__ import annotations

import warnings

import numpy as np

from hofmann._constants import DEFAULT_ATOM_RADIUS
from hofmann.model import AtomStyle
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
