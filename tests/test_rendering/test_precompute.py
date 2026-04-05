"""Unit tests for helpers in ``hofmann.rendering.precompute``."""

from __future__ import annotations

import numpy as np

from hofmann._constants import DEFAULT_ATOM_RADIUS
from hofmann.model import AtomStyle
from hofmann.rendering.precompute import _compute_atom_radii


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
