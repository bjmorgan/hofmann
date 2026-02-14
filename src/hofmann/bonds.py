"""Bond computation from declarative BondSpec rules."""

from __future__ import annotations

import numpy as np

from hofmann.model import Bond, BondSpec


def compute_bonds(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
) -> list[Bond]:
    """Compute bonds for a single frame based on bond specification rules.

    For each pair of atoms (i < j), checks all bond specs in order to
    find the first matching rule where the interatomic distance falls
    within ``[min_length, max_length]``.

    Args:
        species: List of species labels, length ``n_atoms``.
        coords: Coordinates array of shape ``(n_atoms, 3)``.
        bond_specs: List of BondSpec rules to apply.

    Returns:
        List of Bond objects for all detected bonds.
    """
    if len(species) == 0 or len(bond_specs) == 0:
        return []

    coords = np.asarray(coords, dtype=float)
    n_atoms = len(species)

    # Vectorised pairwise distance matrix.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    bonds: list[Bond] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = dist_matrix[i, j]
            for spec in bond_specs:
                if (
                    spec.matches(species[i], species[j])
                    and spec.min_length <= dist <= spec.max_length
                ):
                    bonds.append(Bond(i, j, float(dist), spec))
                    break  # First matching spec wins.

    return bonds
