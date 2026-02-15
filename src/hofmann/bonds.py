"""Bond computation from declarative BondSpec rules."""

from __future__ import annotations

from fnmatch import fnmatch

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

    Species matching is pre-computed per spec so that the inner loop
    over atom pairs is a vectorised numpy operation rather than
    per-pair Python calls.

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

    # Upper-triangle mask: only consider pairs (i < j).
    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)

    # Track which pairs have been claimed by an earlier spec.
    claimed = np.zeros((n_atoms, n_atoms), dtype=bool)

    # Pre-compute unique species for efficient matching.
    unique_species = list(set(species))

    bonds: list[Bond] = []

    for spec in bond_specs:
        # Determine which unique species match each side of the spec.
        sp_a, sp_b = spec.species
        match_a = {s for s in unique_species
                   if fnmatch(s, sp_a)}
        match_b = {s for s in unique_species
                   if fnmatch(s, sp_b)}

        # Boolean masks: which atoms match side a / side b.
        mask_a = np.array([s in match_a for s in species])
        mask_b = np.array([s in match_b for s in species])

        # Species pair mask (symmetric matching): (a[i] & b[j]) | (b[i] & a[j]).
        pair_mask = (
            (mask_a[:, np.newaxis] & mask_b[np.newaxis, :])
            | (mask_b[:, np.newaxis] & mask_a[np.newaxis, :])
        )

        # Distance filter.
        dist_ok = (
            (dist_matrix >= spec.min_length)
            & (dist_matrix <= spec.max_length)
        )

        # Combine: upper triangle, species match, distance match, not claimed.
        hits = upper & pair_mask & dist_ok & ~claimed

        # Extract matching pairs.
        ii, jj = np.nonzero(hits)
        if len(ii) > 0:
            claimed[ii, jj] = True
            for idx in range(len(ii)):
                i, j = int(ii[idx]), int(jj[idx])
                bonds.append(Bond(i, j, float(dist_matrix[i, j]), spec))

    return bonds
