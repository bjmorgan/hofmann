"""Bond computation from declarative BondSpec rules."""

from fnmatch import fnmatch

import numpy as np

from hofmann.model import Bond, BondSpec


def compute_bonds(
    species: list[str],
    coords: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray | None = None,
) -> list[Bond]:
    """Compute bonds for a single frame based on bond specification rules.

    For each pair of atoms (i < j), checks all bond specs in order to
    find the first matching rule where the interatomic distance falls
    within ``[min_length, max_length]``.

    When *lattice* is provided, the minimum image convention is used to
    find the shortest distance between each atom pair across periodic
    images.  Bonds found across a boundary have a non-zero ``image``
    field recording the lattice translation applied to atom b.

    Species matching is pre-computed per spec so that the inner loop
    over atom pairs is a vectorised numpy operation rather than
    per-pair Python calls.

    Args:
        species: List of species labels, length ``n_atoms``.
        coords: Coordinates array of shape ``(n_atoms, 3)``.
        bond_specs: List of BondSpec rules to apply.
        lattice: 3x3 matrix of lattice vectors (row vectors).
            ``None`` for non-periodic scenes.

    Returns:
        List of Bond objects for all detected bonds.
    """
    if len(species) == 0 or len(bond_specs) == 0:
        return []

    coords = np.asarray(coords, dtype=float)
    n_atoms = len(species)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must have 3 columns, got shape {coords.shape}"
        )
    if coords.shape[0] != n_atoms:
        raise ValueError(
            f"species has {n_atoms} entries but coords has "
            f"{coords.shape[0]} rows"
        )

    # Vectorised pairwise difference vectors.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Apply minimum image convention when a lattice is present.
    images: np.ndarray | None = None
    if lattice is not None:
        lattice = np.asarray(lattice, dtype=float)
        lat_inv = np.linalg.inv(lattice)
        diff_frac = diff @ lat_inv
        shifts = np.round(diff_frac)
        images = shifts.astype(int)
        diff_frac -= shifts
        diff = diff_frac @ lattice

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
                image = (
                    tuple(int(x) for x in images[i, j])
                    if images is not None
                    else (0, 0, 0)
                )
                bonds.append(
                    Bond(i, j, float(dist_matrix[i, j]), spec, image=image)
                )

    # Self-bonds: atoms bonding to their own periodic images.
    if lattice is not None:
        bonds.extend(
            _compute_self_bonds(species, lattice, bond_specs)
        )

    return bonds


def _compute_self_bonds(
    species: list[str],
    lattice: np.ndarray,
    bond_specs: list[BondSpec],
) -> list[Bond]:
    """Compute bonds from atoms to their own periodic images.

    For each atom, checks distances to its own images along each
    nearby lattice translation in {-1, 0, 1}^3 (excluding the origin).
    Each image direction produces a distinct bond.

    Args:
        species: List of species labels.
        lattice: 3x3 matrix of lattice vectors (row vectors).
        bond_specs: List of BondSpec rules to apply.

    Returns:
        List of self-bonds.
    """
    offsets = [-1, 0, 1]
    image_vectors = np.array([
        (n1, n2, n3)
        for n1 in offsets for n2 in offsets for n3 in offsets
        if (n1, n2, n3) != (0, 0, 0)
    ])  # shape (26, 3)

    # Distances for each image vector (independent of atom position).
    image_cart = image_vectors @ lattice
    image_distances = np.linalg.norm(image_cart, axis=1)

    bonds: list[Bond] = []
    claimed: set[tuple[int, tuple[int, int, int]]] = set()

    for spec in bond_specs:
        for i, s in enumerate(species):
            if not spec.matches(s, s):
                continue
            for img_idx in range(len(image_vectors)):
                dist = float(image_distances[img_idx])
                if dist < spec.min_length or dist > spec.max_length:
                    continue
                img_tuple = tuple(int(x) for x in image_vectors[img_idx])
                key = (i, img_tuple)
                if key in claimed:
                    continue
                claimed.add(key)
                bonds.append(Bond(i, i, dist, spec, image=img_tuple))

    return bonds
