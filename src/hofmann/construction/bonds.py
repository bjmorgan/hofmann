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

    For each pair of atoms, checks all bond specs in order to find the
    first matching rule where the interatomic distance falls within
    ``[min_length, max_length]``.

    When *lattice* is provided, all images in the ``{-1, 0, 1}^3``
    neighbourhood are checked so that bonds across periodic boundaries
    are found correctly — including cases where the same atom pair is
    bonded through more than one image (e.g. atoms sitting on opposite
    cell faces at half a lattice parameter apart).

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

    # Pre-compute unique species for efficient matching.
    unique_species = list(set(species))

    # Vectorised pairwise difference vectors: diff[i,j] = coords[i] - coords[j].
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    if lattice is not None:
        return _compute_bonds_periodic(
            species, diff, bond_specs, lattice, unique_species, n_atoms,
        )
    else:
        return _compute_bonds_direct(
            species, diff, bond_specs, unique_species, n_atoms,
        )


def _compute_bonds_direct(
    species: list[str],
    diff: np.ndarray,
    bond_specs: list[BondSpec],
    unique_species: list[str],
    n_atoms: int,
) -> list[Bond]:
    """Compute bonds for a non-periodic structure."""
    dist_matrix = np.linalg.norm(diff, axis=2)
    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    claimed = np.zeros((n_atoms, n_atoms), dtype=bool)

    bonds: list[Bond] = []

    for spec in bond_specs:
        pair_mask = _species_pair_mask(spec, species, unique_species)
        dist_ok = (
            (dist_matrix >= spec.min_length)
            & (dist_matrix <= spec.max_length)
        )
        hits = upper & pair_mask & dist_ok & ~claimed
        ii, jj = np.nonzero(hits)
        if len(ii) > 0:
            claimed[ii, jj] = True
            for idx in range(len(ii)):
                i, j = int(ii[idx]), int(jj[idx])
                bonds.append(
                    Bond(i, j, float(dist_matrix[i, j]), spec)
                )

    return bonds


def _compute_bonds_periodic(
    species: list[str],
    diff: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    unique_species: list[str],
    n_atoms: int,
) -> list[Bond]:
    """Compute bonds for a periodic structure.

    Checks all 27 image offsets in ``{-1, 0, 1}^3`` so that bonds
    through multiple images of the same atom pair are found correctly.
    Self-bonds (atom to its own periodic image) are included.
    """
    lattice = np.asarray(lattice, dtype=float)
    lat_inv = np.linalg.inv(lattice)
    diff_frac = diff @ lat_inv  # (n, n, 3) raw fractional differences

    # All 27 image offsets.
    img_offsets = np.array([
        (n1, n2, n3)
        for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) for n3 in (-1, 0, 1)
    ])  # (27, 3)

    # Distance matrices for each image: (27, n, n).
    diff_shifted = (
        diff_frac[np.newaxis, :, :, :]
        - img_offsets[:, np.newaxis, np.newaxis, :]
    )  # (27, n, n, 3)
    diff_cart = diff_shifted @ lattice  # (27, n, n, 3)
    dist_all = np.linalg.norm(diff_cart, axis=3)  # (27, n, n)

    # Pair constraints:
    #   offset (0,0,0): i < j only (no self-bonds, no double-counting).
    #   non-zero offset: i <= j (self-bonds on diagonal, i < j for
    #     inter-atom image bonds; reverse pair at opposite offset is
    #     the same physical bond, so i < j avoids double-counting).
    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    upper_diag = upper | np.eye(n_atoms, dtype=bool)

    zero_mask = np.all(img_offsets == 0, axis=1)
    zero_idx = int(np.argmax(zero_mask))

    valid = np.empty((27, n_atoms, n_atoms), dtype=bool)
    valid[:] = upper_diag
    valid[zero_idx] = upper

    # Track claimed (image, i, j) triples — each triple matches at
    # most one spec (first match wins).
    claimed = np.zeros((27, n_atoms, n_atoms), dtype=bool)

    bonds: list[Bond] = []

    for spec in bond_specs:
        pair_mask = _species_pair_mask(spec, species, unique_species)
        dist_ok = (
            (dist_all >= spec.min_length) & (dist_all <= spec.max_length)
        )
        hits = valid & pair_mask[np.newaxis, :, :] & dist_ok & ~claimed

        kk, ii, jj = np.nonzero(hits)
        if len(kk) > 0:
            claimed[kk, ii, jj] = True
            for idx in range(len(kk)):
                k = int(kk[idx])
                i, j = int(ii[idx]), int(jj[idx])
                image = (
                    int(img_offsets[k][0]),
                    int(img_offsets[k][1]),
                    int(img_offsets[k][2]),
                )
                bonds.append(
                    Bond(i, j, float(dist_all[k, i, j]), spec, image=image)
                )

    return bonds


def _species_pair_mask(
    spec: BondSpec,
    species: list[str],
    unique_species: list[str],
) -> np.ndarray:
    """Build a boolean (n, n) mask for species pairs matching *spec*."""
    sp_a, sp_b = spec.species
    match_a = {s for s in unique_species if fnmatch(s, sp_a)}
    match_b = {s for s in unique_species if fnmatch(s, sp_b)}
    mask_a = np.array([s in match_a for s in species])
    mask_b = np.array([s in match_b for s in species])
    return (
        (mask_a[:, np.newaxis] & mask_b[np.newaxis, :])
        | (mask_b[:, np.newaxis] & mask_a[np.newaxis, :])
    )
