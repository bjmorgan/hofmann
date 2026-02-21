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

    When *lattice* is provided, bonds across periodic boundaries are
    found correctly.  If all bond lengths are shorter than the
    inscribed sphere radius of the cell, the minimum image convention
    (MIC) is used for an efficient O(n²) computation.  Otherwise, all
    27 images in ``{-1, 0, 1}^3`` are checked iteratively to handle
    multi-image bonds (e.g. atoms on opposite cell faces at half a
    lattice parameter apart).

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


def _inscribed_sphere_radius(lattice: np.ndarray) -> float:
    """Radius of the largest sphere fitting inside the unit cell.

    For lattice vectors **a**, **b**, **c** (rows of *lattice*), this
    is ``min(h_a, h_b, h_c) / 2`` where each ``h_i`` is the
    perpendicular distance between the pair of faces normal to the
    *i*-th reciprocal lattice direction.
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    volume = abs(np.dot(a, np.cross(b, c)))
    heights = np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(a, c)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])
    return float(heights.min() / 2.0)


def _compute_bonds_periodic(
    species: list[str],
    diff: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    unique_species: list[str],
    n_atoms: int,
) -> list[Bond]:
    """Compute bonds for a periodic structure.

    Uses a two-tier approach to minimise memory:

    * **MIC fast path** — when the longest bond spec is shorter than
      the inscribed sphere radius, the minimum image convention
      guarantees at most one image per pair.  Memory: O(n²).
    * **Multi-image slow path** — when bond lengths are comparable to
      cell dimensions, all 27 images in ``{-1, 0, 1}^3`` are checked
      by iterating one offset at a time.  Memory: O(n²) per
      iteration.  Self-bonds (atom to its own periodic image) are
      found on this path.
    """
    lattice = np.asarray(lattice, dtype=float)
    lat_inv = np.linalg.inv(lattice)
    diff_frac = diff @ lat_inv  # (n, n, 3) fractional differences

    max_bond = max(s.max_length for s in bond_specs)
    r_ins = _inscribed_sphere_radius(lattice)

    if max_bond < r_ins:
        return _compute_bonds_mic(
            species, diff_frac, bond_specs, lattice,
            unique_species, n_atoms,
        )
    return _compute_bonds_multi_image(
        species, diff_frac, bond_specs, lattice,
        unique_species, n_atoms,
    )


def _compute_bonds_mic(
    species: list[str],
    diff_frac: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    unique_species: list[str],
    n_atoms: int,
) -> list[Bond]:
    """MIC fast path: one image per pair, no self-bonds possible."""
    images = np.rint(diff_frac).astype(int)  # (n, n, 3)
    mic_frac = diff_frac - images  # (n, n, 3)
    mic_cart = mic_frac @ lattice  # (n, n, 3)
    dist = np.linalg.norm(mic_cart, axis=2)  # (n, n)

    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    claimed = np.zeros((n_atoms, n_atoms), dtype=bool)

    bonds: list[Bond] = []

    for spec in bond_specs:
        pair_mask = _species_pair_mask(spec, species, unique_species)
        dist_ok = (dist >= spec.min_length) & (dist <= spec.max_length)
        hits = upper & pair_mask & dist_ok & ~claimed
        ii, jj = np.nonzero(hits)
        if len(ii) > 0:
            claimed[ii, jj] = True
            for idx in range(len(ii)):
                i, j = int(ii[idx]), int(jj[idx])
                image = (
                    int(images[i, j, 0]),
                    int(images[i, j, 1]),
                    int(images[i, j, 2]),
                )
                bonds.append(
                    Bond(i, j, float(dist[i, j]), spec, image=image)
                )

    return bonds


def _compute_bonds_multi_image(
    species: list[str],
    diff_frac: np.ndarray,
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    unique_species: list[str],
    n_atoms: int,
) -> list[Bond]:
    """Multi-image slow path: iterate over 27 offsets one at a time.

    Checks all ``{-1, 0, 1}^3`` image offsets so that bonds through
    multiple images of the same atom pair are found correctly,
    including self-bonds (atom to its own periodic image).

    Peak memory is O(n²) per iteration rather than O(27 n²) for the
    fully vectorised approach.
    """
    img_offsets = np.array([
        (n1, n2, n3)
        for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) for n3 in (-1, 0, 1)
    ])  # (27, 3)

    # Pair constraints:
    #   offset (0,0,0): i < j only (no self-bonds, no double-counting).
    #   non-zero offset: i <= j (self-bonds on diagonal; reverse pair
    #     at opposite offset is the same physical bond).
    upper = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
    upper_diag = upper | np.eye(n_atoms, dtype=bool)

    # Pre-compute pair masks for all specs.
    pair_masks = [
        _species_pair_mask(spec, species, unique_species)
        for spec in bond_specs
    ]

    # Per-image claimed arrays (booleans — negligible vs float64).
    claimed = [
        np.zeros((n_atoms, n_atoms), dtype=bool)
        for _ in range(len(img_offsets))
    ]

    bonds: list[Bond] = []

    for k, offset in enumerate(img_offsets):
        is_zero = not offset.any()
        valid = upper if is_zero else upper_diag

        shifted = diff_frac - offset  # (n, n, 3)
        cart = shifted @ lattice  # (n, n, 3)
        dist = np.linalg.norm(cart, axis=2)  # (n, n)

        for s_idx, spec in enumerate(bond_specs):
            dist_ok = (dist >= spec.min_length) & (dist <= spec.max_length)
            hits = valid & pair_masks[s_idx] & dist_ok & ~claimed[k]
            ii, jj = np.nonzero(hits)
            if len(ii) > 0:
                claimed[k][ii, jj] = True
                image = (int(offset[0]), int(offset[1]), int(offset[2]))
                for idx in range(len(ii)):
                    i, j = int(ii[idx]), int(jj[idx])
                    bonds.append(
                        Bond(i, j, float(dist[i, j]), spec, image=image)
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
