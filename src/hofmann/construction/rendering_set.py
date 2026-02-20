"""Rendering set construction from periodic bond data.

Takes periodic bonds (from :func:`compute_bonds` with a lattice) and
produces a flat set of drawable atoms and bonds.  All periodicity
knowledge is consumed here; the renderer downstream sees only plain
indices into coordinate arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np

from hofmann.model import Bond, BondSpec


@dataclass
class RenderingSet:
    """Expanded set of atoms and bonds ready for rendering.

    Attributes:
        species: Species labels for physical + image atoms.
        coords: Coordinates array, shape ``(n_expanded, 3)``.
        bonds: Bonds with indices into the expanded arrays.
            All have ``image == (0, 0, 0)`` — periodicity has been
            resolved into positions.
        source_indices: Maps each expanded atom back to its physical
            atom index.  ``source_indices[:n_physical]`` is
            ``[0, 1, 2, ...]``.
    """

    species: list[str]
    coords: np.ndarray
    bonds: list[Bond]
    source_indices: np.ndarray


def _complete_matches(complete: str | bool, species: str) -> bool:
    """Check whether *complete* selects the given species.

    Args:
        complete: The ``complete`` field from a :class:`BondSpec`.
        species: Species label to test.

    Returns:
        ``True`` if this species should have its shell completed.
    """
    if complete is False:
        return False
    if complete == "*":
        return True
    return fnmatch(species, str(complete))


def build_rendering_set(
    species: list[str],
    coords: np.ndarray,
    periodic_bonds: list[Bond],
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    max_recursive_depth: int = 5,
) -> RenderingSet:
    """Build the expanded rendering set from periodic bond data.

    Takes physical atoms and their periodic bonds (including cross-
    boundary bonds with non-zero ``image`` fields) and produces an
    expanded set of atoms and bonds suitable for rendering.  Image
    atoms are materialised according to the ``complete`` and
    ``recursive`` settings on each bond spec.

    Args:
        species: Species labels for the physical atoms.
        coords: Physical atom coordinates, shape ``(n_physical, 3)``.
        periodic_bonds: Bonds from :func:`compute_bonds` (may include
            non-zero ``image`` fields).
        bond_specs: Bond specification rules (for ``complete`` /
            ``recursive`` settings).
        lattice: 3x3 matrix of lattice vectors (row vectors).
        max_recursive_depth: Maximum iterations for recursive
            expansion.

    Returns:
        A :class:`RenderingSet` with expanded atoms and bonds.
    """
    coords = np.asarray(coords, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    n_physical = len(species)

    # Partition bonds into direct and periodic.
    direct_bonds: list[Bond] = []
    periodic: list[Bond] = []
    for bond in periodic_bonds:
        if bond.image == (0, 0, 0):
            direct_bonds.append(bond)
        else:
            periodic.append(bond)

    # Image atom registry: (physical_index, image_tuple) → expanded index.
    image_registry: dict[tuple[int, tuple[int, int, int]], int] = {}
    image_species: list[str] = []
    image_coords_list: list[np.ndarray] = []
    image_source: list[int] = []

    def _materialise(phys_idx: int, image: tuple[int, int, int]) -> int:
        """Ensure an image atom exists and return its expanded index."""
        key = (phys_idx, image)
        if key in image_registry:
            return image_registry[key]
        idx = n_physical + len(image_species)
        image_registry[key] = idx
        image_species.append(species[phys_idx])
        offset = np.array(image, dtype=float) @ lattice
        image_coords_list.append(coords[phys_idx] + offset)
        image_source.append(phys_idx)
        return idx

    # --- Single-pass completion (complete set, recursive=False) ---
    rendering_bonds: list[Bond] = list(direct_bonds)

    for bond in periodic:
        spec = bond.spec
        if spec.recursive:
            continue
        if spec.complete is False:
            continue

        a, b = bond.index_a, bond.index_b
        img = bond.image

        # complete matches species[a] → materialise image of b
        if _complete_matches(spec.complete, species[a]):
            b_img_idx = _materialise(b, img)
            rendering_bonds.append(
                Bond(a, b_img_idx, bond.length, spec)
            )

        # complete matches species[b] → materialise image of a
        neg_img = tuple(-x for x in img)
        if _complete_matches(spec.complete, species[b]):
            a_img_idx = _materialise(a, neg_img)
            rendering_bonds.append(
                Bond(b, a_img_idx, bond.length, spec)
            )

    # --- Recursive expansion (recursive=True specs only) ---
    recursive_specs = [s for s in bond_specs if s.recursive]
    if recursive_specs:
        # Build a lookup: for each physical atom, its periodic bonds
        # via recursive specs.
        # atom_periodic_bonds[phys_idx] = [(other_phys_idx, image, spec), ...]
        atom_periodic_bonds: dict[
            int, list[tuple[int, tuple[int, int, int], BondSpec]]
        ] = {}
        for bond in periodic_bonds:
            if not bond.spec.recursive:
                continue
            a, b = bond.index_a, bond.index_b
            img = bond.image
            atom_periodic_bonds.setdefault(a, []).append((b, img, bond.spec))
            # Reverse direction: b sees a at -image
            neg_img = tuple(-x for x in img)
            atom_periodic_bonds.setdefault(b, []).append(
                (a, neg_img, bond.spec)
            )

        # Seed: physical atoms are at shift (0, 0, 0).
        # atom_shifts maps (phys_idx, shift) → expanded index.
        atom_shifts: dict[tuple[int, tuple[int, int, int]], int] = {}
        for i in range(n_physical):
            atom_shifts[(i, (0, 0, 0))] = i

        # Also include image atoms already materialised (from
        # non-recursive complete) — but those aren't relevant here
        # since recursive specs skip the complete path above.

        # Queue of atoms to process: (physical_index, shift, expanded_index)
        queue: list[tuple[int, tuple[int, int, int], int]] = [
            (i, (0, 0, 0), i) for i in range(n_physical)
        ]

        for _ in range(max_recursive_depth):
            next_queue: list[tuple[int, tuple[int, int, int], int]] = []
            for phys_idx, shift, exp_idx in queue:
                if phys_idx not in atom_periodic_bonds:
                    continue
                for other_phys, bond_img, spec in atom_periodic_bonds[phys_idx]:
                    # Target shift = current shift + bond image
                    target_shift = tuple(
                        s + i for s, i in zip(shift, bond_img)
                    )
                    target_key = (other_phys, target_shift)

                    if target_key in atom_shifts:
                        # Target already exists — just add the bond.
                        target_idx = atom_shifts[target_key]
                    else:
                        # Materialise the target.
                        if target_shift == (0, 0, 0):
                            # It's a physical atom — already present.
                            target_idx = other_phys
                            atom_shifts[target_key] = target_idx
                        else:
                            target_idx = _materialise(other_phys, target_shift)
                            atom_shifts[target_key] = target_idx
                            next_queue.append(
                                (other_phys, target_shift, target_idx)
                            )

                    # Add the bond (deduplicate by canonical ordering).
                    a_idx = min(exp_idx, target_idx)
                    b_idx = max(exp_idx, target_idx)
                    if a_idx != b_idx:
                        # Compute length from expanded coordinates.
                        a_coord = (
                            coords[phys_idx]
                            + np.array(shift, dtype=float) @ lattice
                            if shift != (0, 0, 0)
                            else coords[phys_idx]
                        )
                        b_coord = (
                            coords[other_phys]
                            + np.array(target_shift, dtype=float) @ lattice
                            if target_shift != (0, 0, 0)
                            else coords[other_phys]
                        )
                        length = float(np.linalg.norm(b_coord - a_coord))
                        rendering_bonds.append(
                            Bond(a_idx, b_idx, length, spec)
                        )

            if not next_queue:
                break
            queue = next_queue

    # --- Build output ---
    if image_species:
        expanded_species = list(species) + image_species
        expanded_coords = np.vstack([coords] + [
            c[np.newaxis, :] for c in image_coords_list
        ])
        source_indices = np.concatenate([
            np.arange(n_physical),
            np.array(image_source, dtype=int),
        ])
    else:
        expanded_species = list(species)
        expanded_coords = coords.copy()
        source_indices = np.arange(n_physical)

    # Deduplicate rendering bonds.
    seen: set[tuple[int, int]] = set()
    unique_bonds: list[Bond] = []
    for bond in rendering_bonds:
        key = (min(bond.index_a, bond.index_b),
               max(bond.index_a, bond.index_b))
        if key in seen:
            continue
        seen.add(key)
        unique_bonds.append(bond)

    return RenderingSet(
        species=expanded_species,
        coords=expanded_coords,
        bonds=unique_bonds,
        source_indices=source_indices,
    )
