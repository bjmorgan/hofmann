"""Convenience constructors for StructureScene."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from hofmann.defaults import default_atom_style, default_bond_specs
from hofmann.model import BondSpec, Frame, PolyhedronSpec, StructureScene, ViewState
from hofmann.parser import parse_bs, parse_mv

if TYPE_CHECKING:
    from pymatgen.core import Structure


def from_xbs(
    bs_path: str | Path,
    mv_path: str | Path | None = None,
) -> StructureScene:
    """Create a StructureScene from XBS .bs (and optional .mv) files.

    Args:
        bs_path: Path to the ``.bs`` file.
        mv_path: Optional path to a ``.mv`` trajectory file.

    Returns:
        A fully configured StructureScene.
    """
    bs_path = Path(bs_path)
    species, frame, atom_styles, bond_specs, polyhedra_specs = parse_bs(
        bs_path,
    )

    if mv_path is not None:
        frames = parse_mv(mv_path, n_atoms=len(species))
    else:
        frames = [frame]

    # Centre the view on the centroid of the first frame.
    centroid = np.mean(frames[0].coords, axis=0)
    view = ViewState(centre=centroid)

    return StructureScene(
        species=species,
        frames=frames,
        atom_styles=atom_styles,
        bond_specs=bond_specs,
        polyhedra=polyhedra_specs,
        view=view,
        title=bs_path.stem,
    )


def _expand_pbc(
    structure: Structure,
    bond_specs: list[BondSpec],
    cutoff: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """Add periodic image atoms beyond the unit cell boundaries.

    For each atom near a cell face (within *cutoff* of the boundary),
    images are placed on the opposite side of the cell.  An atom near
    the origin face (``frac < cutoff``) gets an image shifted by
    ``+1``; an atom near the far face (``frac > 1 - cutoff``) gets an
    image shifted by ``-1``.

    Fractional coordinates are wrapped into ``[0, 1)`` before the
    proximity check, so the result is invariant to whether the input
    structure places boundary atoms at e.g. ``-0.001`` or ``0.999``.

    For atoms near edges or corners, combined shifts along multiple
    axes are generated as well.

    Args:
        structure: A pymatgen ``Structure`` with lattice information.
        bond_specs: Bond specification rules used to derive a default
            cutoff when *cutoff* is ``None``.
        cutoff: Cartesian distance cutoff (angstroms).  Atoms within
            this distance of a cell face get an image on the opposite
            side.  If ``None``, the maximum ``max_length`` from
            *bond_specs* is used.

    Returns:
        Tuple of ``(species, coords)`` with original atoms followed by
        image atoms.
    """
    species_list = [site.specie.symbol for site in structure]

    if cutoff is None:
        if not bond_specs:
            return species_list, structure.cart_coords.copy()
        cutoff = max(spec.max_length for spec in bond_specs)

    lattice = structure.lattice

    # Wrap fractional coordinates into [0, 1) so the check is
    # invariant to how the CIF places boundary atoms.
    frac_coords = structure.frac_coords % 1.0

    # Cartesian coordinates consistent with the wrapped fractional
    # positions (the raw cart_coords may place atoms outside [0, 1)).
    orig_coords = frac_coords @ lattice.matrix

    # Perpendicular distance from the origin to each face.
    inv_matrix = lattice.inv_matrix
    face_dists = 1.0 / np.linalg.norm(inv_matrix, axis=1)

    # Fractional cutoff per axis.
    frac_cutoffs = cutoff / face_dists

    # Per-atom, per-axis shift directions needed.
    # near_lo: atom near the origin face → needs +1 shift.
    # near_hi: atom near the far face   → needs -1 shift.
    near_lo = frac_coords < frac_cutoffs[np.newaxis, :]
    near_hi = frac_coords > (1.0 - frac_cutoffs)[np.newaxis, :]

    # Build image atoms.  For each atom, collect the set of
    # (axis, direction) pairs, then generate all non-empty subsets
    # (to cover face, edge, and corner images).
    lattice_matrix = lattice.matrix
    expanded_species = list(species_list)
    image_coords_list: list[np.ndarray] = []

    for i in range(len(species_list)):
        # Each axis can contribute a +1 shift, a -1 shift, or both.
        axis_shifts: list[list[int]] = []
        for ax in range(3):
            shifts: list[int] = []
            if near_lo[i, ax]:
                shifts.append(+1)
            if near_hi[i, ax]:
                shifts.append(-1)
            axis_shifts.append(shifts)

        if not any(axis_shifts):
            continue

        # Enumerate all combinations: for each axis pick one of its
        # shifts or 0 (no shift), excluding the all-zero combination.
        options = [s + [0] for s in axis_shifts]
        for sa in options[0]:
            for sb in options[1]:
                for sc in options[2]:
                    if sa == 0 and sb == 0 and sc == 0:
                        continue
                    shift = (sa * lattice_matrix[0]
                             + sb * lattice_matrix[1]
                             + sc * lattice_matrix[2])
                    expanded_species.append(species_list[i])
                    image_coords_list.append(orig_coords[i] + shift)

    if image_coords_list:
        expanded_coords = np.vstack([orig_coords, image_coords_list])
    else:
        expanded_coords = orig_coords.copy()

    return expanded_species, expanded_coords


def _merge_expansions(
    base_species: list[str],
    base_coords: np.ndarray,
    extra_species: list[str],
    extra_coords: np.ndarray,
    tol: float = 1e-6,
) -> tuple[list[str], np.ndarray]:
    """Merge two expansions, deduplicating atoms by position.

    Any atom in *extra* that is within *tol* of an existing atom in
    *base* is skipped.

    Args:
        base_species: Species from the primary expansion.
        base_coords: Coordinates from the primary expansion.
        extra_species: Species from the additional expansion.
        extra_coords: Coordinates from the additional expansion.
        tol: Position tolerance for deduplication (angstroms).

    Returns:
        Merged ``(species, coords)`` tuple.
    """
    if len(extra_species) == 0:
        return base_species, base_coords

    new_species = list(base_species)
    # Track all accepted coordinates for incremental deduplication.
    all_coords = base_coords

    for sp, coord in zip(extra_species, extra_coords):
        # Check against base *and* previously accepted extras.
        diffs = np.linalg.norm(all_coords - coord, axis=1)
        if np.any(diffs < tol):
            continue
        new_species.append(sp)
        all_coords = np.vstack([all_coords, coord[np.newaxis, :]])

    return new_species, all_coords


def _expand_bonds(
    structure: Structure,
    bond_specs: list[BondSpec],
) -> tuple[list[str], np.ndarray]:
    """Add periodic image atoms that form valid bonds with unit-cell atoms.

    For each bond spec, finds all periodic neighbours of unit-cell atoms
    within the bond length range.  Any neighbour whose image vector is
    non-zero (i.e. it is a periodic image, not a unit-cell atom) is
    added to the output.

    This search is **non-recursive**: only atoms in the original unit
    cell are used as centres.  Image atoms added here are never
    themselves searched for further neighbours.

    Args:
        structure: A pymatgen ``Structure``.
        bond_specs: Bond specification rules.

    Returns:
        Tuple of ``(species, coords)`` containing **only** the image
        atoms (not the original unit-cell atoms).
    """
    if not bond_specs:
        return [], np.empty((0, 3))

    max_cutoff = max(spec.max_length for spec in bond_specs)
    lattice = structure.lattice

    # get_all_neighbors returns, for each site, a list of
    # PeriodicNeighbor objects with .species_string, .nn_distance,
    # .index (original site index), .image (lattice image vector).
    all_neighbours = structure.get_all_neighbors(max_cutoff)

    # Deduplicate by (site_index, image_tuple).
    seen: set[tuple[int, tuple[int, ...]]] = set()
    image_species: list[str] = []
    image_coords_list: list[np.ndarray] = []

    species_list = [site.specie.symbol for site in structure]

    for i, neighbours in enumerate(all_neighbours):
        sp_i = species_list[i]
        for nbr in neighbours:
            image = tuple(int(x) for x in nbr.image)
            if image == (0, 0, 0):
                continue  # Unit-cell atom, not an image.

            nbr_index = nbr.index
            key = (nbr_index, image)
            if key in seen:
                continue

            sp_j = nbr.specie.symbol
            dist = nbr.nn_distance

            # Check if any bond spec matches this pair.
            matched = False
            for spec in bond_specs:
                if spec.matches(sp_i, sp_j) and spec.min_length <= dist <= spec.max_length:
                    matched = True
                    break

            if not matched:
                continue

            seen.add(key)
            image_species.append(sp_j)
            # Compute Cartesian position of the image.
            image_cart = structure[nbr_index].coords + np.array(image) @ lattice.matrix
            image_coords_list.append(image_cart)

    if image_coords_list:
        return image_species, np.array(image_coords_list)
    return [], np.empty((0, 3))


def _expand_polyhedra_vertices(
    structure: Structure,
    expanded_species: list[str],
    expanded_coords: np.ndarray,
    n_uc: int,
    bond_specs: list[BondSpec],
    polyhedra_specs: list[PolyhedronSpec],
) -> tuple[list[str], np.ndarray]:
    """Add vertex atoms needed by image atoms that are polyhedron centres.

    For each image atom (index >= *n_uc*) whose species matches a
    polyhedron-centre pattern, find its coordination shell by looking up
    the equivalent unit-cell atom's periodic neighbours and shifting them
    by the same lattice translation.  Any neighbour not already present
    in the expanded set is appended.

    This is **non-recursive**: only existing image atoms are checked as
    potential centres; newly added vertex atoms are never themselves
    expanded.

    Args:
        structure: The pymatgen ``Structure`` (single frame).
        expanded_species: Species list after geometric + bond expansion.
        expanded_coords: Coordinates after geometric + bond expansion.
        n_uc: Number of unit-cell atoms (before any expansion).
        bond_specs: Bond specification rules.
        polyhedra_specs: Polyhedron rendering rules (used to identify
            centre species).

    Returns:
        Updated ``(species, coords)`` with additional vertex atoms
        appended.
    """
    if not polyhedra_specs or not bond_specs:
        return expanded_species, expanded_coords

    centre_patterns = [spec.centre for spec in polyhedra_specs]
    lattice = structure.lattice

    # Precompute periodic neighbour data for each unit-cell atom.
    # For each (centre_index, neighbour_index, image_vector) triple
    # that matches a bond spec, record it so we can replicate for images.
    max_cutoff = max(spec.max_length for spec in bond_specs)
    all_neighbours = structure.get_all_neighbors(max_cutoff)
    species_list = [site.specie.symbol for site in structure]

    # For each unit-cell atom, store its bonded neighbours as
    # (neighbour_uc_index, image_vector) pairs with their Cartesian offset.
    uc_neighbour_offsets: dict[int, list[tuple[str, np.ndarray]]] = {}
    for i, neighbours in enumerate(all_neighbours):
        sp_i = species_list[i]
        offsets: list[tuple[str, np.ndarray]] = []
        for nbr in neighbours:
            sp_j = nbr.specie.symbol
            dist = nbr.nn_distance
            matched = any(
                spec.matches(sp_i, sp_j)
                and spec.min_length <= dist <= spec.max_length
                for spec in bond_specs
            )
            if matched:
                # Cartesian offset from atom i to this neighbour.
                image = np.array([int(x) for x in nbr.image])
                offset = (
                    structure[nbr.index].coords
                    + image @ lattice.matrix
                    - structure[i].coords
                )
                offsets.append((sp_j, offset))
        if offsets:
            uc_neighbour_offsets[i] = offsets

    # For each image atom matching a centre pattern, identify its
    # source unit-cell atom and replicate the neighbour shell.
    inv_matrix = lattice.inv_matrix
    new_species = list(expanded_species)
    new_coords_list: list[np.ndarray] = [expanded_coords]
    # Track all existing coordinates for deduplication.
    all_coords = expanded_coords

    for img_idx in range(n_uc, len(expanded_species)):
        sp = expanded_species[img_idx]
        if not any(fnmatch(sp, pat) for pat in centre_patterns):
            continue

        img_coord = expanded_coords[img_idx]

        # Find which unit-cell atom this image corresponds to.
        # The difference should be a lattice vector.
        source_idx = None
        for j in range(n_uc):
            if species_list[j] != sp:
                continue
            diff = img_coord - structure[j].coords
            # Convert to fractional; should be near-integer.
            frac_diff = diff @ inv_matrix
            if np.allclose(frac_diff, np.round(frac_diff), atol=0.01):
                source_idx = j
                break

        if source_idx is None or source_idx not in uc_neighbour_offsets:
            continue

        # Add each neighbour of the source atom, shifted to the image position.
        translation = img_coord - structure[source_idx].coords
        for nbr_sp, offset in uc_neighbour_offsets[source_idx]:
            nbr_coord = structure[source_idx].coords + translation + offset
            # Check if this coordinate already exists.
            diffs = np.linalg.norm(all_coords - nbr_coord, axis=1)
            if np.any(diffs < 1e-6):
                continue
            new_species.append(nbr_sp)
            new_coords_list.append(nbr_coord[np.newaxis, :])
            # Update all_coords for subsequent deduplication checks.
            all_coords = np.vstack([all_coords, nbr_coord[np.newaxis, :]])

    if len(new_coords_list) > 1:
        return new_species, np.vstack(new_coords_list)
    return new_species, expanded_coords


def from_pymatgen(
    structure: Structure | Sequence[Structure],
    bond_specs: list[BondSpec] | None = None,
    *,
    polyhedra: list[PolyhedronSpec] | None = None,
    pbc: bool = False,
    pbc_cutoff: float | None = None,
    centre_atom: int | None = None,
) -> StructureScene:
    """Create a StructureScene from pymatgen Structure(s).

    Args:
        structure: A single pymatgen ``Structure`` or a list of
            ``Structure`` objects (e.g. from an MD trajectory).
        bond_specs: Optional bond specification rules.  If ``None``,
            sensible defaults are generated from covalent radii.
            Pass an empty list to disable bonds.
        polyhedra: Optional polyhedron rendering rules.  If ``None``,
            no polyhedra are drawn.
        pbc: If ``True``, add periodic image atoms at cell boundaries
            so that bonds crossing periodic boundaries are drawn.
        pbc_cutoff: Cartesian distance cutoff (angstroms) for PBC
            image generation.  Atoms within this distance of a cell
            face get an image on the opposite side.  If ``None``
            (the default), the maximum bond length from *bond_specs*
            is used.
        centre_atom: Index of the atom to centre the unit cell on.
            When set, all fractional coordinates are shifted so that
            this atom sits at (0.5, 0.5, 0.5) before PBC expansion,
            and the view is centred on this atom.

    Returns:
        A StructureScene with default element styles.

    Raises:
        ImportError: If pymatgen is not installed.
    """
    try:
        from pymatgen.core import Structure
    except ImportError:
        raise ImportError(
            "pymatgen is required for from_pymatgen(). "
            "Install it with: pip install pymatgen"
        )

    if isinstance(structure, Structure):
        structures = [structure]
    else:
        structures = list(structure)

    # Recentre: shift fractional coordinates so centre_atom is at
    # (0.5, 0.5, 0.5), then wrap all sites back into [0, 1).
    if centre_atom is not None:
        recentred = []
        for s in structures:
            s = s.copy()
            shift = 0.5 - s.frac_coords[centre_atom]
            s.translate_sites(range(len(s)), shift, frac_coords=True)
            recentred.append(s)
        structures = recentred

    # Extract element symbols (not species strings like "Li+" or "O2-").
    # .symbol works for both Element and Species objects.
    species = [site.specie.symbol for site in structures[0]]

    # Default atom styles from the element lookup.
    unique_species = sorted(set(species))
    atom_styles = {sp: default_atom_style(sp) for sp in unique_species}

    # Generate default bond specs from covalent radii if none provided.
    if bond_specs is None:
        bond_specs = default_bond_specs(species)

    # Build frames, optionally expanding with PBC images.
    poly_specs = polyhedra if polyhedra is not None else []
    n_uc = len(species)  # unit-cell atom count before expansion
    if pbc:
        frames = []
        first_species: list[str] | None = None
        for i, s in enumerate(structures):
            # Geometric expansion: add images near cell faces.
            exp_species, exp_coords = _expand_pbc(s, bond_specs, pbc_cutoff)

            # Bond-aware expansion: add any bonded periodic images
            # not already captured by the geometric expansion.
            if bond_specs:
                bond_img_species, bond_img_coords = _expand_bonds(
                    s, bond_specs,
                )
                if len(bond_img_species) > 0:
                    exp_species, exp_coords = _merge_expansions(
                        exp_species, exp_coords,
                        bond_img_species, bond_img_coords,
                    )

            # Vertex expansion: for image atoms that are polyhedron
            # centres, add their coordination-shell atoms so that
            # boundary polyhedra are complete.
            if poly_specs and bond_specs:
                exp_species, exp_coords = _expand_polyhedra_vertices(
                    s, exp_species, exp_coords, n_uc,
                    bond_specs, poly_specs,
                )

            if first_species is None:
                first_species = exp_species
            frames.append(Frame(coords=exp_coords, label=f"frame_{i}"))
        # Use expanded species from the first frame.
        species = first_species  # type: ignore[assignment]
    else:
        frames = [
            Frame(coords=s.cart_coords, label=f"frame_{i}")
            for i, s in enumerate(structures)
        ]

    # Centre on first frame.
    if centre_atom is not None:
        view = ViewState(centre=frames[0].coords[centre_atom].copy())
    else:
        centroid = np.mean(frames[0].coords, axis=0)
        view = ViewState(centre=centroid)

    return StructureScene(
        species=species,
        frames=frames,
        atom_styles=atom_styles,
        bond_specs=bond_specs,
        polyhedra=polyhedra if polyhedra is not None else [],
        view=view,
        title="",
        n_unit_cell_atoms=n_uc if pbc else None,
    )
