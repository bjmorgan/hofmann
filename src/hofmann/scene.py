"""Convenience constructors for StructureScene."""

from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
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
    structure: "Structure",
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
            # Wrap fractional coordinates to [0, 1) for consistency
            # with the main expansion path.
            frac = structure.frac_coords % 1.0
            coords = frac @ structure.lattice.matrix
            return species_list, coords
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
    structure: "Structure",
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


def _identify_source_atom(
    img_coord: np.ndarray,
    species: str,
    structure: "Structure",
    species_list: list[str],
    inv_matrix: np.ndarray,
    n_uc: int,
) -> int | None:
    """Identify which unit-cell atom a periodic image corresponds to.

    Compares the image coordinate against each unit-cell atom of the
    same species.  If the fractional difference is close to integer
    lattice vectors, the image is a periodic copy of that atom.

    Args:
        img_coord: Cartesian coordinate of the image atom.
        species: Species label of the image atom.
        structure: The pymatgen ``Structure``.
        species_list: Species labels for unit-cell atoms.
        inv_matrix: Inverse of the lattice matrix.
        n_uc: Number of unit-cell atoms.

    Returns:
        Index of the source unit-cell atom, or ``None`` if no match.
    """
    for j in range(n_uc):
        if species_list[j] != species:
            continue
        diff = img_coord - structure[j].coords
        frac_diff = diff @ inv_matrix
        if np.allclose(frac_diff, np.round(frac_diff), atol=0.01):
            return j
    return None


def _expand_neighbour_shells(
    structure: "Structure",
    expanded_species: list[str],
    expanded_coords: np.ndarray,
    n_uc: int,
    bond_specs: list[BondSpec],
    centre_species_only: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Ensure bonded neighbours are present for atoms already in the scene.

    For each atom (unit-cell or image) in the expanded set, look up its
    bonded periodic neighbours from the original structure and add any
    that are missing.  This completes coordination shells at cell
    boundaries without the aggressive expansion of the old
    ``_expand_bonds`` (which searched from *all* unit-cell atoms).

    When *centre_species_only* is given, only atoms whose species
    matches one of the patterns have their shells completed.  This is
    used by polyhedra expansion to avoid adding unnecessary atoms.

    This is **non-recursive**: only atoms already present are checked;
    newly added neighbour atoms are never themselves expanded.

    Args:
        structure: The pymatgen ``Structure`` (single frame).
        expanded_species: Species list after geometric expansion.
        expanded_coords: Coordinates after geometric expansion.
        n_uc: Number of unit-cell atoms (before any expansion).
        bond_specs: Bond specification rules.
        centre_species_only: If given, only expand shells for atoms
            matching one of these species patterns (fnmatch).  If
            ``None``, expand shells for all atoms.

    Returns:
        Updated ``(species, coords)`` with additional neighbour atoms
        appended.
    """
    if not bond_specs:
        return expanded_species, expanded_coords

    lattice = structure.lattice

    # Precompute periodic neighbour data for each unit-cell atom.
    max_cutoff = max(spec.max_length for spec in bond_specs)
    all_neighbours = structure.get_all_neighbors(max_cutoff)
    species_list = [site.specie.symbol for site in structure]

    # For each unit-cell atom, store its bonded neighbours as
    # (species, Cartesian offset) pairs.
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
                image = np.array([int(x) for x in nbr.image])
                offset = (
                    structure[nbr.index].coords
                    + image @ lattice.matrix
                    - structure[i].coords
                )
                offsets.append((sp_j, offset))
        if offsets:
            uc_neighbour_offsets[i] = offsets

    inv_matrix = lattice.inv_matrix
    new_species = list(expanded_species)
    new_coords_list: list[np.ndarray] = [expanded_coords]
    all_coords = expanded_coords

    def _should_expand(sp: str) -> bool:
        if centre_species_only is None:
            return True
        return any(fnmatch(sp, pat) for pat in centre_species_only)

    def _add_neighbour_shell(
        source_idx: int, translation: np.ndarray,
    ) -> None:
        """Add missing bonded neighbours for an atom."""
        nonlocal all_coords
        if source_idx not in uc_neighbour_offsets:
            return
        for nbr_sp, offset in uc_neighbour_offsets[source_idx]:
            nbr_coord = structure[source_idx].coords + translation + offset
            diffs = np.linalg.norm(all_coords - nbr_coord, axis=1)
            if np.any(diffs < 1e-6):
                continue
            new_species.append(nbr_sp)
            new_coords_list.append(nbr_coord[np.newaxis, :])
            all_coords = np.vstack([all_coords, nbr_coord[np.newaxis, :]])

    # Process unit-cell atoms.
    for uc_idx in range(n_uc):
        if not _should_expand(species_list[uc_idx]):
            continue
        _add_neighbour_shell(uc_idx, np.zeros(3))

    # Process image atoms.
    for img_idx in range(n_uc, len(expanded_species)):
        sp = expanded_species[img_idx]
        if not _should_expand(sp):
            continue

        img_coord = expanded_coords[img_idx]
        source_idx = _identify_source_atom(
            img_coord, sp, structure, species_list, inv_matrix, n_uc,
        )
        if source_idx is None:
            continue

        translation = img_coord - structure[source_idx].coords
        _add_neighbour_shell(source_idx, translation)

    if len(new_coords_list) > 1:
        return new_species, np.vstack(new_coords_list)
    return new_species, expanded_coords


def _expand_recursive_bonds(
    structure: "Structure",
    expanded_species: list[str],
    expanded_coords: np.ndarray,
    n_uc: int,
    bond_specs: list[BondSpec],
    max_depth: int,
) -> tuple[list[str], np.ndarray]:
    """Iteratively add bonded atoms across periodic boundaries.

    For each :class:`BondSpec` with ``recursive=True``, atoms already
    in the scene have their bonded periodic neighbours added.  Newly
    added atoms are themselves checked on the next iteration, so that
    molecules spanning multiple cell widths are completed.  Iteration
    stops when no new atoms are found or *max_depth* is reached.

    Args:
        structure: The pymatgen ``Structure`` (single frame).
        expanded_species: Species list after geometric expansion.
        expanded_coords: Coordinates after geometric expansion.
        n_uc: Number of unit-cell atoms (before any expansion).
        bond_specs: All bond specification rules.
        max_depth: Maximum number of iterations.

    Returns:
        Updated ``(species, coords)`` with additional atoms appended.
    """
    recursive_specs = [s for s in bond_specs if s.recursive]
    if not recursive_specs:
        return expanded_species, expanded_coords

    lattice = structure.lattice
    max_cutoff = max(spec.max_length for spec in recursive_specs)
    all_neighbours = structure.get_all_neighbors(max_cutoff)
    species_list = [site.specie.symbol for site in structure]
    inv_matrix = lattice.inv_matrix

    # Precompute bonded neighbours for each UC atom, filtered to
    # recursive specs only.
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
                for spec in recursive_specs
            )
            if matched:
                image = np.array([int(x) for x in nbr.image])
                offset = (
                    structure[nbr.index].coords
                    + image @ lattice.matrix
                    - structure[i].coords
                )
                offsets.append((sp_j, offset))
        if offsets:
            uc_neighbour_offsets[i] = offsets

    new_species = list(expanded_species)
    all_coords = expanded_coords.copy()

    for _depth in range(max_depth):
        added_species: list[str] = []
        added_coords: list[np.ndarray] = []
        n_current = len(new_species)

        for idx in range(n_current):
            sp = new_species[idx]
            coord = all_coords[idx]

            # Identify the UC source atom for this atom.
            if idx < n_uc:
                source_idx: int | None = idx
                translation = np.zeros(3)
            else:
                source_idx = _identify_source_atom(
                    coord, sp, structure, species_list, inv_matrix, n_uc,
                )
                if source_idx is None:
                    continue
                translation = coord - structure[source_idx].coords

            if source_idx not in uc_neighbour_offsets:
                continue

            for nbr_sp, offset in uc_neighbour_offsets[source_idx]:
                nbr_coord = structure[source_idx].coords + translation + offset
                # Check against existing atoms.
                diffs = np.linalg.norm(all_coords - nbr_coord, axis=1)
                if np.any(diffs < 1e-6):
                    continue
                # Check against atoms added earlier this round.
                if added_coords and any(
                    np.linalg.norm(c - nbr_coord) < 1e-6
                    for c in added_coords
                ):
                    continue
                added_species.append(nbr_sp)
                added_coords.append(nbr_coord)

        if not added_coords:
            break

        new_species.extend(added_species)
        all_coords = np.vstack(
            [all_coords] + [c[np.newaxis, :] for c in added_coords],
        )

    if len(new_species) > len(expanded_species):
        return new_species, all_coords
    return expanded_species, expanded_coords


def from_pymatgen(
    structure: "Structure | Sequence[Structure]",
    bond_specs: list[BondSpec] | None = None,
    *,
    polyhedra: list[PolyhedronSpec] | None = None,
    pbc: bool = True,
    pbc_padding: float | None = 0.1,
    centre_atom: int | None = None,
    max_recursive_depth: int = 5,
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
        pbc: If ``True`` (the default), add periodic image atoms at
            cell boundaries so that bonds crossing periodic boundaries
            are drawn.  Set to ``False`` to disable all PBC expansion.
        pbc_padding: Cartesian margin (angstroms) around the unit cell
            for placing periodic image atoms.  Atoms within this
            distance of a cell face get an image on the opposite side.
            The default of 0.1 angstroms captures atoms sitting on cell
            boundaries without cluttering the scene.  Set to ``None``
            to fall back to the maximum bond length from *bond_specs*
            for wider geometric expansion.
        centre_atom: Index of the atom to centre the unit cell on.
            When set, all fractional coordinates are shifted so that
            this atom sits at (0.5, 0.5, 0.5) before PBC expansion,
            and the view is centred on this atom.
        max_recursive_depth: Maximum number of iterations for
            recursive bond expansion.  Only relevant when one or
            more *bond_specs* have ``recursive=True``.

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
            exp_species, exp_coords = _expand_pbc(s, bond_specs, pbc_padding)

            # Single-pass bond completion: for each bond spec with
            # complete set, add missing bonded partners across
            # periodic boundaries (non-recursive).  Each spec is
            # processed individually so that its centre selector
            # is applied independently.  Specs that are also
            # recursive are skipped — recursive already subsumes
            # the single-pass search.
            for sp in bond_specs:
                if not sp.complete or sp.recursive:
                    continue
                centres: list[str] | None = (
                    None if sp.complete == "*"
                    else [str(sp.complete)]
                )
                exp_species, exp_coords = _expand_neighbour_shells(
                    s, exp_species, exp_coords, n_uc, [sp],
                    centre_species_only=centres,
                )

            # Recursive bond expansion: iteratively add bonded
            # atoms across periodic boundaries for bond specs
            # with recursive=True.
            if bond_specs:
                exp_species, exp_coords = _expand_recursive_bonds(
                    s, exp_species, exp_coords, n_uc, bond_specs,
                    max_depth=max_recursive_depth,
                )

            # Neighbour-shell completion for polyhedra centres:
            # ensure every atom matching a polyhedron centre pattern
            # has its full coordination shell present, so that
            # boundary polyhedra are complete.
            if poly_specs and bond_specs:
                centre_patterns = [sp.centre for sp in poly_specs]
                exp_species, exp_coords = _expand_neighbour_shells(
                    s, exp_species, exp_coords, n_uc, bond_specs,
                    centre_species_only=centre_patterns,
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
        lattice=structures[0].lattice.matrix.copy(),
    )
