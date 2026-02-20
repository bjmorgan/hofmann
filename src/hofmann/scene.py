"""Convenience constructors for StructureScene."""

from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hofmann.defaults import default_atom_style, default_bond_specs
from hofmann.model import (
    AtomStyle, BondSpec, Frame, PolyhedronSpec, StructureScene, ViewState,
)
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
    # Collect accepted coordinates in a list and vstack once at the
    # end, avoiding quadratic copies from repeated vstack in the loop.
    added_coords: list[np.ndarray] = []

    # Round-key set for O(1) deduplication of newly added atoms,
    # matching the approach used in _expand_neighbour_shells.
    _DEDUP_DECIMALS = 5  # 0.00001 A — well within tol
    added_keys: set[tuple[float, ...]] = set()

    for sp, coord in zip(extra_species, extra_coords):
        # Check against base atoms (vectorised).
        diffs = np.linalg.norm(base_coords - coord, axis=1)
        if np.any(diffs < tol):
            continue
        # Check against previously accepted extras (O(1) lookup).
        key = tuple(np.round(coord, _DEDUP_DECIMALS))
        if key in added_keys:
            continue
        new_species.append(sp)
        added_coords.append(coord)
        added_keys.add(key)

    if added_coords:
        return new_species, np.vstack([base_coords] + [c[np.newaxis, :] for c in added_coords])
    return new_species, base_coords


def _identify_source_atom(
    img_coord: np.ndarray,
    species: str,
    uc_coords: np.ndarray,
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
        uc_coords: Wrapped Cartesian coordinates of unit-cell atoms,
            shape ``(n_uc, 3)``.  Must use the same coordinate basis
            as *img_coord* (i.e. both derived from wrapped fractional
            coordinates).
        species_list: Species labels for unit-cell atoms.
        inv_matrix: Inverse of the lattice matrix.
        n_uc: Number of unit-cell atoms.

    Returns:
        Index of the source unit-cell atom, or ``None`` if no match.
    """
    for j in range(n_uc):
        if species_list[j] != species:
            continue
        diff = img_coord - uc_coords[j]
        frac_diff = diff @ inv_matrix
        if np.allclose(frac_diff, np.round(frac_diff), atol=0.01):
            return j
    return None


def _precompute_bonded_neighbours(
    structure: "Structure",
    bond_specs: list[BondSpec],
    uc_coords: np.ndarray | None = None,
) -> dict[int, list[tuple[str, np.ndarray]]]:
    """Build a lookup of bonded periodic neighbours for each unit-cell atom.

    For each atom in *structure*, finds all periodic neighbours that
    match at least one *bond_spec* (species + distance).  Results are
    stored as ``(species, Cartesian_offset)`` pairs, where the offset
    is relative to the source atom's position in *uc_coords*.

    When *uc_coords* are provided, offsets are computed relative to
    those (wrapped) positions, ensuring consistency with coordinates
    produced by :func:`_expand_pbc`.  Pymatgen's
    ``get_all_neighbors`` returns image vectors relative to the raw
    fractional coordinates, so the neighbour's absolute Cartesian
    position is always ``structure[j].coords + image @ L`` (using raw
    coords), and the offset stored here is that absolute position
    minus ``uc_coords[i]``.

    Args:
        structure: The pymatgen ``Structure`` (single frame).
        bond_specs: Bond specification rules to match against.
        uc_coords: Wrapped Cartesian coordinates of unit-cell atoms,
            shape ``(n_atoms, 3)``.  When ``None``, raw
            ``structure[i].coords`` are used.

    Returns:
        Dict mapping unit-cell atom index to a list of
        ``(neighbour_species, offset_vector)`` tuples.
    """
    lattice = structure.lattice
    max_cutoff = max(spec.max_length for spec in bond_specs)
    all_neighbours = structure.get_all_neighbors(max_cutoff)
    species_list = [site.specie.symbol for site in structure]

    # When uc_coords differs from raw coords (i.e. coordinates were
    # wrapped to [0, 1)), the same physical bond can map to a
    # different periodic image.  Pymatgen's image vectors are relative
    # to the raw fractional basis, so we must adjust them to the
    # wrapped basis.  The shift per atom is the integer lattice
    # translation introduced by wrapping:
    #   raw_cart = wrap_cart + wrap_shift @ L
    # where wrap_shift = floor(raw_frac).
    if uc_coords is not None:
        raw_frac = structure.frac_coords
        wrap_shift = np.floor(raw_frac).astype(int)
    else:
        wrap_shift = None

    uc_neighbour_offsets: dict[int, list[tuple[str, np.ndarray]]] = {}
    for i, neighbours in enumerate(all_neighbours):
        sp_i = species_list[i]
        offsets: list[tuple[str, np.ndarray]] = []
        src_pos = uc_coords[i] if uc_coords is not None else structure[i].coords
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
                if uc_coords is not None:
                    # Adjust the image vector from the raw basis to
                    # the wrapped basis.  In the raw basis, the
                    # neighbour is at raw[j] + image*L.  Since
                    # raw[k] = wrap[k] + shift_k*L, this equals
                    # wrap[j] + (shift_j + image)*L.  The offset
                    # from wrap[i] = raw[i] - shift_i*L becomes
                    # wrap[j] + (image + shift_j - shift_i)*L - wrap[i].
                    adj_image = image + wrap_shift[nbr.index] - wrap_shift[i]
                    nbr_abs = (
                        uc_coords[nbr.index]
                        + adj_image @ lattice.matrix
                    )
                else:
                    nbr_abs = (
                        structure[nbr.index].coords
                        + image @ lattice.matrix
                    )
                offset = nbr_abs - src_pos
                offsets.append((sp_j, offset))
        if offsets:
            uc_neighbour_offsets[i] = offsets

    return uc_neighbour_offsets


def _expand_neighbour_shells(
    structure: "Structure",
    expanded_species: list[str],
    expanded_coords: np.ndarray,
    n_uc: int,
    bond_specs: list[BondSpec],
    centre_species_only: list[str] | None = None,
    *,
    _neighbour_cache: dict[int, list[tuple[str, np.ndarray]]] | None = None,
    _uc_coords: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Ensure bonded neighbours are present for atoms already in the scene.

    For each atom (unit-cell or image) in the expanded set, look up its
    bonded periodic neighbours from the original structure and add any
    that are missing.  This completes coordination shells at cell
    boundaries by only searching atoms already present in the scene.

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
        _neighbour_cache: Pre-computed neighbour lookup from
            :func:`_precompute_bonded_neighbours`.  When provided,
            ``get_all_neighbors`` is not called again.  Callers that
            invoke this function repeatedly with the same *structure*
            and *bond_specs* should pass this to avoid redundant work.
        _uc_coords: Wrapped Cartesian coordinates of unit-cell atoms,
            shape ``(n_uc, 3)``.  When ``None``, the first *n_uc*
            rows of *expanded_coords* are used.  Callers that have
            already computed wrapped coordinates should pass them for
            consistency with *expanded_coords*.

    Returns:
        Updated ``(species, coords)`` with additional neighbour atoms
        appended.
    """
    if not bond_specs:
        return expanded_species, expanded_coords

    species_list = [site.specie.symbol for site in structure]
    # Use wrapped UC coordinates — either explicitly provided or
    # extracted from the head of expanded_coords (which _expand_pbc
    # already wrapped).
    uc_coords = _uc_coords if _uc_coords is not None else expanded_coords[:n_uc]
    if _neighbour_cache is not None:
        uc_neighbour_offsets = _neighbour_cache
    else:
        uc_neighbour_offsets = _precompute_bonded_neighbours(
            structure, bond_specs, uc_coords=uc_coords,
        )
    inv_matrix = structure.lattice.inv_matrix
    new_species = list(expanded_species)
    # Collect new coordinates in a plain list; only vstack once at
    # the end.  Deduplication checks the original array (vectorised)
    # plus a set of rounded coordinate tuples for O(1) lookup of
    # newly added atoms.
    added_coords: list[np.ndarray] = []
    _DEDUP_DECIMALS = 5  # 0.00001 A — well within 1e-6 tolerance
    added_keys: set[tuple[float, ...]] = set()

    def _should_expand(sp: str) -> bool:
        if centre_species_only is None:
            return True
        return any(fnmatch(sp, pat) for pat in centre_species_only)

    def _coord_key(coord: np.ndarray) -> tuple[float, ...]:
        """Round a coordinate to a hashable tuple for deduplication."""
        return tuple(np.round(coord, _DEDUP_DECIMALS))

    def _is_duplicate(coord: np.ndarray) -> bool:
        """Check whether *coord* duplicates an existing atom."""
        diffs = np.linalg.norm(expanded_coords - coord, axis=1)
        if np.any(diffs < 1e-6):
            return True
        return _coord_key(coord) in added_keys

    def _add_neighbour_shell(
        source_idx: int, translation: np.ndarray,
    ) -> None:
        """Add missing bonded neighbours for an atom."""
        if source_idx not in uc_neighbour_offsets:
            return
        for nbr_sp, offset in uc_neighbour_offsets[source_idx]:
            nbr_coord = uc_coords[source_idx] + translation + offset
            if _is_duplicate(nbr_coord):
                continue
            new_species.append(nbr_sp)
            added_coords.append(nbr_coord)
            added_keys.add(_coord_key(nbr_coord))

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
            img_coord, sp, uc_coords, species_list, inv_matrix, n_uc,
        )
        if source_idx is None:
            continue

        translation = img_coord - uc_coords[source_idx]
        _add_neighbour_shell(source_idx, translation)

    if added_coords:
        all_new = np.vstack(
            [expanded_coords] + [c[np.newaxis, :] for c in added_coords],
        )
        return new_species, all_new
    return new_species, expanded_coords


def _expand_recursive_bonds(
    structure: "Structure",
    expanded_species: list[str],
    expanded_coords: np.ndarray,
    n_uc: int,
    bond_specs: list[BondSpec],
    max_depth: int,
    uc_coords: np.ndarray | None = None,
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
        uc_coords: Wrapped Cartesian coordinates of unit-cell atoms,
            shape ``(n_uc, 3)``.  See :func:`_expand_neighbour_shells`.

    Returns:
        Updated ``(species, coords)`` with additional atoms appended.
    """
    recursive_specs = [s for s in bond_specs if s.recursive]
    if not recursive_specs:
        return expanded_species, expanded_coords

    cache = _precompute_bonded_neighbours(
        structure, recursive_specs, uc_coords=uc_coords,
    )
    species = list(expanded_species)
    coords = expanded_coords

    for _ in range(max_depth):
        prev_count = len(species)
        species, coords = _expand_neighbour_shells(
            structure, species, coords, n_uc, recursive_specs,
            _neighbour_cache=cache, _uc_coords=uc_coords,
        )
        if len(species) == prev_count:
            break

    if len(species) > len(expanded_species):
        return species, coords
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
    atom_styles: dict[str, AtomStyle] | None = None,
    title: str = "",
    view: ViewState | None = None,
    atom_data: dict[str, np.ndarray] | None = None,
) -> StructureScene:
    """Create a StructureScene from pymatgen Structure(s).

    Args:
        structure: A single pymatgen ``Structure`` or a list of
            ``Structure`` objects (e.g. from an MD trajectory).
        bond_specs: Optional bond specification rules.  If ``None``,
            sensible defaults are generated from VESTA bond length
            cutoffs.  Pass an empty list to disable bonds.
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
            and the view is centred on this atom.  If *view* is also
            provided, the explicit view takes precedence and only the
            fractional-coordinate shift is applied.
        max_recursive_depth: Maximum number of iterations for
            recursive bond expansion (must be >= 1).  Only relevant
            when one or more *bond_specs* have ``recursive=True``.
        atom_styles: Per-species style overrides.  When provided,
            these are merged on top of the auto-generated defaults
            so you only need to specify the species you want to
            customise.
        title: Scene title for display.
        view: Camera / projection state.  When ``None`` (the
            default), the view is auto-centred on the structure.
        atom_data: Per-atom metadata arrays, keyed by name.

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

    if max_recursive_depth < 1:
        raise ValueError(
            f"max_recursive_depth must be >= 1, "
            f"got {max_recursive_depth}"
        )

    if isinstance(structure, Structure):
        structures = [structure]
    else:
        structures = list(structure)

    if not structures:
        raise ValueError("structure must not be empty")

    # Recentre: shift fractional coordinates so centre_atom is at
    # (0.5, 0.5, 0.5), then wrap all sites back into [0, 1).
    if centre_atom is not None:
        n_sites = len(structures[0])
        if not 0 <= centre_atom < n_sites:
            raise ValueError(
                f"centre_atom {centre_atom} out of range for structure "
                f"with {n_sites} site(s)"
            )
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

    # Default atom styles from the element lookup, merged with overrides.
    unique_species = sorted(set(species))
    merged_styles = {sp: default_atom_style(sp) for sp in unique_species}
    if atom_styles is not None:
        merged_styles.update(atom_styles)

    # Generate default bond specs from VESTA cutoffs if none provided.
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

            # Wrapped UC coordinates: _expand_pbc wraps fractional
            # coordinates to [0, 1) before converting to Cartesian.
            # All downstream expansion functions must use the same
            # wrapped basis to avoid coordinate-frame mismatches when
            # pymatgen stores frac_coords outside [0, 1).
            uc_coords = exp_coords[:n_uc]

            # Single-pass bond completion: for each bond spec with
            # complete set, add missing bonded partners across
            # periodic boundaries (non-recursive).  Each spec is
            # processed individually so that its centre selector
            # is applied independently.  Specs that are also
            # recursive are skipped — recursive already subsumes
            # the single-pass search.
            for bond_spec in bond_specs:
                if not bond_spec.complete or bond_spec.recursive:
                    continue
                centres: list[str] | None = (
                    None if bond_spec.complete == "*"
                    else [str(bond_spec.complete)]
                )
                exp_species, exp_coords = _expand_neighbour_shells(
                    s, exp_species, exp_coords, n_uc, [bond_spec],
                    centre_species_only=centres,
                    _uc_coords=uc_coords,
                )

            # Recursive bond expansion: iteratively add bonded
            # atoms across periodic boundaries for bond specs
            # with recursive=True.
            exp_species, exp_coords = _expand_recursive_bonds(
                s, exp_species, exp_coords, n_uc, bond_specs,
                max_depth=max_recursive_depth,
                uc_coords=uc_coords,
            )

            # Neighbour-shell completion for polyhedra centres:
            # ensure every atom matching a polyhedron centre pattern
            # has its full coordination shell present, so that
            # boundary polyhedra are complete.
            if poly_specs and bond_specs:
                centre_patterns = [sp.centre for sp in poly_specs]
                poly_cache = _precompute_bonded_neighbours(
                    s, bond_specs, uc_coords=uc_coords,
                )
                exp_species, exp_coords = _expand_neighbour_shells(
                    s, exp_species, exp_coords, n_uc, bond_specs,
                    centre_species_only=centre_patterns,
                    _neighbour_cache=poly_cache,
                    _uc_coords=uc_coords,
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

    # Centre on first frame (unless the caller supplied a view).
    if view is None:
        if centre_atom is not None:
            view = ViewState(centre=frames[0].coords[centre_atom].copy())
        else:
            centroid = np.mean(frames[0].coords, axis=0)
            view = ViewState(centre=centroid)

    return StructureScene(
        species=species,
        frames=frames,
        atom_styles=merged_styles,
        bond_specs=bond_specs,
        polyhedra=polyhedra if polyhedra is not None else [],
        view=view,
        title=title,
        lattice=structures[0].lattice.matrix.copy(),
        atom_data=dict(atom_data) if atom_data is not None else {},
    )
