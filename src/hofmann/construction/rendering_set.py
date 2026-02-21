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


def _expand_padding(
    coords: np.ndarray,
    lattice: np.ndarray,
    n_physical: int,
    pbc_padding: float,
    materialise: object,
) -> list[tuple[int, tuple[int, int, int], int]]:
    """Materialise image atoms for physical atoms near cell faces.

    For each physical atom within *pbc_padding* angstroms of a unit
    cell face, creates image atoms on the opposite side(s).  Handles
    edges and corners via the Cartesian product of per-axis shifts.

    Args:
        coords: Physical atom coordinates, shape ``(n_physical, 3)``.
        lattice: 3x3 lattice matrix (row vectors).
        n_physical: Number of physical atoms.
        pbc_padding: Cartesian distance threshold (angstroms).
        materialise: Callback ``(phys_idx, image_tuple) -> exp_idx``.

    Returns:
        List of ``(phys_idx, shift, expanded_idx)`` for newly created
        padding atoms.
    """
    inv_lattice = np.linalg.inv(lattice)
    frac_coords = coords[:n_physical] @ inv_lattice

    # Perpendicular heights: d_i = volume / |a_j x a_k|
    volume = abs(np.linalg.det(lattice))
    heights = np.array([
        volume / np.linalg.norm(np.cross(lattice[1], lattice[2])),
        volume / np.linalg.norm(np.cross(lattice[2], lattice[0])),
        volume / np.linalg.norm(np.cross(lattice[0], lattice[1])),
    ])

    # For each axis, determine threshold in fractional coordinates.
    frac_thresholds = pbc_padding / heights

    new_atoms: list[tuple[int, tuple[int, int, int], int]] = []
    for i in range(n_physical):
        frac = frac_coords[i]
        # Per-axis shifts needed.
        axis_shifts: list[list[int]] = []
        for ax in range(3):
            shifts = [0]
            if frac[ax] < frac_thresholds[ax]:
                shifts.append(1)
            if frac[ax] > 1.0 - frac_thresholds[ax]:
                shifts.append(-1)
            axis_shifts.append(shifts)

        # Cartesian product of per-axis shifts, excluding (0, 0, 0).
        for sx in axis_shifts[0]:
            for sy in axis_shifts[1]:
                for sz in axis_shifts[2]:
                    if sx == 0 and sy == 0 and sz == 0:
                        continue
                    shift = (sx, sy, sz)
                    exp_idx = materialise(i, shift)
                    new_atoms.append((i, shift, exp_idx))
    return new_atoms


def _discover_bonds_for_new_atoms(
    new_atoms: list[tuple[int, tuple[int, int, int], int]],
    periodic_bonds: list[Bond],
    coords: np.ndarray,
    lattice: np.ndarray,
    n_physical: int,
    image_registry: dict[tuple[int, tuple[int, int, int]], int],
    image_coords_list: list[np.ndarray],
    rendering_bonds: list[Bond],
) -> None:
    """Find bonds between newly created image atoms and existing atoms.

    For each new atom ``(phys_idx, shift)``, scans periodic bonds
    involving ``phys_idx`` and adds rendering bonds to any target that
    already exists in the expanded set.  Does NOT materialise new
    atoms — only connects to existing ones.

    Args:
        new_atoms: List of ``(phys_idx, shift, expanded_idx)``.
        periodic_bonds: Original periodic bonds.
        coords: Physical atom coordinates.
        lattice: Lattice matrix.
        n_physical: Number of physical atoms.
        image_registry: Maps ``(phys_idx, image)`` to expanded index.
        image_coords_list: Coordinates of image atoms.
        rendering_bonds: Mutable list to append new bonds to.
    """
    # Build per-atom bond lookup.
    atom_bonds: dict[int, list[tuple[int, tuple[int, int, int], BondSpec]]] = {}
    for bond in periodic_bonds:
        a, b = bond.index_a, bond.index_b
        img = bond.image
        atom_bonds.setdefault(a, []).append((b, img, bond.spec))
        neg_img = tuple(-x for x in img)
        atom_bonds.setdefault(b, []).append((a, neg_img, bond.spec))

    for phys_idx, shift, exp_idx in new_atoms:
        if phys_idx not in atom_bonds:
            continue
        for other_phys, bond_img, spec in atom_bonds[phys_idx]:
            target_shift = tuple(
                s + i for s, i in zip(shift, bond_img)
            )
            # Find the target in the expanded set.
            if target_shift == (0, 0, 0):
                target_idx = other_phys
            else:
                target_key = (other_phys, target_shift)
                if target_key not in image_registry:
                    continue  # Target not materialised; skip.
                target_idx = image_registry[target_key]

            # Compute bond length from coordinates.
            a_coord = coords[phys_idx] + np.array(shift, dtype=float) @ lattice
            if target_shift == (0, 0, 0):
                b_coord = coords[other_phys]
            else:
                b_coord = (
                    coords[other_phys]
                    + np.array(target_shift, dtype=float) @ lattice
                )
            length = float(np.linalg.norm(b_coord - a_coord))
            rendering_bonds.append(Bond(exp_idx, target_idx, length, spec))


def _complete_polyhedra_vertices(
    species: list[str],
    coords: np.ndarray,
    lattice: np.ndarray,
    n_physical: int,
    periodic_bonds: list[Bond],
    polyhedra_specs: list,
    image_registry: dict[tuple[int, tuple[int, int, int]], int],
    materialise: object,
) -> list[tuple[int, tuple[int, int, int], int]]:
    """Ensure polyhedron centres have complete coordination shells.

    For each atom (physical or image) matching a
    :class:`PolyhedronSpec` centre pattern, materialises any missing
    bonded neighbours.  Single-pass: newly created vertex atoms are
    not themselves checked as potential centres.

    Args:
        species: Species labels for physical atoms.
        coords: Physical atom coordinates.
        lattice: Lattice matrix.
        n_physical: Number of physical atoms.
        periodic_bonds: Original periodic bonds.
        polyhedra_specs: Polyhedron specification rules.
        image_registry: Current ``(phys_idx, image)`` → expanded index map.
        materialise: Callback ``(phys_idx, image_tuple) -> exp_idx``.

    Returns:
        List of ``(phys_idx, shift, expanded_idx)`` for newly created
        vertex atoms.
    """
    from hofmann.model import PolyhedronSpec as _PSpec

    # Collect all centre patterns.
    centre_patterns = [s.centre for s in polyhedra_specs
                       if isinstance(s, _PSpec)]
    if not centre_patterns:
        return []

    def _is_centre(sp: str) -> bool:
        return any(fnmatch(sp, pat) for pat in centre_patterns)

    # Build per-atom bond lookup from periodic bonds.
    atom_bonds: dict[int, list[tuple[int, tuple[int, int, int], BondSpec]]] = {}
    for bond in periodic_bonds:
        a, b = bond.index_a, bond.index_b
        img = bond.image
        atom_bonds.setdefault(a, []).append((b, img, bond.spec))
        neg_img = tuple(-x for x in img)
        atom_bonds.setdefault(b, []).append((a, neg_img, bond.spec))

    new_atoms: list[tuple[int, tuple[int, int, int], int]] = []

    # Snapshot of all atoms to check (physical + current images).
    # We freeze this before adding vertices so the step is single-pass.
    atoms_to_check: list[tuple[int, tuple[int, int, int]]] = [
        (i, (0, 0, 0)) for i in range(n_physical)
    ] + list(image_registry.keys())

    for phys_idx, shift in atoms_to_check:
        if not _is_centre(species[phys_idx]):
            continue
        if phys_idx not in atom_bonds:
            continue

        for other_phys, bond_img, spec in atom_bonds[phys_idx]:
            target_shift = tuple(
                s + i for s, i in zip(shift, bond_img)
            )
            if target_shift == (0, 0, 0):
                continue  # Physical atom, already exists.
            target_key = (other_phys, target_shift)
            if target_key in image_registry:
                continue  # Already materialised.
            exp_idx = materialise(other_phys, target_shift)
            new_atoms.append((other_phys, target_shift, exp_idx))

    return new_atoms


def build_rendering_set(
    species: list[str],
    coords: np.ndarray,
    periodic_bonds: list[Bond],
    bond_specs: list[BondSpec],
    lattice: np.ndarray,
    max_recursive_depth: int = 5,
    pbc_padding: float | None = None,
    polyhedra_specs: list | None = None,
) -> RenderingSet:
    """Build the expanded rendering set from periodic bond data.

    Takes physical atoms and their periodic bonds (including cross-
    boundary bonds with non-zero ``image`` fields) and produces an
    expanded set of atoms and bonds suitable for rendering.  Image
    atoms are materialised according to the ``complete`` and
    ``recursive`` settings on each bond spec, geometric cell-face
    padding, and polyhedra vertex completion.

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
        pbc_padding: Cartesian margin (angstroms) for geometric
            cell-face expansion.  Atoms within this distance of a
            cell face are duplicated on the opposite side.  ``None``
            disables geometric expansion.
        polyhedra_specs: Polyhedron rules.  When provided, ensures
            that every atom matching a centre pattern has its full
            coordination shell materialised.

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

    # --- Geometric padding (pbc_padding) ---
    # Materialise image atoms for physical atoms near cell faces.
    padding_new_atoms: list[tuple[int, tuple[int, int, int], int]] = []
    if pbc_padding is not None and pbc_padding > 0:
        padding_new_atoms = _expand_padding(
            coords, lattice, n_physical, pbc_padding, _materialise,
        )

    # --- Bond discovery for padding atoms ---
    # Scan periodic bonds for connections to newly created padding atoms.
    if padding_new_atoms:
        _discover_bonds_for_new_atoms(
            padding_new_atoms, periodic_bonds, coords, lattice,
            n_physical, image_registry, image_coords_list, rendering_bonds,
        )

    # --- Polyhedra vertex completion ---
    # Ensure every polyhedron centre has its full coordination shell.
    if polyhedra_specs:
        vertex_new = _complete_polyhedra_vertices(
            species, coords, lattice, n_physical, periodic_bonds,
            polyhedra_specs, image_registry, _materialise,
        )
        if vertex_new:
            _discover_bonds_for_new_atoms(
                vertex_new, periodic_bonds, coords, lattice,
                n_physical, image_registry, image_coords_list,
                rendering_bonds,
            )

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


def deduplicate_molecules(
    rset: RenderingSet,
    lattice: np.ndarray,
) -> RenderingSet:
    """Remove duplicate molecular fragments from a rendering set.

    When bonds span periodic boundaries and ``complete`` materialises
    image atoms, the same molecule may appear more than once — once as
    a spatially contiguous cluster and once as orphaned physical atoms.
    This function identifies connected components, groups those that
    share source atoms, and keeps only the canonical representative of
    each group.

    Extended structures (slabs, frameworks) that wrap around the
    periodic cell are detected and left untouched.  A component
    "wraps" when it contains both a physical atom and an image of
    that atom (the same source index appears more than once),
    indicating a structure that is infinite in at least one direction.

    The canonical component is chosen by lexicographic comparison of
    ``(n_atoms, n_physical, -frac_com)``, where *frac_com* is the
    fractional centre of mass.  This selects the largest fragment,
    breaking ties by preferring the copy closest to the cell origin
    — a deterministic, view-independent rule.

    Args:
        rset: The rendering set to deduplicate.
        lattice: 3x3 lattice matrix (row vectors), used to convert
            centre-of-mass coordinates to fractional space for the
            tiebreaker.

    Returns:
        A new :class:`RenderingSet` with duplicate fragments removed
        and indices remapped.
    """
    n = len(rset.species)
    if n == 0:
        return rset

    # --- Find connected components via union-find ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bond in rset.bonds:
        union(bond.index_a, bond.index_b)

    # Group atoms by component root.
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)

    # --- Group components by shared source indices ---
    # Build source_set for each component.
    comp_roots = list(components.keys())
    source_sets: dict[int, set[int]] = {}
    for root, members in components.items():
        source_sets[root] = {int(rset.source_indices[i]) for i in members}

    # Transitive closure: merge groups that share any source atom.
    # Use union-find on component roots.
    group_parent = {r: r for r in comp_roots}

    def gfind(x: int) -> int:
        while group_parent[x] != x:
            group_parent[x] = group_parent[group_parent[x]]
            x = group_parent[x]
        return x

    def gunion(a: int, b: int) -> None:
        ra, rb = gfind(a), gfind(b)
        if ra != rb:
            group_parent[rb] = ra

    # Check all pairs for shared source atoms.
    # For efficiency, build source_atom → component roots mapping.
    src_to_roots: dict[int, list[int]] = {}
    for root, srcs in source_sets.items():
        for s in srcs:
            src_to_roots.setdefault(s, []).append(root)

    for roots_list in src_to_roots.values():
        for i in range(1, len(roots_list)):
            gunion(roots_list[0], roots_list[i])

    # Collect groups of components.
    groups: dict[int, list[int]] = {}
    for root in comp_roots:
        g = gfind(root)
        groups.setdefault(g, []).append(root)

    # --- Select canonical component per group ---
    keep_atoms: set[int] = set()

    lattice = np.asarray(lattice, dtype=float)
    inv_lattice = np.linalg.inv(lattice)

    for group_roots in groups.values():
        if len(group_roots) == 1:
            # Only one component in this group — keep it.
            keep_atoms.update(components[group_roots[0]])
            continue

        # Check if any component in this group wraps around the cell.
        # A component wraps when it contains both a physical atom and
        # an image of that atom (i.e. the same source index appears
        # more than once).  This identifies extended structures (slabs,
        # frameworks) whose edge fragments should not be discarded.
        group_wraps = False
        for root in group_roots:
            members = components[root]
            if len(source_sets[root]) < len(members):
                group_wraps = True
                break

        if group_wraps:
            for root in group_roots:
                keep_atoms.update(components[root])
            continue

        # Score each component: (n_atoms, n_physical, frac_key).
        # frac_key is the negative fractional CoM coordinates so that
        # lexicographic max picks the fragment closest to the origin —
        # this gives deterministic, view-independent selection.
        best_root = None
        best_score: tuple = (-1, -1, ())
        for root in group_roots:
            members = components[root]
            n_members = len(members)
            n_physical = sum(1 for i in members
                             if rset.source_indices[i] == i)
            com = rset.coords[members].mean(axis=0)
            frac_com = com @ inv_lattice
            # Negate so that smaller fractional coords win via max().
            frac_key = tuple(-float(f) for f in frac_com)
            score = (n_members, n_physical, frac_key)
            if score > best_score:
                best_score = score
                best_root = root

        keep_atoms.update(components[best_root])

    # --- Build output with remapped indices ---
    if len(keep_atoms) == n:
        return rset  # Nothing to remove.

    keep_list = sorted(keep_atoms)
    old_to_new = {old: new for new, old in enumerate(keep_list)}

    new_species = [rset.species[i] for i in keep_list]
    new_coords = rset.coords[keep_list]
    new_source = rset.source_indices[keep_list]
    new_bonds = []
    for bond in rset.bonds:
        if bond.index_a in old_to_new and bond.index_b in old_to_new:
            new_bonds.append(Bond(
                old_to_new[bond.index_a],
                old_to_new[bond.index_b],
                bond.length,
                bond.spec,
            ))

    return RenderingSet(
        species=new_species,
        coords=new_coords,
        bonds=new_bonds,
        source_indices=new_source,
    )
