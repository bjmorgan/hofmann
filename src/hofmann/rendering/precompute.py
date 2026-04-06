"""Scene pre-computation and polyhedra helpers for the painter's algorithm.

Contains view-independent scene assembly (``_precompute_scene``) and
polyhedra-related helpers (slab clipping, face collection) used by
the drawing loop in ``painter.py``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import sys
import warnings

import numpy as np

from hofmann.construction.bonds import compute_bonds
from hofmann.construction.polyhedra import compute_polyhedra
from hofmann.construction.rendering_set import (
    build_rendering_set,
    deduplicate_molecules,
)
from hofmann.model import (
    AtomStyle,
    Bond,
    CmapSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    normalise_colour,
    resolve_atom_colours,
)
from hofmann._constants import DEFAULT_ATOM_RADIUS


@dataclass(frozen=True)
class _PolyhedronRenderData:
    """Resolved rendering style for a single polyhedron.

    Groups the per-polyhedron visual properties computed once by
    :func:`_precompute_scene` and consumed by
    :func:`_collect_polyhedra_faces`.

    Attributes:
        base_colour: Face base colour before shading.
        alpha: Face transparency (0 = transparent, 1 = opaque).
        edge_colour: Wireframe edge colour.
        edge_width: Wireframe edge line width (points).
    """

    base_colour: tuple[float, float, float]
    alpha: float
    edge_colour: tuple[float, float, float]
    edge_width: float


@dataclass
class _PrecomputedScene:
    """View-independent data cached between redraws.

    Built once per frame by :func:`_precompute_scene` and reused across rotation /
    zoom changes in the interactive viewer.
    """

    coords: np.ndarray
    radii_3d: np.ndarray
    atom_colours: list[tuple[float, float, float]]
    bond_half_colours: list[tuple[float, float, float]]
    adjacency: dict[int, list[tuple[int, Bond]]]
    bond_ia: np.ndarray
    bond_ib: np.ndarray
    bond_radii: np.ndarray
    bond_index: dict[int, int]
    polyhedra: list
    style_hidden_atoms: set[int]
    style_hidden_bond_ids: set[int]
    hidden_atoms: set[int]
    hidden_bond_ids: set[int]
    poly_render_data: list[_PolyhedronRenderData]


def _compute_atom_radii(
    species: Sequence[str],
    atom_styles: Mapping[str, AtomStyle],
) -> np.ndarray:
    """Compute per-atom display radii from a species list and style map.

    Atoms whose species is not present in *atom_styles* fall back to
    :data:`DEFAULT_ATOM_RADIUS`.

    Args:
        species: Per-atom species names, one entry per atom.
        atom_styles: Mapping from species name to :class:`AtomStyle`.
            Species missing from the map trigger the default fallback.

    Returns:
        A ``(n_atoms,)`` array of ``float`` display radii.
    """
    species_arr = np.asarray(species)
    radii = np.full(len(species), DEFAULT_ATOM_RADIUS)
    for sp in set(species):
        style = atom_styles.get(sp)
        if style is not None:
            radii[species_arr == sp] = style.radius
    return radii


def _warn_missing_atom_styles(
    species: Sequence[str],
    atom_styles: Mapping[str, AtomStyle],
) -> None:
    """Emit a single warning listing all species that lack an ``AtomStyle``.

    Called at the start of each :func:`_precompute_scene` invocation
    (which runs once per frame during animation and on each frame
    change in the interactive viewer).  Python's default warning
    filter deduplicates identical messages from the same call site,
    so the warning typically appears once per process.

    Covers the radius fallback (:data:`DEFAULT_ATOM_RADIUS`) and,
    when no colourmap overrides species colours, the colour fallback
    (grey).

    Does nothing if all species have styles or the species list is empty.

    Args:
        species: Per-atom species names, one entry per atom.
        atom_styles: Mapping from species name to :class:`AtomStyle`.
    """
    missing = sorted(frozenset(species) - atom_styles.keys())
    if not missing:
        return
    species_list = ", ".join(f"'{sp}'" for sp in missing)
    # Walk the stack to the first frame outside the hofmann package
    # so the warning points at user code, not library internals.
    depth = 1
    frame = sys._getframe(0)
    while frame.f_back is not None:
        frame = frame.f_back
        depth += 1
        mod = frame.f_globals.get("__name__", "")
        if mod != "hofmann" and not mod.startswith("hofmann."):
            break
    warnings.warn(
        f"No AtomStyle defined for species: {species_list}. "
        f"Using default radius and colour.",
        UserWarning,
        stacklevel=depth,
    )


def _precompute_scene(
    scene: StructureScene,
    frame_index: int,
    render_style: RenderStyle | None = None,
    *,
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
) -> _PrecomputedScene:
    """Pre-compute view-independent data for a single frame.

    Returns a :class:`_PrecomputedScene` of radii, colours, bonds, and
    adjacency that stay constant across rotation / zoom changes but
    must be recomputed when the frame changes.
    """
    rs = render_style or RenderStyle()
    frame = scene.frames[frame_index]
    coords = frame.coords

    # Run periodic bond pipeline: compute bonds (with MIC when
    # periodic), then build the expanded rendering set.
    lattice = frame.lattice if rs.pbc else None
    periodic_bonds = compute_bonds(
        scene.species, coords, scene.bond_specs, lattice=lattice,
    )

    if lattice is not None:
        rset = build_rendering_set(
            scene.species, coords, periodic_bonds,
            scene.bond_specs, lattice,
            max_recursive_depth=rs.max_recursive_depth,
            pbc_padding=rs.pbc_padding,
            polyhedra_specs=scene.polyhedra,
        )
        if rs.deduplicate_molecules:
            rset = deduplicate_molecules(rset, lattice)
        species = rset.species
        coords = rset.coords
        bonds = rset.bonds
        source_indices = rset.source_indices
    else:
        species = scene.species
        bonds = periodic_bonds
        source_indices = np.arange(len(scene.species))

    _warn_missing_atom_styles(species, scene.atom_styles)

    n_atoms = len(species)

    # Map atom_data through source_indices for expanded set.
    atom_data = {}
    for key, arr in scene.atom_data.items():
        row = arr[frame_index] if arr.ndim == 2 else arr
        atom_data[key] = row[source_indices]

    radii_3d = _compute_atom_radii(species, scene.atom_styles)

    atom_colours = resolve_atom_colours(
        species, scene.atom_styles, atom_data,
        colour_by=colour_by, cmap=cmap, colour_range=colour_range,
        scene_atom_data=scene.atom_data,
    )
    bond_half_colours = list(atom_colours)

    adjacency: dict[int, list[tuple[int, Bond]]] = defaultdict(list)
    for bond in bonds:
        adjacency[bond.index_a].append((bond.index_b, bond))
        adjacency[bond.index_b].append((bond.index_a, bond))

    # Stacked bond arrays for vectorised geometry computation.
    n_bonds = len(bonds)
    bond_ia = np.empty(n_bonds, dtype=int)
    bond_ib = np.empty(n_bonds, dtype=int)
    bond_radii = np.empty(n_bonds)
    bond_index: dict[int, int] = {}
    for i, bond in enumerate(bonds):
        bond_ia[i] = bond.index_a
        bond_ib[i] = bond.index_b
        bond_radii[i] = bond.spec.radius
        bond_index[id(bond)] = i

    # ---- Polyhedra ----
    polyhedra = compute_polyhedra(
        species, coords, bonds, scene.polyhedra,
    )

    # Atoms hidden by AtomStyle.visible=False — always applied,
    # regardless of show_polyhedra.
    style_hidden_atoms: set[int] = set()
    style_hidden_bond_ids: set[int] = set()
    for i, sp in enumerate(species):
        style = scene.atom_styles.get(sp)
        if style is not None and not style.visible:
            style_hidden_atoms.add(i)
    if style_hidden_atoms:
        for bond in bonds:
            if bond.index_a in style_hidden_atoms or bond.index_b in style_hidden_atoms:
                style_hidden_bond_ids.add(id(bond))

    # Atoms/bonds hidden by polyhedra options (hide_centre, hide_bonds,
    # hide_vertices) — only applied when show_polyhedra is True.
    hidden_atoms: set[int] = set()
    hidden_bond_ids: set[int] = set()
    # For hide_vertices: an atom is hidden only if *every* polyhedron
    # it participates in has hide_vertices=True AND it has no bonds
    # to atoms outside those polyhedra (e.g. Li-O bonds keep O visible
    # even when Zr-O polyhedra hide vertices).
    vertex_hide_candidates: set[int] = set()
    vertex_keep: set[int] = set()
    poly_centres: set[int] = set()
    poly_members: set[int] = set()  # All centres + vertices in any polyhedron.
    for poly in polyhedra:
        poly_centres.add(poly.centre_index)
        poly_members.add(poly.centre_index)
        poly_members.update(poly.neighbour_indices)
        if poly.spec.hide_centre:
            hidden_atoms.add(poly.centre_index)
        # Always hide centre-to-vertex bonds when a polyhedron is
        # drawn.  These bonds are entirely inside the convex hull and
        # cannot be depth-sorted correctly against the polyhedral
        # faces in a painter's algorithm.
        neighbour_set = set(poly.neighbour_indices)
        for kk, bond in adjacency.get(poly.centre_index, []):
            if poly.spec.hide_bonds or kk in neighbour_set:
                hidden_bond_ids.add(id(bond))
        for ni in poly.neighbour_indices:
            if poly.spec.hide_vertices:
                vertex_hide_candidates.add(ni)
            else:
                vertex_keep.add(ni)
    # A vertex with bonds to non-polyhedron atoms must stay visible.
    for vi in vertex_hide_candidates - vertex_keep:
        for neighbour_idx, _ in adjacency.get(vi, []):
            if neighbour_idx not in poly_centres:
                vertex_keep.add(vi)
                break
    hidden_atoms |= vertex_hide_candidates - vertex_keep

    # Resolve rendering style per polyhedron.
    edge_width_override = (
        render_style.polyhedra_outline_width if render_style is not None
        else None
    )
    poly_render_data: list[_PolyhedronRenderData] = []
    for poly in polyhedra:
        if poly.spec.colour is not None:
            base_rgb = normalise_colour(poly.spec.colour)
        else:
            # Inherit from centre atom's resolved colour, which
            # accounts for colour_by / cmap when active.
            base_rgb = atom_colours[poly.centre_index]
        poly_render_data.append(_PolyhedronRenderData(
            base_colour=base_rgb,
            alpha=poly.spec.alpha,
            edge_colour=normalise_colour(poly.spec.edge_colour),
            edge_width=(
                edge_width_override if edge_width_override is not None
                else poly.spec.edge_width
            ),
        ))

    return _PrecomputedScene(
        coords=coords,
        radii_3d=radii_3d,
        atom_colours=atom_colours,
        bond_half_colours=bond_half_colours,
        adjacency=adjacency,
        bond_ia=bond_ia,
        bond_ib=bond_ib,
        bond_radii=bond_radii,
        bond_index=bond_index,
        polyhedra=polyhedra,
        style_hidden_atoms=style_hidden_atoms,
        style_hidden_bond_ids=style_hidden_bond_ids,
        hidden_atoms=hidden_atoms,
        hidden_bond_ids=hidden_bond_ids,
        poly_render_data=poly_render_data,
    )


def _apply_slab_clip(
    slab_visible: np.ndarray,
    slab_clip_mode: SlabClipMode,
    polyhedra_list: list,
    adjacency: dict[int, list[tuple[int, Bond]]],
    show_polyhedra: bool,
) -> tuple[np.ndarray, set[int], set[int]]:
    """Apply polyhedra-aware slab-clip overrides.

    Returns:
        Updated *slab_visible* array (copied if modified),
        *poly_skip* (polyhedron indices to skip entirely), and
        *poly_clip_hidden_bonds* (bond ``id()`` values to hide).
    """
    poly_skip: set[int] = set()
    poly_clip_hidden_bonds: set[int] = set()
    if (not show_polyhedra or not polyhedra_list
            or slab_clip_mode == SlabClipMode.PER_FACE):
        return slab_visible, poly_skip, poly_clip_hidden_bonds

    slab_force_visible: set[int] = set()
    for pi, poly in enumerate(polyhedra_list):
        all_vertices = set(poly.neighbour_indices) | {poly.centre_index}
        if slab_clip_mode == SlabClipMode.CLIP_WHOLE:
            if not all(slab_visible[v] for v in all_vertices):
                poly_skip.add(pi)
                for kk, bond in adjacency.get(poly.centre_index, []):
                    if kk in poly.neighbour_indices:
                        poly_clip_hidden_bonds.add(id(bond))
        elif slab_clip_mode == SlabClipMode.INCLUDE_WHOLE:
            if slab_visible[poly.centre_index]:
                slab_force_visible.update(all_vertices)
            else:
                poly_skip.add(pi)
    if slab_clip_mode == SlabClipMode.INCLUDE_WHOLE and slab_force_visible:
        slab_visible = slab_visible.copy()
        for v in slab_force_visible:
            slab_visible[v] = True
    return slab_visible, poly_skip, poly_clip_hidden_bonds


def _collect_polyhedra_faces(
    precomputed: _PrecomputedScene,
    polyhedra_list: list,
    poly_skip: set[int],
    slab_visible: np.ndarray,
    show_polyhedra: bool,
    polyhedra_shading: float,
    light_direction: np.ndarray,
    rotated: np.ndarray,
    depth: np.ndarray,
    xy: np.ndarray,
    order: np.ndarray,
) -> tuple[
    dict[int, list[tuple[np.ndarray, tuple, tuple, float, float]]],
    dict[int, int],
]:
    """Build per-face draw data and assign each face to a depth slot.

    Each face is slotted at its mean vertex depth.

    Returns:
        Tuple of ``(face_by_depth_slot, vertex_max_face_slot)`` where
        *face_by_depth_slot* maps depth-slot index to a list of
        ``(verts_2d, face_rgba, edge_rgba, edge_width, face_depth)``
        tuples sorted back-to-front, and *vertex_max_face_slot* maps
        each vertex atom index to the highest (most front-facing)
        depth-slot containing one of its connected faces.
    """
    face_by_depth_slot: dict[
        int, list[tuple[np.ndarray, tuple, tuple, float, float]]
    ] = defaultdict(list)
    vertex_max_face_slot: dict[int, int] = {}
    if not show_polyhedra or not polyhedra_list:
        return face_by_depth_slot, vertex_max_face_slot

    atom_depths_sorted = depth[order]
    for pi, poly in enumerate(polyhedra_list):
        if pi in poly_skip:
            continue
        prd = precomputed.poly_render_data[pi]
        base_rgb = prd.base_colour
        alpha = prd.alpha
        edge_rgb = prd.edge_colour
        edge_w = prd.edge_width
        for face_row in poly.faces:
            global_idx = [poly.neighbour_indices[j] for j in face_row]

            # Slab check: all vertices must be visible.
            # (In include_whole mode, slab_visible has already been
            # updated to force polyhedron vertices visible.)
            if not all(slab_visible[gi] for gi in global_idx):
                continue

            # Face normal from first two edges (works for any polygon).
            face_verts = rotated[global_idx]
            normal = np.cross(
                face_verts[1] - face_verts[0],
                face_verts[2] - face_verts[0],
            )
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                cos_angle = abs(np.dot(normal, light_direction) / norm_len)
            else:
                cos_angle = 0.0
            shading = 1.0 - polyhedra_shading * 0.6 * (1.0 - cos_angle)
            shaded = tuple(min(1.0, c * shading) for c in base_rgb)

            face_depth = np.mean(depth[global_idx])
            verts_2d = xy[global_idx]

            slot = int(np.searchsorted(atom_depths_sorted, face_depth))

            face_by_depth_slot[slot].append((
                verts_2d,
                (*shaded, alpha),
                (*edge_rgb, 1.0),
                edge_w,
                float(face_depth),
            ))

            # Track the latest (most front-facing) slot for each vertex.
            for gi in global_idx:
                prev = vertex_max_face_slot.get(gi, -1)
                if slot > prev:
                    vertex_max_face_slot[gi] = slot

    # Sort faces within each slot back-to-front (ascending depth).
    for slot in face_by_depth_slot:
        face_by_depth_slot[slot].sort(key=lambda entry: entry[4])

    return face_by_depth_slot, vertex_max_face_slot
