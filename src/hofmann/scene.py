"""Convenience constructors for StructureScene."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hofmann.defaults import default_atom_style, default_bond_specs
from hofmann.model import BondSpec, Frame, StructureScene, ViewState
from hofmann.parser import parse_bs, parse_mv

if TYPE_CHECKING:
    pass


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
    species, frame, atom_styles, bond_specs = parse_bs(bs_path)

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
        view=view,
        title=bs_path.stem,
    )


def _expand_pbc(
    structure,
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


def from_pymatgen(
    structure,
    bond_specs: list[BondSpec] | None = None,
    *,
    pbc: bool = False,
    pbc_cutoff: float | None = None,
) -> StructureScene:
    """Create a StructureScene from pymatgen Structure(s).

    Args:
        structure: A single pymatgen ``Structure`` or a list of
            ``Structure`` objects (e.g. from an MD trajectory).
        bond_specs: Optional bond specification rules.  If ``None``,
            sensible defaults are generated from covalent radii.
            Pass an empty list to disable bonds.
        pbc: If ``True``, add periodic image atoms at cell boundaries
            so that bonds crossing periodic boundaries are drawn.
        pbc_cutoff: Cartesian distance cutoff (angstroms) for PBC
            image generation.  Atoms within this distance of a cell
            face get an image on the opposite side.  If ``None``
            (the default), the maximum bond length from *bond_specs*
            is used.

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
    if pbc:
        frames = []
        for i, s in enumerate(structures):
            exp_species, exp_coords = _expand_pbc(s, bond_specs, pbc_cutoff)
            frames.append(Frame(coords=exp_coords, label=f"frame_{i}"))
        # Use expanded species from the first frame.
        species = exp_species
    else:
        frames = [
            Frame(coords=s.cart_coords, label=f"frame_{i}")
            for i, s in enumerate(structures)
        ]

    # Centre on first frame.
    centroid = np.mean(frames[0].coords, axis=0)
    view = ViewState(centre=centroid)

    return StructureScene(
        species=species,
        frames=frames,
        atom_styles=atom_styles,
        bond_specs=bond_specs,
        view=view,
        title="",
    )
