"""Convenience constructors for StructureScene."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hofmann.construction.defaults import default_atom_style, default_bond_specs
from hofmann.model import (
    AtomStyle, BondSpec, Frame, PolyhedronSpec, StructureScene, ViewState,
)
from hofmann.construction.parser import parse_bs, parse_mv

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


def from_pymatgen(
    structure: "Structure | Sequence[Structure]",
    bond_specs: list[BondSpec] | None = None,
    *,
    polyhedra: list[PolyhedronSpec] | None = None,
    pbc: bool = True,
    pbc_padding: float | None = 0.1,
    centre_atom: int | None = None,
    max_recursive_depth: int = 5,
    deduplicate_molecules: bool = False,
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
        pbc: If ``True`` (the default), the renderer uses the lattice
            for periodic bond computation and image-atom expansion.
            Set to ``False`` to disable all periodic boundary
            handling.
        pbc_padding: Cartesian margin (angstroms) around the unit
            cell for geometric cell-face expansion.  Atoms within
            this distance of a cell face are duplicated on the
            opposite side.  The default of 0.1 angstroms is suitable
            for most structures.  ``None`` disables geometric
            expansion.
        centre_atom: Index of the atom to centre the unit cell on.
            When set, all fractional coordinates are shifted so that
            this atom sits at (0.5, 0.5, 0.5), and the view is
            centred on this atom.  If *view* is also provided, the
            explicit view takes precedence and only the fractional-
            coordinate shift is applied.
        max_recursive_depth: Maximum number of iterations for
            recursive bond expansion (must be >= 1).  Only relevant
            when one or more *bond_specs* have ``recursive=True``.
        deduplicate_molecules: If ``True``, molecules that span cell
            boundaries are shown only once.  Orphaned fragments are
            removed in favour of the most-connected cluster.
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

    # Build frames.  Physical atoms only — periodic expansion is
    # handled at render time by the periodic bond pipeline
    # (compute_bonds + build_rendering_set).
    if pbc:
        frames = []
        for i, s in enumerate(structures):
            # Wrap fractional coordinates to [0, 1) so atoms sit
            # inside the unit cell for consistent periodic bond
            # computation at render time.
            frac = s.frac_coords % 1.0
            wrapped_coords = frac @ s.lattice.matrix
            frames.append(Frame(coords=wrapped_coords, label=f"frame_{i}"))
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
        pbc=pbc,
        pbc_padding=pbc_padding,
        max_recursive_depth=max_recursive_depth,
        deduplicate_molecules=deduplicate_molecules,
        atom_data=dict(atom_data) if atom_data is not None else {},
    )
