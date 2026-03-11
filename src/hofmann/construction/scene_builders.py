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
    from ase import Atoms
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


def from_ase(
    atoms: "Atoms | Sequence[Atoms]",
    bond_specs: list[BondSpec] | None = None,
    *,
    polyhedra: list[PolyhedronSpec] | None = None,
    centre_atom: int | None = None,
    atom_styles: dict[str, AtomStyle] | None = None,
    title: str = "",
    view: ViewState | None = None,
    atom_data: dict[str, np.ndarray] | None = None,
) -> StructureScene:
    """Create a StructureScene from ASE ``Atoms`` object(s).

    For periodic systems (where ``atoms.pbc`` is set and the cell is
    non-degenerate), fractional coordinates are wrapped to ``[0, 1)``
    and stored as Cartesian coordinates, following the same approach
    as :func:`from_pymatgen`.  For non-periodic systems, Cartesian
    positions are stored directly and ``lattice`` is ``None``.

    Args:
        atoms: A single ASE ``Atoms`` object or a sequence of ``Atoms``
            (e.g. from an MD trajectory or ``ase.io.Trajectory``).
        bond_specs: Bond detection rules.  ``None`` generates sensible
            defaults from VESTA bond length cutoffs; pass an empty list
            to disable bonds.
        polyhedra: Polyhedron rendering rules.  ``None`` disables
            polyhedra.
        centre_atom: Index of the atom to centre the unit cell on.
            Fractional coordinates are shifted so this atom sits at
            (0.5, 0.5, 0.5).  Only valid for periodic systems.
            If *view* is also provided, the explicit view takes
            precedence and only the fractional-coordinate shift is
            applied.
        atom_styles: Per-species style overrides.  When provided,
            these are merged on top of the auto-generated defaults
            so you only need to specify the species you want to
            customise.
        title: Scene title for display.
        view: Camera / projection state.  When ``None`` (the
            default), the view is auto-centred on the centre atom
            (if set) or the centroid of all atoms.
        atom_data: Per-atom metadata arrays, keyed by name.

    Returns:
        A StructureScene with default element styles.

    Raises:
        ImportError: If ASE is not installed.
        ValueError: If *atoms* is an empty sequence, if
            *centre_atom* is out of range, if *centre_atom* is used
            with a non-periodic system, or if frames in a trajectory
            have inconsistent species, atom counts, or periodicity.
    """
    try:
        from ase import Atoms
    except ImportError:
        raise ImportError(
            "ase is required for from_ase(). "
            "Install it with: pip install ase"
        )

    if isinstance(atoms, Atoms):
        atoms_list = [atoms]
    else:
        atoms_list = list(atoms)

    if not atoms_list:
        raise ValueError("atoms must not be empty")

    first = atoms_list[0]
    periodic = first.pbc.any() and first.cell.rank == 3

    if centre_atom is not None and not periodic:
        raise ValueError(
            "centre_atom is only supported for periodic systems"
        )

    species = first.get_chemical_symbols()

    # Validate trajectory consistency.
    for i, a in enumerate(atoms_list[1:], start=1):
        if len(a) != len(first):
            raise ValueError(
                f"all Atoms in a trajectory must have the same number "
                f"of atoms. Frame 0 has {len(first)} but frame {i} "
                f"has {len(a)}."
            )
        if a.get_chemical_symbols() != species:
            raise ValueError(
                f"all Atoms in a trajectory must have the same "
                f"species. Frame 0 has {species} but frame {i} has "
                f"{a.get_chemical_symbols()}."
            )
        frame_periodic = a.pbc.any() and a.cell.rank == 3
        if frame_periodic != periodic:
            raise ValueError(
                f"inconsistent periodicity in trajectory: frame 0 is "
                f"{'periodic' if periodic else 'non-periodic'} but "
                f"frame {i} is "
                f"{'periodic' if frame_periodic else 'non-periodic'}."
            )

    if centre_atom is not None:
        n_atoms = len(first)
        if not 0 <= centre_atom < n_atoms:
            raise ValueError(
                f"centre_atom {centre_atom} out of range for structure "
                f"with {n_atoms} atom(s)"
            )

    # Default atom styles from the element lookup, merged with overrides.
    unique_species = sorted(set(species))
    merged_styles = {sp: default_atom_style(sp) for sp in unique_species}
    if atom_styles is not None:
        merged_styles.update(atom_styles)

    # Generate default bond specs from VESTA cutoffs if none provided.
    if bond_specs is None:
        bond_specs = default_bond_specs(species)

    # Build frames.  For periodic systems, wrap fractional coordinates
    # to [0, 1) and store the lattice per frame (supporting NPT
    # trajectories with variable cell).
    frames = []
    if periodic:
        for i, a in enumerate(atoms_list):
            frac = a.get_scaled_positions(wrap=False) % 1.0
            if centre_atom is not None:
                shift = 0.5 - frac[centre_atom]
                frac = (frac + shift) % 1.0
            coords = frac @ a.cell.array
            frames.append(Frame(
                coords=coords,
                lattice=a.cell.array.copy(),
                label=f"frame_{i}",
            ))
    else:
        for i, a in enumerate(atoms_list):
            frames.append(
                Frame(coords=a.positions.copy(), label=f"frame_{i}"),
            )

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
        atom_data=dict(atom_data) if atom_data is not None else {},
    )


def from_pymatgen(
    structure: "Structure | Sequence[Structure]",
    bond_specs: list[BondSpec] | None = None,
    *,
    polyhedra: list[PolyhedronSpec] | None = None,
    centre_atom: int | None = None,
    atom_styles: dict[str, AtomStyle] | None = None,
    title: str = "",
    view: ViewState | None = None,
    atom_data: dict[str, np.ndarray] | None = None,
) -> StructureScene:
    """Create a StructureScene from pymatgen Structure(s).

    Fractional coordinates are wrapped to ``[0, 1)`` and stored as
    Cartesian coordinates.  Periodic boundary handling (PBC bond
    computation, image-atom expansion, recursive depth, molecule
    deduplication) is controlled at render time via
    :class:`~hofmann.model.RenderStyle`.

    Args:
        structure: A single pymatgen ``Structure`` or a list of
            ``Structure`` objects (e.g. from an MD trajectory).
        bond_specs: Optional bond specification rules.  If ``None``,
            sensible defaults are generated from VESTA bond length
            cutoffs.  Pass an empty list to disable bonds.
        polyhedra: Optional polyhedron rendering rules.  If ``None``,
            no polyhedra are drawn.
        centre_atom: Index of the atom to centre the unit cell on.
            When set, all fractional coordinates are shifted so that
            this atom sits at (0.5, 0.5, 0.5), and the view is
            centred on this atom.  If *view* is also provided, the
            explicit view takes precedence and only the fractional-
            coordinate shift is applied.
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

    # Build frames.  Wrap fractional coordinates to [0, 1) so atoms
    # sit inside the unit cell for consistent periodic bond
    # computation at render time.  Physical atoms only — periodic
    # expansion is handled at render time by the periodic bond
    # pipeline (compute_bonds + build_rendering_set).
    frames = []
    for i, s in enumerate(structures):
        frac = s.frac_coords % 1.0
        wrapped_coords = frac @ s.lattice.matrix
        frames.append(Frame(
            coords=wrapped_coords,
            lattice=s.lattice.matrix.copy(),
            label=f"frame_{i}",
        ))

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
        atom_data=dict(atom_data) if atom_data is not None else {},
    )
