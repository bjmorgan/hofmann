"""Convenience constructors for StructureScene."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hofmann.defaults import default_atom_style
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


def from_pymatgen(
    structure,
    bond_specs: list[BondSpec] | None = None,
) -> StructureScene:
    """Create a StructureScene from pymatgen Structure(s).

    Args:
        structure: A single pymatgen ``Structure`` or a list of
            ``Structure`` objects (e.g. from an MD trajectory).
        bond_specs: Optional bond specification rules. If ``None``,
            no bonds are drawn.

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

    # Extract species labels from the first structure.
    species = [str(site.specie) for site in structures[0]]

    # Build frames from each structure's Cartesian coordinates.
    frames = [
        Frame(coords=s.cart_coords, label=f"frame_{i}")
        for i, s in enumerate(structures)
    ]

    # Default atom styles from the element lookup.
    unique_species = sorted(set(species))
    atom_styles = {sp: default_atom_style(sp) for sp in unique_species}

    # Centre on first frame.
    centroid = np.mean(frames[0].coords, axis=0)
    view = ViewState(centre=centroid)

    return StructureScene(
        species=species,
        frames=frames,
        atom_styles=atom_styles,
        bond_specs=bond_specs or [],
        view=view,
        title="",
    )
