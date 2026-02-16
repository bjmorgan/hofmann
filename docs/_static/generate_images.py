"""Generate static images for the documentation."""

from pathlib import Path

import numpy as np

from hofmann import (
    AtomStyle, BondSpec, Frame, PolyhedronSpec,
    RenderStyle, StructureScene, ViewState,
)

OUT = Path(__file__).resolve().parent


def ch4_scene() -> StructureScene:
    """Build a CH4 scene from the test fixture."""
    fixture = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "ch4.bs"
    return StructureScene.from_xbs(fixture)


def _perovskite_structure():
    """Return the SrTiO3 pymatgen Structure and shared style parameters."""
    from pymatgen.core import Lattice, Structure as PmgStructure

    lattice = Lattice.cubic(3.905)
    return PmgStructure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],   # Sr (A-site)
            [0.5, 0.5, 0.5],   # Ti (B-site)
            [0.5, 0.5, 0.0],   # O
            [0.5, 0.0, 0.5],   # O
            [0.0, 0.5, 0.5],   # O
        ],
    )


def _style_perovskite(scene: StructureScene) -> None:
    """Apply shared atom colours and view to a perovskite scene."""
    scene.atom_styles["Sr"].colour = (0.0, 0.8, 0.2)
    scene.atom_styles["Sr"].radius = 1.4
    scene.atom_styles["Ti"].colour = (0.2, 0.4, 0.9)
    scene.atom_styles["Ti"].radius = 1.0
    scene.atom_styles["O"].colour = (0.9, 0.1, 0.1)
    scene.atom_styles["O"].radius = 0.8
    scene.view.look_along([1, 0.18, 0.2])


def perovskite_plain_scene() -> StructureScene:
    """Build SrTiO3 with default PBC but no polyhedra.

    Without polyhedra specs, neighbour-shell expansion is not
    triggered, so only atoms within *pbc_padding* of a cell face
    are replicated.  Bonds connect only existing atoms.
    """
    structure = _perovskite_structure()

    bond_specs = [
        BondSpec(species=("Ti", "O"), min_length=0.5, max_length=2.5,
                 radius=0.12, colour=(0.4, 0.4, 0.4)),
    ]

    scene = StructureScene.from_pymatgen(
        structure, bond_specs, centre_atom=0,
    )
    _style_perovskite(scene)
    return scene


def perovskite_scene() -> StructureScene:
    """Build SrTiO3 with PBC expansion and TiO6 polyhedra."""
    structure = _perovskite_structure()

    bond_specs = [
        BondSpec(species=("Ti", "O"), min_length=0.5, max_length=2.5,
                 radius=0.12, colour=(0.4, 0.4, 0.4)),
    ]
    polyhedra = [
        PolyhedronSpec(
            centre="Ti",
            colour=(0.5, 0.7, 1.0), alpha=1.0,
            edge_colour=(0.3, 0.3, 0.3),
            hide_centre=True, hide_bonds=True, hide_vertices=True,
        ),
    ]

    # Centre on Sr (index 0) so PBC expansion surrounds the A-site.
    scene = StructureScene.from_pymatgen(
        structure, bond_specs, polyhedra=polyhedra,
        pbc=True, pbc_padding=0.1, centre_atom=0,
    )
    _style_perovskite(scene)
    return scene


def si_scene() -> StructureScene:
    """Build a diamond-cubic Si scene from pymatgen."""
    from pymatgen.core import Lattice, Structure as PmgStructure

    lattice = Lattice.cubic(5.43)
    structure = PmgStructure(
        lattice,
        ["Si"] * 8,
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
    )

    bond_specs = [
        BondSpec(species=("Si", "Si"), min_length=0.0, max_length=2.8,
                 radius=0.1, colour=0.5),
    ]
    scene = StructureScene.from_pymatgen(structure, bond_specs, pbc=True, pbc_padding=0.1)
    scene.view.look_along([1, 1, 1])
    return scene


def llzo_scene() -> StructureScene:
    """Build an LLZO garnet scene from a bundled CIF file."""
    from pymatgen.core import Structure

    cif = Path(__file__).resolve().parent.parent.parent / "examples" / "Li7La3Zr2O12.cif"
    structure = Structure.from_file(str(cif))

    bond_specs = [
        BondSpec(species=("Li", "Li"), min_length=0.0, max_length=2.8,
                 radius=0.1, colour=1.0),
        BondSpec(species=("Zr", "O"), min_length=0.0, max_length=2.5,
                 radius=0.1, colour=0.5),
    ]
    polyhedra_specs = [
        PolyhedronSpec(centre="Zr", alpha=0.5, hide_vertices=True,
                       hide_centre=True, hide_bonds=True, min_vertices=6),
    ]

    scene = StructureScene.from_pymatgen(
        structure, bond_specs=bond_specs, polyhedra=polyhedra_specs,
        pbc=True, pbc_padding=0.1, centre_atom=95,
    )

    # Custom colours for clarity.
    scene.atom_styles["Li"].colour = (0.38, 0.71, 0.64)
    scene.atom_styles["La"].colour = (0.88, 0.43, 0.43)
    scene.atom_styles["O"].colour = (0.76, 0.10, 0.13)

    scene.view.look_along([1, 0, 0])
    scene.view.slab_near = -3.5
    scene.view.slab_far = 3.5

    return scene


def main() -> None:
    # CH4 -- simple ball-and-stick
    ch4 = ch4_scene()
    ch4.render_mpl(OUT / "ch4.svg", figsize=(4, 4), dpi=150)
    print(f"  wrote {OUT / 'ch4.svg'}")

    # Si diamond cubic -- pymatgen example
    si = si_scene()
    si.render_mpl(OUT / "si.svg", figsize=(4, 4), dpi=150)
    print(f"  wrote {OUT / 'si.svg'}")

    # SrTiO3 perovskite with polyhedra -- hero image
    perov = perovskite_scene()
    perov.render_mpl(
        OUT / "perovskite.svg",
        figsize=(6, 6), dpi=150, half_bonds=False,
        slab_clip_mode="include_whole",
    )
    print(f"  wrote {OUT / 'perovskite.svg'}")

    # Style variations for user guide
    perov_plain = perovskite_plain_scene()
    perov_plain.render_mpl(
        OUT / "perovskite_plain.svg",
        figsize=(4, 4), dpi=150,
    )
    print(f"  wrote {OUT / 'perovskite_plain.svg'}")

    perov_plain.render_mpl(
        OUT / "perovskite_spacefill.svg",
        figsize=(4, 4), dpi=150, atom_scale=1.0,
        show_bonds=False, show_polyhedra=False,
    )
    print(f"  wrote {OUT / 'perovskite_spacefill.svg'}")

    perov.render_mpl(
        OUT / "perovskite_no_outlines.svg",
        figsize=(4, 4), dpi=150, half_bonds=False,
        show_outlines=False, slab_clip_mode="include_whole",
    )
    print(f"  wrote {OUT / 'perovskite_no_outlines.svg'}")

    # LLZO garnet with ZrO6 polyhedra and slab clipping
    llzo = llzo_scene()
    style = RenderStyle(
        show_outlines=True,
        bond_outline_width=0.6,
        atom_outline_width=0.6,
        polyhedra_outline_width=0.6,
        slab_clip_mode="include_whole",
        half_bonds=False,
        show_axes=False,
    )
    llzo.render_mpl(
        OUT / "llzo.svg",
        figsize=(5, 5), dpi=150, style=style,
    )
    print(f"  wrote {OUT / 'llzo.svg'}")


if __name__ == "__main__":
    main()
