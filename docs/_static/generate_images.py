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


def perovskite_scene() -> StructureScene:
    """Build a 2x2x2 SrTiO3 perovskite supercell with polyhedra."""
    a = 3.905

    frac_positions = {
        "Sr": [(0.0, 0.0, 0.0)],
        "Ti": [(0.5, 0.5, 0.5)],
        "O":  [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
    }

    species = []
    coords_list = []
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                offset = np.array([dx, dy, dz]) * a
                for sp, positions in frac_positions.items():
                    for frac in positions:
                        species.append(sp)
                        coords_list.append(np.array(frac) * a + offset)

    coords = np.array(coords_list)

    atom_styles = {
        "Sr": AtomStyle(radius=1.4, colour=(0.0, 0.8, 0.2)),
        "Ti": AtomStyle(radius=1.0, colour=(0.2, 0.4, 0.9)),
        "O":  AtomStyle(radius=0.8, colour=(0.9, 0.1, 0.1)),
    }

    bond_specs = [
        BondSpec(species=("Ti", "O"), min_length=0.5, max_length=2.5,
                 radius=0.12, colour=(0.4, 0.4, 0.4)),
    ]

    polyhedra = [
        PolyhedronSpec(
            centre="Ti",
            colour=(0.5, 0.7, 1.0), alpha=0.25,
            edge_colour=(0.3, 0.3, 0.3), edge_width=0.8,
        ),
    ]

    angle_x = np.radians(25)
    angle_y = np.radians(-35)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)],
    ])
    ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])
    rotation = ry @ rx

    centroid = coords.mean(axis=0)
    view = ViewState(rotation=rotation, centre=centroid, zoom=1.0)

    return StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles=atom_styles,
        bond_specs=bond_specs,
        polyhedra=polyhedra,
        view=view,
    )


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
        PolyhedronSpec(centre="Zr", alpha=1.0, hide_vertices=True,
                       hide_centre=True, hide_bonds=True, min_vertices=6),
    ]

    scene = StructureScene.from_pymatgen(
        structure, bond_specs=bond_specs, polyhedra=polyhedra_specs,
        pbc=True, pbc_cutoff=0.1, centre_atom=95,
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
    ch4.render_mpl(OUT / "ch4.svg", show=False, figsize=(4, 4), dpi=150)
    print(f"  wrote {OUT / 'ch4.svg'}")

    # SrTiO3 perovskite with polyhedra -- hero image
    perov = perovskite_scene()
    perov.render_mpl(
        OUT / "perovskite.svg", show=False,
        figsize=(6, 6), dpi=150, half_bonds=False,
    )
    print(f"  wrote {OUT / 'perovskite.svg'}")

    # Style variations for user guide
    perov.render_mpl(
        OUT / "perovskite_spacefill.svg", show=False,
        figsize=(4, 4), dpi=150, atom_scale=1.0,
        show_bonds=False, show_polyhedra=False,
    )
    print(f"  wrote {OUT / 'perovskite_spacefill.svg'}")

    perov.render_mpl(
        OUT / "perovskite_no_outlines.svg", show=False,
        figsize=(4, 4), dpi=150, half_bonds=False,
        show_outlines=False,
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
        circle_segments=72,
        arc_segments=12,
    )
    llzo.render_mpl(
        OUT / "llzo.svg", show=False,
        figsize=(5, 5), dpi=150, style=style,
    )
    print(f"  wrote {OUT / 'llzo.svg'}")


if __name__ == "__main__":
    main()
