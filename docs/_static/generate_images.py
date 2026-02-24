"""Generate static images for the documentation."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hofmann import (
    AtomStyle, AxesStyle, BondSpec, Frame, LegendItem, LegendStyle,
    PolyhedronSpec, RenderStyle, StructureScene, ViewState,
)

OUT = Path(__file__).resolve().parent


def ch4_scene() -> StructureScene:
    """Build a CH4 scene from the test fixture."""
    fixture = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "ch4.bs"
    return StructureScene.from_xbs(fixture)


def octahedron_scene(**poly_kwargs) -> StructureScene:
    """Build a TiO6 octahedron scene for visual examples."""
    species = ["Ti"] + ["O"] * 6
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -2.0],
    ])
    scene = StructureScene(
        species=species,
        frames=[Frame(coords=coords)],
        atom_styles={
            "Ti": AtomStyle(1.0, "#477B9D"),
            "O": AtomStyle(0.8, "#F03F37"),
        },
        bond_specs=[BondSpec(
            species=("Ti", "O"), min_length=0.0, max_length=3.0,
            radius=0.1, colour=0.5,
        )],
        polyhedra=[PolyhedronSpec(centre="Ti", **poly_kwargs)],
    )
    scene.view.look_along([1, 0.3, 0.15])
    return scene


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
    scene.atom_styles["Sr"].colour = "#51b04d"
    scene.atom_styles["Sr"].radius = 1.4
    scene.atom_styles["Ti"].colour = "#477B9D"
    scene.atom_styles["Ti"].radius = 1.0
    scene.atom_styles["O"].colour = "#F03F37"
    scene.atom_styles["O"].radius = 0.8
    scene.view.look_along([1, 0.18, 0.2])


def perovskite_plain_scene() -> StructureScene:
    """Build SrTiO3 with default PBC but no polyhedra.

    Without polyhedra specs, only bonds between physical atoms and
    their periodic images are drawn.  The ``complete`` and
    ``recursive`` settings on bond specs control image-atom
    expansion.
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
            colour=(0.5, 0.7, 1.0), alpha=0.3,
            edge_colour=(0.3, 0.3, 0.3),
        ),
    ]

    # Centre on Sr (index 0) so PBC expansion surrounds the A-site.
    scene = StructureScene.from_pymatgen(
        structure, bond_specs, polyhedra=polyhedra,
        centre_atom=0,
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
    scene = StructureScene.from_pymatgen(structure, bond_specs)
    scene.view.look_along([1, 1, 1])
    return scene


def rutile_scene() -> StructureScene:
    """Build a rutile TiO2 scene from pymatgen."""
    from pymatgen.core import Lattice, Structure as PmgStructure

    # Tetragonal rutile: a = b = 4.594, c = 2.959
    lattice = Lattice.tetragonal(4.594, 2.959)
    structure = PmgStructure(
        lattice,
        ["Ti", "Ti", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.3053, 0.3053, 0.0],
            [0.6947, 0.6947, 0.0],
            [0.1947, 0.8053, 0.5],
            [0.8053, 0.1947, 0.5],
        ],
    )

    bond_specs = [
        BondSpec(species=("Ti", "O"), min_length=0.5, max_length=2.1,
                 radius=0.1, colour=(0.4, 0.4, 0.4)),
    ]
    scene = StructureScene.from_pymatgen(
        structure, bond_specs,
    )
    scene.atom_styles["Ti"].colour = "#477B9D"
    scene.atom_styles["Ti"].radius = 1.0
    scene.atom_styles["O"].colour = "#F03F37"
    scene.atom_styles["O"].radius = 0.8
    return scene


def pbc_bonds_scene(
    *, complete: bool = False, recursive: bool = False,
    polyhedra: bool = False, molecular: bool = True,
) -> StructureScene:
    """Build the Zr-S-N-H structure for bond completion examples.

    The rendered scene uses the default ``pbc_padding=0.1`` for
    geometric cell-face expansion.  The *complete* and *recursive*
    flags control additional bond-driven image-atom expansion.

    Args:
        complete: If ``True``, set ``complete="Zr"`` on the S-Zr bond
            spec; if ``False``, leave bond completion disabled.
        recursive: If ``True``, set ``recursive=True`` on the N-N and
            H-N bond specs; if ``False``, leave those bonds
            non-recursive.
        polyhedra: Include ZrS6 polyhedra with hidden centres.
        molecular: Include N-N and H-N bond specs.  Set to ``False``
            to show only the Zr-S network.
    """
    from pymatgen.core import Structure as PmgStructure

    vasp = Path(__file__).resolve().parent.parent.parent / "examples" / "Zr4S12N8H24.vasp"
    structure = PmgStructure.from_file(str(vasp))

    bond_specs = [
        BondSpec(species=("S", "Zr"), min_length=0.0, max_length=2.9,
                 radius=0.1, colour=0.5,
                 complete="Zr" if complete else False),
        BondSpec(species=("N", "N"), min_length=0.0, max_length=1.9,
                 radius=0.1, colour=0.5, recursive=recursive),
        BondSpec(species=("H", "N"), min_length=0.0, max_length=1.2,
                 radius=0.1, colour=0.5, recursive=recursive),
    ]

    poly_specs = []
    if polyhedra:
        poly_specs = [
            PolyhedronSpec(
                centre="Zr", colour=(0.55, 0.71, 0.67), alpha=0.4,
                hide_centre=True,
            ),
        ]

    scene = StructureScene.from_pymatgen(
        structure, bond_specs, polyhedra=poly_specs,
    )
    scene.atom_styles["Zr"].colour = (0.55, 0.71, 0.67)
    scene.atom_styles["N"].colour = (0.35, 0.55, 0.75)
    if not molecular:
        scene.atom_styles["N"].visible = False
        scene.atom_styles["H"].visible = False
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
        centre_atom=95,
    )

    # Custom colours for clarity.
    scene.atom_styles["Li"].colour = (0.38, 0.71, 0.64)
    scene.atom_styles["La"].colour = (0.88, 0.43, 0.43)
    scene.atom_styles["O"].colour = (0.76, 0.10, 0.13)

    scene.view.look_along([1, 0, 0])
    scene.view.slab_near = -3.5
    scene.view.slab_far = 3.5

    return scene


def logo_scene() -> StructureScene:
    """Build a dodecahedron scene for the project logo."""
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi

    verts_orig = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        [0,  phi,  inv_phi], [0,  phi, -inv_phi],
        [0, -phi,  inv_phi], [0, -phi, -inv_phi],
        [ inv_phi, 0,  phi], [-inv_phi, 0,  phi],
        [ inv_phi, 0, -phi], [-inv_phi, 0, -phi],
        [ phi,  inv_phi, 0], [ phi, -inv_phi, 0],
        [-phi,  inv_phi, 0], [-phi, -inv_phi, 0],
    ])
    edge_len = 2 * inv_phi
    n_verts = len(verts_orig)

    # Spin around the view direction for a pleasing orientation.
    look_dir = np.array([0.3, 0.5, 1.0])
    look_dir = look_dir / np.linalg.norm(look_dir)
    R_spin = Rotation.from_rotvec(look_dir * np.radians(261.3)).as_matrix()
    verts = (R_spin @ verts_orig.T).T

    # Colour by projected screen x-coordinate (reversed: warm left,
    # cool right) so the gradient follows the inferno colourmap.
    right_ax = np.cross([0, 1, 0], look_dir)
    right_ax /= np.linalg.norm(right_ax)
    proj_x = verts @ right_ax
    colour_vals = 1.0 - (proj_x - proj_x.min()) / (proj_x.max() - proj_x.min())

    # Hidden centre atom to anchor dodecahedron polyhedra faces.
    centre = np.array([[0.0, 0.0, 0.0]])
    all_coords = np.vstack([centre, verts])
    species = ["M"] + ["V"] * n_verts

    scene = StructureScene(
        species=species,
        frames=[Frame(coords=all_coords)],
        atom_styles={
            "M": AtomStyle(0.01, (0.5, 0.5, 0.5)),
            "V": AtomStyle(0.40, (0.5, 0.5, 0.5)),
        },
        bond_specs=[
            BondSpec(species=("V", "V"), min_length=0.0,
                     max_length=edge_len + 0.05, radius=0.03, colour=0.15),
            BondSpec(species=("M", "V"), min_length=0.0,
                     max_length=2.5, radius=0.01, colour=0.5),
        ],
        polyhedra=[PolyhedronSpec(
            centre="M", colour=(0.5, 0.5, 0.5), alpha=0.05,
            hide_bonds=True, hide_centre=True, hide_vertices=False,
        )],
    )

    data = dict(zip(range(1, 1 + n_verts), colour_vals))
    scene.set_atom_data("gradient", data)

    scene.view.look_along(look_dir)
    scene.view.perspective = 0.12
    scene.view.view_distance = 5.0

    return scene


def main() -> None:
    # Project logo: dodecahedron with inferno gradient
    logo = logo_scene()
    logo_style = RenderStyle(polyhedra_shading=1.0, half_bonds=False)
    logo_kw = dict(colour_by="gradient", cmap="inferno", show=False,
                   style=logo_style)
    repo_root = OUT.parent.parent
    logo.render_mpl(repo_root / "logo.svg", figsize=(3, 3), dpi=150,
                    **logo_kw)
    print(f"  wrote {repo_root / 'logo.svg'}")
    logo.render_mpl(repo_root / "logo.png", figsize=(3, 3), dpi=300,
                    **logo_kw)
    print(f"  wrote {repo_root / 'logo.png'}")

    # CH4 -- simple ball-and-stick
    ch4 = ch4_scene()
    ch4.render_mpl(OUT / "ch4.svg", figsize=(4, 4), dpi=150)
    print(f"  wrote {OUT / 'ch4.svg'}")

    # Si diamond cubic -- pymatgen example
    si = si_scene()
    si.render_mpl(OUT / "si.svg", figsize=(4, 4), dpi=150)
    print(f"  wrote {OUT / 'si.svg'}")

    # PBC bond completion -- Zr-S network only
    pbc_kw = dict(figsize=(4, 4), dpi=150, show=False,
                  show_axes=False, atom_scale=0.4, half_bonds=False)
    # Zr-S only, no completion: missing bonds at boundary.
    plain = pbc_bonds_scene(molecular=False)
    plain.render_mpl(OUT / "pbc_bonds_plain.svg", **pbc_kw)
    print(f"  wrote {OUT / 'pbc_bonds_plain.svg'}")
    # Zr-S with complete="Zr": boundary bonds filled in.
    comp = pbc_bonds_scene(complete=True, molecular=False)
    comp.render_mpl(OUT / "pbc_bonds_complete.svg", **pbc_kw)
    print(f"  wrote {OUT / 'pbc_bonds_complete.svg'}")
    # Full structure without recursive: N2H6 molecules broken.
    no_rec = pbc_bonds_scene(complete=True)
    no_rec.render_mpl(OUT / "pbc_bonds_no_recursive.svg", **pbc_kw)
    print(f"  wrote {OUT / 'pbc_bonds_no_recursive.svg'}")
    # Full structure with recursive: N2H6 molecules completed.
    rec = pbc_bonds_scene(complete=True, recursive=True)
    rec.render_mpl(OUT / "pbc_bonds_recursive.svg", **pbc_kw)
    print(f"  wrote {OUT / 'pbc_bonds_recursive.svg'}")
    # Full structure with recursive + deduplication: duplicate molecules removed.
    dedup = pbc_bonds_scene(complete=True, recursive=True)
    dedup.render_mpl(OUT / "pbc_bonds_deduplicated.svg",
                     deduplicate_molecules=True, **pbc_kw)
    print(f"  wrote {OUT / 'pbc_bonds_deduplicated.svg'}")

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

    perov_plain.render_mpl(
        OUT / "perovskite_no_outlines.svg",
        figsize=(4, 4), dpi=150,
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
        show_axes=False,
    )
    llzo.render_mpl(
        OUT / "llzo.svg",
        figsize=(5, 5), dpi=150, style=style,
    )
    print(f"  wrote {OUT / 'llzo.svg'}")

    # --- Visual examples for user guide settings ---

    # 1. Polyhedra shading: flat vs full Lambertian
    octa_shading = octahedron_scene(
        colour=(0.5, 0.7, 1.0), alpha=0.8,
        edge_colour=(0.15, 0.15, 0.15),
    )
    octa_shading.render_mpl(
        OUT / "octahedron_shading_flat.svg",
        figsize=(3, 3), dpi=150, polyhedra_shading=0.0,
    )
    print(f"  wrote {OUT / 'octahedron_shading_flat.svg'}")
    octa_shading.render_mpl(
        OUT / "octahedron_shading_full.svg",
        figsize=(3, 3), dpi=150, polyhedra_shading=1.0,
    )
    print(f"  wrote {OUT / 'octahedron_shading_full.svg'}")

    # 2. Polyhedra vertex ordering: opaque and transparent examples.
    octa_in_front = octahedron_scene(
        colour=(0.5, 0.7, 1.0), alpha=1.0,
        edge_colour=(0.15, 0.15, 0.15),
    )
    octa_in_front.render_mpl(
        OUT / "octahedron_vertex_in_front.svg",
        figsize=(3, 3), dpi=150,
    )
    print(f"  wrote {OUT / 'octahedron_vertex_in_front.svg'}")
    octa_in_front_trans = octahedron_scene(
        colour=(0.5, 0.7, 1.0), alpha=0.4,
        edge_colour=(0.15, 0.15, 0.15),
    )
    octa_in_front_trans.render_mpl(
        OUT / "octahedron_vertex_in_front_transparent.svg",
        figsize=(3, 3), dpi=150,
    )
    print(f"  wrote {OUT / 'octahedron_vertex_in_front_transparent.svg'}")

    # 3. Slab clipping modes on LLZO garnet
    #    The LLZO scene already has a depth slab that clips through
    #    several ZrO6 octahedra, making the three modes clearly distinct.
    llzo_clip = llzo_scene()
    clip_style = RenderStyle(
        show_outlines=True,
        bond_outline_width=0.6,
        atom_outline_width=0.6,
        polyhedra_outline_width=0.6,
        half_bonds=False,
        show_axes=False,
    )
    clip_names = {
        "per_face": "llzo_clip_per_face.svg",
        "clip_whole": "llzo_clip_whole.svg",
        "include_whole": "llzo_clip_include_whole.svg",
    }
    for mode, filename in clip_names.items():
        clip_style.slab_clip_mode = mode
        llzo_clip.render_mpl(
            OUT / filename,
            figsize=(3, 3), dpi=150, style=clip_style,
        )
        print(f"  wrote {OUT / filename}")

    # 4. Half-bonds: on vs off
    #    Build without polyhedra so bonds are visible.
    octa_bonds = StructureScene(
        species=["Ti"] + ["O"] * 6,
        frames=[Frame(coords=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ]))],
        atom_styles={
            "Ti": AtomStyle(1.0, "#477B9D"),
            "O": AtomStyle(0.8, "#F03F37"),
        },
        bond_specs=[BondSpec(
            species=("Ti", "O"), min_length=0.0, max_length=3.0,
            radius=0.1, colour=0.5,
        )],
    )
    octa_bonds.view.look_along([1, 0.3, 0.15])
    octa_bonds.render_mpl(
        OUT / "octahedron_half_bonds.svg",
        figsize=(3, 3), dpi=150, half_bonds=True,
    )
    print(f"  wrote {OUT / 'octahedron_half_bonds.svg'}")
    octa_bonds.render_mpl(
        OUT / "octahedron_no_half_bonds.svg",
        figsize=(3, 3), dpi=150, half_bonds=False,
    )
    print(f"  wrote {OUT / 'octahedron_no_half_bonds.svg'}")

    # 5. Perspective: orthographic vs perspective (perovskite)
    perov_plain.render_mpl(
        OUT / "perovskite_ortho.svg",
        figsize=(3, 3), dpi=150,
    )
    print(f"  wrote {OUT / 'perovskite_ortho.svg'}")
    perov_plain.view.perspective = 0.5
    perov_plain.render_mpl(
        OUT / "perovskite_perspective.svg",
        figsize=(3, 3), dpi=150,
    )
    print(f"  wrote {OUT / 'perovskite_perspective.svg'}")
    perov_plain.view.perspective = 0.0  # Reset

    # 6–9. Per-atom colouring examples: ring of atoms.
    n = 16
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 3.0
    ring_coords = np.column_stack([
        r * np.cos(angles),
        r * np.sin(angles),
        np.zeros(n),
    ])
    ring_species = ["X"] * n

    # Bond each atom to its neighbour around the ring.
    chord = 2 * r * np.sin(np.pi / n)
    ring_bond = BondSpec(
        species=("X", "X"), min_length=0.0,
        max_length=chord + 0.1, radius=0.08, colour=0.5,
    )
    colour_by_style = dict(show=False, figsize=(3, 3), dpi=150,
                           half_bonds=False)

    # 6. Continuous: colour by angle.
    cont_scene = StructureScene(
        species=ring_species,
        frames=[Frame(coords=ring_coords)],
        atom_styles={"X": AtomStyle(0.7, "grey")},
        bond_specs=[ring_bond],
    )
    cont_scene.set_atom_data("angle", np.degrees(angles))
    cont_scene.render_mpl(
        OUT / "colour_by_continuous.svg",
        colour_by="angle", cmap="twilight",
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_continuous.svg'}")

    # 7. Categorical: alternate labels around the ring.
    cat_scene = StructureScene(
        species=ring_species,
        frames=[Frame(coords=ring_coords)],
        atom_styles={"X": AtomStyle(0.7, "grey")},
        bond_specs=[ring_bond],
    )
    labels = ["alpha", "beta", "gamma", "delta"] * (n // 4)
    cat_scene.set_atom_data("site", labels)
    cat_scene.render_mpl(
        OUT / "colour_by_categorical.svg",
        colour_by="site", cmap="Set2",
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_categorical.svg'}")

    # 8. Custom colouring function: red-to-blue interpolation.
    custom_scene = StructureScene(
        species=ring_species,
        frames=[Frame(coords=ring_coords)],
        atom_styles={"X": AtomStyle(0.7, "grey")},
        bond_specs=[ring_bond],
    )
    custom_scene.set_atom_data("angle", np.degrees(angles))

    def red_blue(t: float) -> tuple[float, float, float]:
        return (1.0 - t, 0.0, t)

    custom_scene.render_mpl(
        OUT / "colour_by_custom.svg",
        colour_by="angle", cmap=red_blue,
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_custom.svg'}")

    # 9. Multiple colouring layers: two concentric rings.
    #    Outer ring coloured by categorical type, inner by numerical value.
    n_outer, n_inner = 12, 8
    n_total = n_outer + n_inner
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    r_outer, r_inner = 3.5, 1.8
    outer_coords = np.column_stack([
        r_outer * np.cos(outer_angles),
        r_outer * np.sin(outer_angles),
        np.zeros(n_outer),
    ])
    inner_coords = np.column_stack([
        r_inner * np.cos(inner_angles),
        r_inner * np.sin(inner_angles),
        np.zeros(n_inner),
    ])
    multi_coords = np.vstack([outer_coords, inner_coords])
    # Use separate species so bonds only form within each ring.
    multi_species = ["A"] * n_outer + ["B"] * n_inner

    outer_chord = 2 * r_outer * np.sin(np.pi / n_outer)
    inner_chord = 2 * r_inner * np.sin(np.pi / n_inner)
    multi_scene = StructureScene(
        species=multi_species,
        frames=[Frame(coords=multi_coords)],
        atom_styles={
            "A": AtomStyle(0.7, "grey"),
            "B": AtomStyle(0.7, "grey"),
        },
        bond_specs=[
            BondSpec(species=("A", "A"), min_length=0.0,
                     max_length=outer_chord + 0.1, radius=0.08, colour=0.5),
            BondSpec(species=("B", "B"), min_length=0.0,
                     max_length=inner_chord + 0.1, radius=0.08, colour=0.5),
        ],
    )
    # Outer ring: categorical type labels.
    type_dict: dict[int, object] = {}
    for i in range(n_outer):
        type_dict[i] = ["Fe", "Co", "Ni"][i % 3]
    multi_scene.set_atom_data("metal", type_dict)
    # Inner ring: numerical charge.
    charge_dict: dict[int, object] = {}
    for i in range(n_inner):
        charge_dict[n_outer + i] = float(i) / max(n_inner - 1, 1)
    multi_scene.set_atom_data("charge", charge_dict)
    multi_scene.render_mpl(
        OUT / "colour_by_multi.svg",
        colour_by=["metal", "charge"],
        cmap=["Set2", "YlOrRd"],
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_multi.svg'}")

    # 10. Polyhedra inheriting colour_by: ring of corner-sharing tetrahedra.
    #     Build regular tetrahedra first, then place centres at centroids.
    n_tet = 8
    tet_edge = 2.5
    tet_ring_r = tet_edge / (2 * np.sin(np.pi / n_tet))

    # Shared (bridging) vertices on a ring in the z=0 plane.
    shared_angles = np.linspace(0, 2 * np.pi, n_tet, endpoint=False)
    shared_verts = np.column_stack([
        tet_ring_r * np.cos(shared_angles),
        tet_ring_r * np.sin(shared_angles),
        np.zeros(n_tet),
    ])

    # For each tetrahedron, compute the other two vertices so all six
    # edges equal tet_edge, then place the centre at the centroid.
    all_tet_verts: list[np.ndarray] = []
    all_centroids: list[np.ndarray] = []
    for i in range(n_tet):
        va = shared_verts[i]
        vb = shared_verts[(i + 1) % n_tet]
        mid = (va + vb) / 2
        u = (vb - va) / np.linalg.norm(vb - va)
        n1 = np.cross(u, [0, 0, 1])
        if np.linalg.norm(n1) < 1e-10:
            n1 = np.cross(u, [1, 0, 0])
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(u, n1)
        d_perp = tet_edge * np.sqrt(3) / 2
        h2 = tet_edge / 2
        h1 = np.sqrt(d_perp**2 - h2**2)
        vc = mid + h1 * n1 + h2 * n2
        vd = mid + h1 * n1 - h2 * n2
        verts = np.array([va, vb, vc, vd])
        all_tet_verts.append(verts)
        all_centroids.append(verts.mean(axis=0))

    poly_species: list[str] = []
    poly_coords: list[list[float]] = []

    # Centres (at centroids).
    for i in range(n_tet):
        poly_coords.append(all_centroids[i].tolist())
        poly_species.append("M")

    # Shared bridging vertices.
    for i in range(n_tet):
        poly_coords.append(shared_verts[i].tolist())
        poly_species.append("O")

    # Non-bridging vertices (C, D for each tetrahedron).
    for i in range(n_tet):
        poly_coords.append(all_tet_verts[i][2].tolist())
        poly_species.append("O")
        poly_coords.append(all_tet_verts[i][3].tolist())
        poly_species.append("O")

    centroid_to_vert = tet_edge * np.sqrt(3.0 / 8.0)
    poly_scene = StructureScene(
        species=poly_species,
        frames=[Frame(coords=np.array(poly_coords))],
        atom_styles={
            "M": AtomStyle(0.45, "grey"),
            "O": AtomStyle(0.3, (0.6, 0.6, 0.6)),
        },
        bond_specs=[BondSpec(
            species=("M", "O"), min_length=0.0,
            max_length=centroid_to_vert + 0.2,
            radius=0.06, colour=0.5,
        )],
        polyhedra=[PolyhedronSpec(
            centre="M", alpha=0.4,
            hide_bonds=True,
        )],
    )
    poly_scene.view.look_along([0.15, 0.05, 1])

    poly_vals = np.full(len(poly_species), np.nan)
    for i in range(n_tet):
        poly_vals[i] = float(i) / (n_tet - 1)
    poly_scene.set_atom_data("val", poly_vals)

    # With atoms visible -- shows inheritance clearly.
    poly_scene.render_mpl(
        OUT / "colour_by_polyhedra_atoms.svg",
        colour_by="val", cmap="coolwarm",
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_polyhedra_atoms.svg'}")

    # Without atoms -- typical usage.
    poly_scene.polyhedra[0].hide_centre = True
    poly_scene.polyhedra[0].hide_vertices = True
    poly_scene.render_mpl(
        OUT / "colour_by_polyhedra.svg",
        colour_by="val", cmap="coolwarm",
        **colour_by_style,
    )
    print(f"  wrote {OUT / 'colour_by_polyhedra.svg'}")

    # 11. Hero image: helix of corner-sharing tetrahedra coloured by position.
    #     Helix axis along x, viewed from the side so the axis is horizontal.
    n_helix = 30
    helix_edge = 2.0
    helix_turns = 4.0
    helix_total_angle = helix_turns * 2 * np.pi
    helix_dtheta = helix_total_angle / n_helix
    helix_length = 20.0  # total length along x-axis
    helix_dx = helix_length / n_helix

    # Helix radius so consecutive shared vertices are exactly edge apart.
    helix_chord_yz = np.sqrt(helix_edge**2 - helix_dx**2)
    helix_r = helix_chord_yz / (2 * np.sin(helix_dtheta / 2))

    # Shared vertices on the helix: axis along x, coil in yz-plane.
    n_helix_shared = n_helix + 1
    helix_angles = np.linspace(0, helix_total_angle, n_helix_shared)
    helix_x = np.linspace(-helix_length / 2, helix_length / 2, n_helix_shared)
    helix_shared = np.column_stack([
        helix_x,
        helix_r * np.cos(helix_angles),
        helix_r * np.sin(helix_angles),
    ])

    # Build regular tetrahedra from consecutive shared vertex pairs.
    helix_tet_verts: list[np.ndarray] = []
    helix_centroids: list[np.ndarray] = []
    for i in range(n_helix):
        va = helix_shared[i]
        vb = helix_shared[i + 1]
        mid = (va + vb) / 2
        u = (vb - va) / np.linalg.norm(vb - va)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n1 = np.cross(u, ref)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(u, n1)
        d1 = helix_edge / np.sqrt(2)
        d2 = helix_edge / 2
        vc = mid + d1 * n1 + d2 * n2
        vd = mid + d1 * n1 - d2 * n2
        helix_tet_verts.append(np.array([va, vb, vc, vd]))
        helix_centroids.append(np.array([va, vb, vc, vd]).mean(axis=0))

    helix_species: list[str] = []
    helix_coords: list[np.ndarray] = []
    for i in range(n_helix):
        helix_coords.append(helix_centroids[i])
        helix_species.append("M")
    for i in range(n_helix_shared):
        helix_coords.append(helix_shared[i])
        helix_species.append("O")
    for i in range(n_helix):
        helix_coords.append(helix_tet_verts[i][2])
        helix_species.append("O")
        helix_coords.append(helix_tet_verts[i][3])
        helix_species.append("O")

    helix_ctv = helix_edge * np.sqrt(3.0 / 8.0)
    helix_scene = StructureScene(
        species=helix_species,
        frames=[Frame(coords=np.array(helix_coords))],
        atom_styles={
            "M": AtomStyle(0.35, "grey"),
            "O": AtomStyle(0.2, (0.6, 0.6, 0.6)),
        },
        bond_specs=[BondSpec(
            species=("M", "O"), min_length=0.0,
            max_length=helix_ctv + 0.2,
            radius=0.06, colour=0.5,
        )],
        polyhedra=[PolyhedronSpec(
            centre="M", alpha=0.5,
            hide_bonds=True, hide_centre=True, hide_vertices=True,
        )],
    )

    # Colour by position along the helix.
    helix_vals = np.full(len(helix_species), np.nan)
    for i in range(n_helix):
        helix_vals[i] = float(i) / (n_helix - 1)
    helix_scene.set_atom_data("pos", helix_vals)

    # View from the side: look along y so the x-axis is horizontal.
    helix_scene.view.look_along([0, 1, 0.15])
    helix_scene.render_mpl(
        OUT / "helix.svg",
        colour_by="pos", cmap="viridis",
        show=False, figsize=(5, 3), dpi=150, half_bonds=False,
    )
    print(f"  wrote {OUT / 'helix.svg'}")

    # 12. Multi-panel projections: rutile TiO2 along [100] and [001].
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    proj_scene = rutile_scene()
    big_labels = RenderStyle(
        axes_style=AxesStyle(font_size=14.0),
    )
    for ax, direction, label in zip(
        [ax1, ax2], [[1, 0, 0], [0, 0, 1]], ["[100]", "[001]"],
    ):
        proj_scene.view.look_along(direction)
        proj_scene.title = label
        proj_scene.render_mpl(ax=ax, style=big_labels)
    fig.tight_layout(w_pad=0)
    fig.savefig(OUT / "multi_panel_projections.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'multi_panel_projections.svg'}")

    # 13. Species legend: hero image and circle sizing comparison.
    from hofmann.rendering.static import render_legend

    # Hero: full perovskite scene with legend.
    perov_legend = perovskite_plain_scene()
    perov_legend.render_mpl(
        OUT / "legend_perovskite.svg",
        figsize=(4, 4), dpi=150, show_legend=True,
    )
    print(f"  wrote {OUT / 'legend_perovskite.svg'}")

    # Legend-only images via render_legend.
    _legend_scene = perovskite_plain_scene()
    _legend_species = ("Sr", "Ti", "O")

    _legend_figsize = (0.55, 0.8)

    render_legend(
        _legend_scene, OUT / "legend_uniform.svg",
        legend_style=LegendStyle(circle_radius=5.0, species=_legend_species),
        figsize=_legend_figsize,
    )
    print(f"  wrote {OUT / 'legend_uniform.svg'}")

    render_legend(
        _legend_scene, OUT / "legend_proportional.svg",
        legend_style=LegendStyle(circle_radius=(3.0, 7.0), species=_legend_species),
        figsize=_legend_figsize,
    )
    print(f"  wrote {OUT / 'legend_proportional.svg'}")

    render_legend(
        _legend_scene, OUT / "legend_dict.svg",
        legend_style=LegendStyle(
            circle_radius={"Sr": 4.0, "Ti": 7.0, "O": 5.0},
            species=_legend_species,
        ),
        figsize=_legend_figsize,
    )
    print(f"  wrote {OUT / 'legend_dict.svg'}")

    # Custom labels with mathtext.
    render_legend(
        _legend_scene, OUT / "legend_labels.svg",
        legend_style=LegendStyle(
            species=_legend_species,
            labels={
                "Sr": "Sr2+",
                "Ti": "TiO6",
                "O":  r"$\mathrm{O^{2\!-}}$",
            },
        ),
    )
    print(f"  wrote {OUT / 'legend_labels.svg'}")

    # Polygon markers: hexagon, rotated square, and circle.
    render_legend(
        _legend_scene, OUT / "legend_polygon_markers.svg",
        legend_style=LegendStyle(
            items=(
                LegendItem(key="oct", colour="blue",
                           label="Octahedral", sides=6),
                LegendItem(key="tet", colour="red",
                           label="Tetrahedral", sides=4, rotation=45.0),
                LegendItem(key="round", colour="green",
                           label="Spherical"),
            ),
        ),
        figsize=_legend_figsize,
    )
    print(f"  wrote {OUT / 'legend_polygon_markers.svg'}")

    # Non-uniform spacing with gap_after.
    render_legend(
        _legend_scene, OUT / "legend_spacing.svg",
        legend_style=LegendStyle(
            items=(
                LegendItem(key="Sr", colour="#51b04d", label="Sr2+",
                           gap_after=10.0),
                LegendItem(key="Ti", colour="#477B9D",
                           gap_after=10.0),
                LegendItem(key="O", colour="#F03F37",
                           gap_after=0.5),
                LegendItem(key="oct", colour=(0.5, 0.7, 1.0),
                           label="TiO6", sides=6),
            ),
        ),
        figsize=_legend_figsize,
    )
    print(f"  wrote {OUT / 'legend_spacing.svg'}")


if __name__ == "__main__":
    main()
