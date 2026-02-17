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
            colour=(0.5, 0.7, 1.0), alpha=0.3,
            edge_colour=(0.3, 0.3, 0.3),
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

    # 2. Polyhedra vertex mode: in_front vs depth_sorted
    #    Use different angles so no face is perpendicular to the viewer.
    #    in_front: opaque polyhedra (its intended use case).
    octa_in_front = octahedron_scene(
        colour=(0.5, 0.7, 1.0), alpha=1.0,
        edge_colour=(0.15, 0.15, 0.15),
    )
    octa_in_front.render_mpl(
        OUT / "octahedron_vertex_in_front.svg",
        figsize=(3, 3), dpi=150, polyhedra_vertex_mode="in_front",
    )
    print(f"  wrote {OUT / 'octahedron_vertex_in_front.svg'}")
    #    in_front with transparent polyhedra.
    octa_in_front_trans = octahedron_scene(
        colour=(0.5, 0.7, 1.0), alpha=0.4,
        edge_colour=(0.15, 0.15, 0.15),
    )
    octa_in_front_trans.render_mpl(
        OUT / "octahedron_vertex_in_front_transparent.svg",
        figsize=(3, 3), dpi=150, polyhedra_vertex_mode="in_front",
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


if __name__ == "__main__":
    main()
