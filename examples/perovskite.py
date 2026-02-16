"""Demo: SrTiO3 perovskite unit cell rendered from a rotated viewpoint."""

from pathlib import Path

import numpy as np

from hofmann import AtomStyle, BondSpec, Frame, StructureScene, ViewState

OUTPUT = Path(__file__).resolve().parent / "perovskite.pdf"

# SrTiO3 cubic perovskite, a = 3.905 Angstroms.
a = 3.905

# Fractional positions -> Cartesian (cubic, so just multiply by a).
frac_positions = {
    "Sr": [
        (0.0, 0.0, 0.0),
    ],
    "Ti": [
        (0.5, 0.5, 0.5),
    ],
    "O": [
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ],
}

# Build a 2x2x2 supercell for a more interesting view.
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

# Styles: Sr = green, Ti = blue, O = red.
atom_styles = {
    "Sr": AtomStyle(radius=1.4, colour=(0.0, 0.8, 0.2)),
    "Ti": AtomStyle(radius=1.0, colour=(0.2, 0.4, 0.9)),
    "O":  AtomStyle(radius=0.8, colour=(0.9, 0.1, 0.1)),
}

# Bond rules: Ti-O bonds within ~2.2 Angstroms.
bond_specs = [
    BondSpec(species=("Ti", "O"), min_length=0.5, max_length=2.5,
             radius=0.12, colour=(0.4, 0.4, 0.4)),
]

# A nice oblique viewing angle (rotation around x then y).
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

scene = StructureScene(
    species=species,
    frames=[Frame(coords=coords)],
    atom_styles=atom_styles,
    bond_specs=bond_specs,
    view=view,
    title="SrTiO3 perovskite",
)

scene.render_mpl(output=OUTPUT, half_bonds=False)
print(f"Rendered {len(species)} atoms to {OUTPUT}")
