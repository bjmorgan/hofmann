"""Generate a high-temperature SrTiO3 MD trajectory for tilt colouring.

Builds a 4x4x4 cubic perovskite supercell and runs a short NVT
molecular dynamics simulation at 1500 K.  The trajectory is filtered
to a single octahedral layer and saved for rendering by the docs
build.

Requires: ase, chgnet
"""

from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from chgnet.model.dynamics import CHGNetCalculator

EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT = EXAMPLES_DIR / "srtio3_hot.traj"

# Load the cubic perovskite unit cell and build a 4x4x4 supercell.
unit_cell = read(str(EXAMPLES_DIR / "srtio3.cif"))
atoms = unit_cell.repeat((4, 4, 4))
print(f"Supercell: {len(atoms)} atoms")

atoms.calc = CHGNetCalculator()

# High temperature for large tilt fluctuations.
dyn = Langevin(atoms, timestep=2.0 * units.fs, temperature_K=1500, friction=0.01)

equil_steps = 200
dyn.run(equil_steps)
print(f"Equilibrated for {equil_steps} steps at 1500 K")

n_steps = 400
sample_interval = 4

with Trajectory(str(OUTPUT), "w", atoms) as traj:
    traj.write(atoms)
    for step in range(1, n_steps + 1):
        dyn.run(1)
        if step % sample_interval == 0:
            traj.write(atoms)
            if step % 40 == 0:
                print(f"  step {step}/{n_steps}")

full_traj = read(str(OUTPUT), index=":")
n_frames = len(full_traj)
print(f"Wrote {n_frames} frames to {OUTPUT}")

# Filter to one octahedral layer.
ref = full_traj[0]
z = ref.positions[:, 2]
symbols = np.array(ref.get_chemical_symbols())

sr_mask = (symbols == "Sr") & (np.abs(z - 3.9) < 0.5)
ti_mask = (symbols == "Ti") & (np.abs(z - 5.9) < 0.5)
o_mask = (symbols == "O") & (
    (np.abs(z - 3.9) < 0.5)
    | (np.abs(z - 5.9) < 0.5)
    | (np.abs(z - 7.8) < 0.5)
)
indices = np.where(sr_mask | ti_mask | o_mask)[0]
print(f"Selected {len(indices)} atoms for layer")

layer_traj = [frame[indices] for frame in full_traj]
filtered_path = EXAMPLES_DIR / "srtio3_hot_filtered.traj"
write(str(filtered_path), layer_traj)
print(f"Filtered trajectory saved to {filtered_path}")
