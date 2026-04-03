"""Generate an SrTiO3 MD trajectory using ASE and CHGNet.

Builds a 4x4x4 cubic perovskite supercell and runs a short NVT
molecular dynamics simulation at 1000 K.  The trajectory is saved as an ASE
``.traj`` file for rendering with hofmann.

Requires: ase, chgnet
"""

from pathlib import Path

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from chgnet.model.dynamics import CHGNetCalculator

EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT = EXAMPLES_DIR / "srtio3_md.traj"

# Load the cubic perovskite unit cell and build a 4x4x4 supercell.
unit_cell = read(str(EXAMPLES_DIR / "srtio3.cif"))
atoms = unit_cell.repeat((4, 4, 4))
print(f"Supercell: {len(atoms)} atoms")

atoms.calc = CHGNetCalculator()
dyn = Langevin(atoms, timestep=2.0 * units.fs, temperature_K=1000, friction=0.01)

# Equilibrate before sampling.
equil_steps = 100
dyn.run(equil_steps)
print(f"Equilibrated for {equil_steps} steps")

n_steps = 200
sample_interval = 2

with Trajectory(str(OUTPUT), "w", atoms) as traj:
    traj.write(atoms)
    for step in range(1, n_steps + 1):
        dyn.run(1)
        if step % sample_interval == 0:
            traj.write(atoms)

n_frames = n_steps // sample_interval + 1
print(f"Wrote {n_frames} frames to {OUTPUT}")
