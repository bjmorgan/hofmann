"""Generate a CH4 vibration trajectory using ASE and CHGNet.

Places a methane molecule in a large box and runs short NVE
molecular dynamics.  The trajectory is saved as an ASE ``.traj``
file for rendering with hofmann.

Requires: ase, chgnet
"""

from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from chgnet.model.dynamics import CHGNetCalculator

OUTPUT = Path(__file__).resolve().parent / "ch4_md.traj"

# Build a tetrahedral CH4 molecule in a large box.
d = 1.09  # C-H bond length (angstroms)
# Tetrahedral directions.
t = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
], dtype=float)
t = t / np.linalg.norm(t[0]) * d

centre = np.array([5.0, 5.0, 5.0])
positions = np.vstack([centre, centre + t])
ch4 = Atoms("CH4", positions=positions, cell=[10, 10, 10], pbc=False)

# Attach CHGNet calculator.
ch4.calc = CHGNetCalculator()

# Give a small random velocity kick to excite vibrations.
rng = np.random.default_rng(42)
ch4.set_velocities(rng.normal(0, 0.02, (5, 3)))

# Zero centre-of-mass velocity so the molecule vibrates in place.
masses = ch4.get_masses()
vel = ch4.get_velocities()
com_vel = (masses[:, None] * vel).sum(axis=0) / masses.sum()
vel -= com_vel
ch4.set_velocities(vel)

# Run NVE dynamics.
dyn = VelocityVerlet(ch4, timestep=0.5 * units.fs)

n_steps = 200
sample_interval = 2

with Trajectory(str(OUTPUT), "w", ch4) as traj:
    traj.write(ch4)
    for step in range(1, n_steps + 1):
        dyn.run(1)
        if step % sample_interval == 0:
            traj.write(ch4)

n_frames = n_steps // sample_interval + 1
print(f"Wrote {n_frames} frames to {OUTPUT}")
