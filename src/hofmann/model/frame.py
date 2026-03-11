from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single snapshot of atomic coordinates.

    Attributes:
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``.
        lattice: Unit-cell matrix, shape ``(3, 3)`` with rows as lattice
            vectors, or ``None`` for non-periodic structures.
        label: Optional frame label or identifier.

    Raises:
        ValueError: If *coords* does not have shape ``(n_atoms, 3)``.
        ValueError: If *lattice* is not ``None`` and does not have
            shape ``(3, 3)``.
    """

    coords: np.ndarray
    lattice: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (n_atoms, 3), got {self.coords.shape}"
            )
        if self.lattice is not None:
            self.lattice = np.asarray(self.lattice, dtype=float)
            if self.lattice.shape != (3, 3):
                raise ValueError(
                    f"lattice must have shape (3, 3), got {self.lattice.shape}"
                )
