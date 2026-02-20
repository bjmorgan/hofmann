from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A single snapshot of atomic coordinates.

    Attributes:
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``.
        label: Optional frame label or identifier.

    Raises:
        ValueError: If *coords* does not have shape ``(n_atoms, 3)``.
    """

    coords: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (n_atoms, 3), got {self.coords.shape}"
            )
