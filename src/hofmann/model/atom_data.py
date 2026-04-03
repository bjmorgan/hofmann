"""Validated container for per-atom metadata arrays."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping

import numpy as np


class AtomData(MutableMapping[str, np.ndarray]):
    """Validated mapping of named per-atom arrays.

    Every value must be a numpy array of shape ``(n_atoms,)`` (static)
    or ``(n_frames, n_atoms)`` (per-frame).  Arrays are validated on
    assignment and array-likes are converted via :func:`numpy.asarray`.

    Args:
        n_atoms: Number of atoms in the scene.
        n_frames: Number of frames in the trajectory.
    """

    def __init__(self, *, n_atoms: int, n_frames: int) -> None:
        self._n_atoms = n_atoms
        self._n_frames = n_frames
        self._data: dict[str, np.ndarray] = {}

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def __setitem__(self, key: str, value: object) -> None:
        arr = np.asarray(value)
        if arr.ndim == 1:
            if len(arr) != self._n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length "
                    f"{self._n_atoms}, got {len(arr)}"
                )
        elif arr.ndim == 2:
            if arr.shape[0] != self._n_frames:
                raise ValueError(
                    f"atom_data[{key!r}] has {arr.shape[0]} rows but "
                    f"scene has {self._n_frames} frames"
                )
            if arr.shape[1] != self._n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have {self._n_atoms} "
                    f"columns (one per atom), got {arr.shape[1]}"
                )
        else:
            raise ValueError(
                f"atom_data[{key!r}] must be 1-D or 2-D, "
                f"got {arr.ndim}-D"
            )
        self._data[key] = arr

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        keys = ", ".join(repr(k) for k in self._data)
        return f"AtomData({{{keys}}})"
