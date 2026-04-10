"""Validated container for per-atom metadata arrays."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping

import numpy as np

from hofmann.model.colour import _is_categorical_missing


def _compute_global_range(
    arr: np.ndarray,
) -> tuple[float, float] | None:
    """Compute the global ``(min, max)`` for a 2-D numeric array.

    Returns ``None`` for 1-D arrays, categorical (string/object)
    dtypes, or arrays where every value is NaN.
    """
    if arr.ndim != 2 or arr.dtype.kind in ("U", "O"):
        return None
    with np.errstate(all="ignore"):
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if np.isnan(lo):
        return None
    return (lo, hi)


def _compute_global_labels(arr: np.ndarray) -> list[str] | None:
    """Return the unique non-missing labels across all frames.

    Returns ``None`` for 1-D arrays or non-categorical (numeric)
    dtypes.  Missing values (``None``, ``""``, NaN) are excluded.
    """
    if arr.ndim != 2 or arr.dtype.kind not in ("U", "O"):
        return None
    seen: dict[str, None] = {}
    for v in arr.ravel():
        if _is_categorical_missing(v):
            continue
        s = str(v)
        if s not in seen:
            seen[s] = None
    if not seen:
        return None
    return list(seen)


class AtomData(MutableMapping[str, np.ndarray]):
    """Validated mapping of named per-atom arrays.

    Every value must be a numpy array of shape ``(n_atoms,)`` (static)
    or ``(n_frames, n_atoms)`` (per-frame).  Assigned values are always
    copied via :func:`numpy.array` — including existing numpy arrays —
    so the container owns the buffer and the caller's source array is
    left untouched.

    .. note::

       Stored arrays are returned read-only.  In-place mutation (e.g.
       ``ad["charge"][0] = 99``) raises
       ``ValueError: assignment destination is read-only``.  To update
       values, build a new array and reassign the key; reassignment
       re-validates the shape and invalidates the
       :meth:`global_range` and :meth:`global_labels` caches.  Only
       the array buffer is frozen — for ``object``-dtype arrays, any
       mutable objects stored inside remain mutable.

    The frame count is read live from the *frames* list so that arrays
    added after appending frames are validated against the current
    trajectory length.  However, existing 2-D arrays are not
    re-validated when frames are appended — re-assign them after
    changing the trajectory length.

    Args:
        n_atoms: Number of atoms in the scene.
        frames: The scene's live frame list.  The length of this list
            is used for 2-D array validation.
    """

    def __init__(self, *, n_atoms: int, frames: list) -> None:
        if n_atoms < 0:
            raise ValueError(f"n_atoms must be non-negative, got {n_atoms}")
        self._n_atoms = n_atoms
        self._frames = frames
        self._data: dict[str, np.ndarray] = {}
        self._ranges: dict[str, tuple[float, float] | None] = {}
        self._labels: dict[str, list[str] | None] = {}

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def __setitem__(self, key: str, value: object) -> None:
        arr = np.array(value)
        if arr.ndim == 1:
            if len(arr) != self._n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length "
                    f"{self._n_atoms}, got {len(arr)}"
                )
        elif arr.ndim == 2:
            if arr.shape[0] != self.n_frames:
                raise ValueError(
                    f"atom_data[{key!r}] has {arr.shape[0]} rows but "
                    f"scene has {self.n_frames} frames"
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
        arr.flags.writeable = False
        self._data[key] = arr
        self._ranges[key] = _compute_global_range(arr)
        self._labels[key] = _compute_global_labels(arr)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        del self._ranges[key]
        del self._labels[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def global_range(self, key: str) -> tuple[float, float] | None:
        """Return the global ``(min, max)`` for a 2-D numeric array.

        Returns ``None`` for 1-D arrays, categorical (string/object)
        arrays, or arrays where every value is NaN.
        """
        return self._ranges[key]

    def global_labels(self, key: str) -> list[str] | None:
        """Return the unique non-missing labels across all frames.

        Returns ``None`` for 1-D arrays or non-categorical (numeric)
        arrays.  Missing values (``None``, ``""``, ``NaN``) are
        excluded.
        """
        return self._labels[key]

    def __repr__(self) -> str:
        keys = ", ".join(repr(k) for k in self._data)
        return f"AtomData({{{keys}}})"
