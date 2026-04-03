"""Validated container for per-atom metadata arrays."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping

import numpy as np


class AtomData(MutableMapping[str, np.ndarray]):
    """Validated mapping of named per-atom arrays.

    Every value must be a numpy array of shape ``(n_atoms,)`` (static)
    or ``(n_frames, n_atoms)`` (per-frame).  Arrays are validated on
    assignment and array-likes are converted via :func:`numpy.asarray`.

    .. note::

       Arrays are returned by reference.  In-place mutation (e.g.
       ``ad["charge"][0] = 99``) bypasses validation and does not
       invalidate the :meth:`global_range` cache.  Re-assign the key
       to trigger re-validation and cache invalidation.

    The frame count is read live from the *frames* list so that arrays
    added after appending frames are validated against the current
    trajectory length.

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
        self._range_cache: dict[str, tuple[float, float] | None] = {}
        self._labels_cache: dict[str, list[str] | None] = {}

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def __setitem__(self, key: str, value: object) -> None:
        arr = np.asarray(value)
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
        self._data[key] = arr
        self._range_cache.pop(key, None)
        self._labels_cache.pop(key, None)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._range_cache.pop(key, None)
        self._labels_cache.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def global_range(self, key: str) -> tuple[float, float] | None:
        """Return the global ``(min, max)`` for a 2-D numeric array.

        The result is cached and invalidated when the key is reassigned
        or deleted.  Returns ``None`` for 1-D arrays, categorical
        (string/object) arrays, or arrays where every value is NaN.
        """
        if key in self._range_cache:
            return self._range_cache[key]
        arr = self._data[key]
        if arr.ndim != 2 or arr.dtype.kind in ("U", "O"):
            self._range_cache[key] = None
            return None
        flat = arr.astype(float, copy=False).ravel()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            self._range_cache[key] = None
            return None
        result = (float(np.min(valid)), float(np.max(valid)))
        self._range_cache[key] = result
        return result

    def global_labels(self, key: str) -> list[str] | None:
        """Return the unique non-missing labels across all frames.

        The result is cached and invalidated when the key is reassigned
        or deleted.  Returns ``None`` for 1-D arrays or non-categorical
        (numeric) arrays.  Missing values (``None``, ``""``, ``NaN``)
        are excluded.
        """
        if key in self._labels_cache:
            return self._labels_cache[key]
        arr = self._data[key]
        if arr.ndim != 2 or arr.dtype.kind not in ("U", "O"):
            self._labels_cache[key] = None
            return None
        seen: dict[str, None] = {}
        for v in arr.ravel():
            s = str(v)
            if v is None or s == "" or s == "nan":
                continue
            if s not in seen:
                seen[s] = None
        if not seen:
            self._labels_cache[key] = None
            return None
        result = list(seen)
        self._labels_cache[key] = result
        return result

    def __repr__(self) -> str:
        keys = ", ".join(repr(k) for k in self._data)
        return f"AtomData({{{keys}}})"
