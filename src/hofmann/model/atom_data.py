"""Validated container for per-atom metadata arrays."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from types import MappingProxyType

import numpy as np

from hofmann.model.colour import _is_categorical_missing


def _compute_global_range(
    arr: np.ndarray,
) -> tuple[float, float] | None:
    """Compute the global ``(min, max)`` for a 2-D numeric array.

    Returns ``None`` for 1-D arrays, categorical (string/object)
    dtypes, empty arrays, or arrays where every value is NaN.
    """
    if arr.ndim != 2 or arr.dtype.kind in ("U", "O"):
        return None
    if arr.size == 0:
        return None
    with np.errstate(all="ignore"):
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if np.isnan(lo):
        return None
    return (lo, hi)


def _compute_global_labels(arr: np.ndarray) -> tuple[str, ...] | None:
    """Return the unique non-missing labels in a 2-D categorical array.

    Returns ``None`` for 1-D arrays or non-categorical (numeric)
    dtypes.  Missing values (``None``, ``""``, NaN) are excluded.

    Labels are returned in first-encountered order across
    ``arr.ravel()``.  This ordering is load-bearing: downstream
    colourmap assignment (``_resolve_categorical``) indexes into this
    tuple via :func:`enumerate`, so the order determines which label
    gets which colour.  Preserve the insertion-order semantics;
    sorting or otherwise reordering the result will silently change
    per-atom colours.
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
    return tuple(seen)


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
       re-validates the shape and recomputes the :attr:`ranges` and
       :attr:`labels` entries.  Only the array buffer is frozen — for
       ``object``-dtype arrays, any mutable objects stored inside
       remain mutable.

    The frame count is read live from the *frames* list so that arrays
    added after appending frames are validated against the current
    trajectory length.  However, existing 2-D arrays are not
    re-validated when frames are appended — re-assign them after
    changing the trajectory length.

    Attributes:
        ranges: Read-only mapping of keys to ``(min, max)`` tuples
            for 2-D numeric arrays, or ``None`` for keys that do not
            have a meaningful numeric range (1-D arrays, categorical
            arrays, empty arrays, all-NaN numeric arrays).  Entries
            are added on assignment, replaced on reassignment, and
            removed on deletion.
        labels: Read-only mapping of keys to tuples of unique
            non-missing categorical labels across all frames, or
            ``None`` for keys without a meaningful label set (1-D
            arrays, numeric dtypes, categorical arrays with no
            non-missing values).  Missing values (``None``, ``""``,
            NaN) are excluded from the label set.  Entries are added
            on assignment, replaced on reassignment, and removed on
            deletion.

    For 2-D arrays, exactly one of ``ranges[key]`` and
    ``labels[key]`` holds a value, depending on dtype: numeric
    arrays populate ``ranges``, categorical arrays populate
    ``labels``.  For 1-D arrays, both are ``None``.

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
        self._labels: dict[str, tuple[str, ...] | None] = {}
        self._ranges_view: Mapping[str, tuple[float, float] | None] = (
            MappingProxyType(self._ranges)
        )
        self._labels_view: Mapping[str, tuple[str, ...] | None] = (
            MappingProxyType(self._labels)
        )

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def ranges(self) -> Mapping[str, tuple[float, float] | None]:
        return self._ranges_view

    @property
    def labels(self) -> Mapping[str, tuple[str, ...] | None]:
        return self._labels_view

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
        if arr.dtype.kind not in ("b", "i", "u", "f", "U", "O"):
            raise ValueError(
                f"atom_data[{key!r}] has unsupported dtype "
                f"{arr.dtype}; supported dtypes are bool, integer, "
                f"float, string, and object"
            )
        arr.flags.writeable = False
        new_range = _compute_global_range(arr)
        new_labels = _compute_global_labels(arr)
        self._data[key] = arr
        self._ranges[key] = new_range
        self._labels[key] = new_labels

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

    def __repr__(self) -> str:
        keys = ", ".join(repr(k) for k in self._data)
        return f"AtomData({{{keys}}})"
