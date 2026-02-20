from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hofmann.model.atom_style import AtomStyle

#: A colour specification accepted throughout hofmann.
#:
#: Can be any of:
#:
#: - A CSS colour name or hex string (e.g. ``"red"``, ``"#ff0000"``).
#: - A single float for grey (``0.0`` = black, ``1.0`` = white).
#: - An RGB tuple or list with values in ``[0, 1]``
#:   (e.g. ``(1.0, 0.0, 0.0)``).
#:
#: See :func:`normalise_colour` for conversion to a normalised RGB tuple.
Colour = str | float | tuple[float, float, float] | list[float]

#: A colourmap specification for atom-data colouring.
#:
#: Can be any of:
#:
#: - A matplotlib colourmap name (e.g. ``"viridis"``).
#: - A callable mapping a float in ``[0, 1]`` to an RGB or RGBA sequence.
#: - A matplotlib :class:`~matplotlib.colors.Colormap` object (which is
#:   callable and returns RGBA).
#:
#: Callables returning RGBA are automatically truncated to RGB by
#: :func:`_resolve_cmap`.
CmapSpec = str | Callable[[float], Sequence[float]]


def normalise_colour(colour: Colour) -> tuple[float, float, float]:
    """Convert a colour specification to a normalised (r, g, b) tuple.

    Accepts CSS colour names (e.g. ``"red"``), hex strings
    (e.g. ``"#FF0000"``), grey floats (e.g. ``0.7``), or RGB tuples
    (e.g. ``(1.0, 0.3, 0.3)``).

    Args:
        colour: The colour to normalise.

    Returns:
        A tuple of three floats in [0, 1].

    Raises:
        ValueError: If the colour cannot be interpreted.
    """
    if isinstance(colour, (int, float)) and not isinstance(colour, bool):
        f = float(colour)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"Grey value must be in [0, 1], got {f}")
        return (f, f, f)

    if isinstance(colour, (tuple, list)):
        if len(colour) != 3:
            raise ValueError(
                f"RGB sequence must have 3 elements, got {len(colour)}"
            )
        r, g, b = (float(c) for c in colour)
        for name, val in [("r", r), ("g", g), ("b", b)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"RGB component {name} must be in [0, 1], got {val}"
                )
        return (r, g, b)

    if isinstance(colour, str):
        from matplotlib.colors import to_rgb

        try:
            return to_rgb(colour)
        except ValueError:
            raise ValueError(f"Unrecognised colour name: {colour!r}")

    raise ValueError(f"Cannot interpret colour: {colour!r}")


def _species_colours(
    species: list[str],
    atom_styles: dict[str, AtomStyle],
) -> list[tuple[float, float, float]]:
    """Return per-atom colours from species styles (the default path).

    Colours are normalised once per species and cached, so the cost is
    proportional to the number of unique species rather than the number
    of atoms.
    """
    cache: dict[str, tuple[float, float, float]] = {}
    colours: list[tuple[float, float, float]] = []
    for sp in species:
        if sp not in cache:
            style = atom_styles.get(sp)
            if style is not None:
                cache[sp] = normalise_colour(style.colour)
            else:
                cache[sp] = (0.5, 0.5, 0.5)
        colours.append(cache[sp])
    return colours


def _resolve_cmap(
    cmap: CmapSpec,
) -> Callable[[float], tuple[float, float, float]]:
    """Turn a colourmap specification into a callable float -> RGB.

    Accepts a colourmap name (string), a callable mapping a float in
    ``[0, 1]`` to a colour tuple, or a matplotlib ``Colormap`` object.
    The returned wrapper always produces 3-tuple ``(r, g, b)`` even if
    the underlying callable returns RGBA.

    Raises:
        TypeError: If *cmap* is not a string and not callable.
    """
    if isinstance(cmap, str):
        import matplotlib
        fn: Callable[..., Sequence[float]] = matplotlib.colormaps[cmap]
    elif callable(cmap):
        fn = cmap
    else:
        raise TypeError(f"Unsupported cmap type: {type(cmap)}")

    def _wrap(val: float) -> tuple[float, float, float]:
        result = fn(val)
        return (result[0], result[1], result[2])
    return _wrap


def _resolve_single_layer(
    atom_data: dict[str, np.ndarray],
    key: str,
    fallback: list[tuple[float, float, float]],
    cmap: CmapSpec,
    colour_range: tuple[float, float] | None,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Resolve colours for a single colour_by key.

    Returns:
        A tuple of ``(colours, missing_mask)`` where *colours* is a
        per-atom list of ``(r, g, b)`` tuples and *missing_mask* is a
        boolean array that is ``True`` for atoms with missing data
        (which received their species fallback colour).
    """
    values = atom_data[key]
    cmap_fn = _resolve_cmap(cmap)
    if values.dtype.kind in ("U", "O"):
        return _resolve_categorical(values, fallback, cmap_fn)
    return _resolve_numerical(values, fallback, cmap_fn, colour_range)


def resolve_atom_colours(
    species: list[str],
    atom_styles: dict[str, AtomStyle],
    atom_data: dict[str, np.ndarray],
    colour_by: str | list[str] | None = None,
    cmap: CmapSpec | list[CmapSpec] = "viridis",
    colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
) -> list[tuple[float, float, float]]:
    """Resolve per-atom RGB colours, optionally using a colourmap.

    When *colour_by* is ``None`` (the default) the usual species-based
    colours from *atom_styles* are returned.  When it is a single
    string, the named array from *atom_data* is mapped through *cmap*.

    When *colour_by* is a **list** of keys, each layer is tried in
    order and the first non-missing value (non-NaN for numerical,
    non-empty for categorical) determines the atom's colour.  This
    allows different colouring rules for different atom subsets::

        scene.set_atom_data("metal_type", {0: "Fe", 2: "Co"})
        scene.set_atom_data("o_coord", {1: 4, 3: 6})
        scene.render_mpl(
            colour_by=["metal_type", "o_coord"],
            cmap=["Set1", "Blues"],
        )

    Args:
        species: Per-atom species labels.
        atom_styles: Species-to-style mapping.
        atom_data: Per-atom metadata arrays from the scene.
        colour_by: Key (or list of keys) into *atom_data* to colour
            by, or ``None`` for species-based colouring.  When a
            list, layers are tried in priority order.
        cmap: A matplotlib colourmap name (e.g. ``"viridis"``), a
            matplotlib ``Colormap`` object, or a callable mapping a
            float in ``[0, 1]`` to an ``(r, g, b)`` tuple.  When
            *colour_by* is a list, *cmap* may also be a list of the
            same length (one per layer).  A single value is broadcast
            to all layers.
        colour_range: Explicit ``(vmin, vmax)`` for normalising
            numerical data.  ``None`` auto-ranges from the data.
            Ignored for categorical data.  When *colour_by* is a
            list, may also be a list of the same length.

    Returns:
        List of ``(r, g, b)`` tuples, one per atom.

    Raises:
        KeyError: If *colour_by* (or any key in the list) is not
            found in *atom_data*.
        ValueError: If *colour_by* is a list and *cmap* or
            *colour_range* is also a list of a different length, or
            if *colour_by* is a single string and *cmap* or
            *colour_range* is a list.
    """
    if colour_by is None:
        return _species_colours(species, atom_styles)

    fallback = _species_colours(species, atom_styles)

    # --- Single key (common case) ---
    if isinstance(colour_by, str):
        if isinstance(cmap, list):
            raise ValueError(
                "cmap must not be a list when colour_by is a single string"
            )
        if isinstance(colour_range, list):
            raise ValueError(
                "colour_range must not be a list when colour_by is a "
                "single string"
            )
        colours, _mask = _resolve_single_layer(
            atom_data, colour_by, fallback, cmap, colour_range,
        )
        return colours

    # --- List of keys (priority merge) ---
    n_layers = len(colour_by)

    # Broadcast cmap / colour_range to lists.
    if not isinstance(cmap, list):
        cmaps = [cmap] * n_layers
    else:
        cmaps = cmap

    if not isinstance(colour_range, list):
        ranges: list[tuple[float, float] | None] = [colour_range] * n_layers
    else:
        ranges = colour_range

    if len(cmaps) != n_layers:
        raise ValueError(
            f"colour_by has {n_layers} keys but cmap has "
            f"{len(cmaps)} entries"
        )
    if len(ranges) != n_layers:
        raise ValueError(
            f"colour_by has {n_layers} keys but colour_range has "
            f"{len(ranges)} entries"
        )

    # Resolve each layer independently.
    layers = [
        _resolve_single_layer(atom_data, key, fallback, cm, cr)
        for key, cm, cr in zip(colour_by, cmaps, ranges)
    ]

    # Merge: first layer with non-missing data wins.
    n_atoms = len(species)
    result: list[tuple[float, float, float]] = list(fallback)
    for i in range(n_atoms):
        for layer_colours, layer_mask in layers:
            if not layer_mask[i]:
                result[i] = layer_colours[i]
                break

    return result


def _resolve_numerical(
    values: np.ndarray,
    fallback: list[tuple[float, float, float]],
    cmap_fn: Callable[[float], tuple[float, float, float]],
    colour_range: tuple[float, float] | None,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Map numerical values through a colourmap.

    Integer arrays are automatically coerced to float so that NaN
    sentinels (used for missing data) are representable.

    Returns:
        A tuple of ``(colours, missing_mask)`` where *missing_mask* is
        a boolean array that is ``True`` for atoms whose values are
        NaN.
    """
    values = values.astype(float, copy=False)
    mask = np.isnan(values)

    if colour_range is not None:
        vmin, vmax = colour_range
    else:
        valid = values[~mask]
        if len(valid) == 0:
            return list(fallback), mask
        vmin, vmax = float(np.min(valid)), float(np.max(valid))

    if vmin == vmax:
        normalised = np.where(mask, np.nan, 0.5)
    else:
        normalised = (values - vmin) / (vmax - vmin)
        normalised = np.clip(normalised, 0.0, 1.0)

    colours: list[tuple[float, float, float]] = []
    for i, val in enumerate(normalised):
        if mask[i]:
            colours.append(fallback[i])
        else:
            colours.append(cmap_fn(float(val)))
    return colours, mask


def _is_categorical_missing(v: object) -> bool:
    """Return True if *v* should be treated as a missing categorical value.

    Missing values are ``None``, empty strings, and float ``NaN``
    (including numpy floating scalars such as ``np.float64('nan')``).
    """
    if v is None:
        return True
    if isinstance(v, str) and v == "":
        return True
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return True
    return False


def _resolve_categorical(
    values: np.ndarray,
    fallback: list[tuple[float, float, float]],
    cmap_fn: Callable[[float], tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Map categorical labels through a colourmap.

    Values of ``None``, ``NaN``, and empty strings are treated as
    missing and receive their species fallback colour.

    Returns:
        A tuple of ``(colours, missing_mask)`` where *missing_mask* is
        a boolean array that is ``True`` for atoms whose values are
        missing (``None``, ``NaN``, or empty string).
    """
    # Build missing mask and unique labels in a single pass.
    missing = np.empty(len(values), dtype=bool)
    seen: dict[str, int] = {}
    for i, v in enumerate(values):
        if _is_categorical_missing(v):
            missing[i] = True
        else:
            missing[i] = False
            s = str(v)
            if s not in seen:
                seen[s] = len(seen)

    n_labels = len(seen)
    if n_labels == 0:
        return list(fallback), missing

    # Space labels evenly across [0, 1].
    if n_labels == 1:
        positions = {label: 0.5 for label in seen}
    else:
        positions = {
            label: idx / (n_labels - 1) for label, idx in seen.items()
        }

    colours: list[tuple[float, float, float]] = []
    for i, v in enumerate(values):
        if missing[i]:
            colours.append(fallback[i])
        else:
            colours.append(cmap_fn(positions[str(v)]))
    return colours, missing
