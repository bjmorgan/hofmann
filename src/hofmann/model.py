"""Core data model for hofmann: dataclasses, colour handling, and projection."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pymatgen.core import Structure

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
    atom_styles: dict[str, "AtomStyle"],
) -> list[tuple[float, float, float]]:
    """Return per-atom colours from species styles (the default path)."""
    colours: list[tuple[float, float, float]] = []
    for sp in species:
        style = atom_styles.get(sp)
        if style is not None:
            colours.append(normalise_colour(style.colour))
        else:
            colours.append((0.5, 0.5, 0.5))
    return colours


def _resolve_cmap(
    cmap: "str | object",
) -> Callable[[float], tuple[float, float, float]]:
    """Turn a colourmap specification into a callable float -> RGB."""
    if callable(cmap) and not isinstance(cmap, str):
        return cmap  # type: ignore[return-value]
    if isinstance(cmap, str):
        import matplotlib
        mpl_cmap = matplotlib.colormaps[cmap]
        def _wrap(val: float) -> tuple[float, float, float]:
            rgba = mpl_cmap(val)
            return (rgba[0], rgba[1], rgba[2])
        return _wrap
    # Assume a matplotlib Colormap-like object.
    def _wrap_obj(val: float) -> tuple[float, float, float]:
        rgba = cmap(val)  # type: ignore[operator]
        return (rgba[0], rgba[1], rgba[2])
    return _wrap_obj


def _resolve_single_layer(
    atom_data: dict[str, np.ndarray],
    key: str,
    fallback: list[tuple[float, float, float]],
    cmap: "str | Callable[[float], tuple[float, float, float]] | object",
    colour_range: tuple[float, float] | None,
) -> list[tuple[float, float, float]]:
    """Resolve colours for a single colour_by key."""
    values = atom_data[key]
    cmap_fn = _resolve_cmap(cmap)
    if values.dtype.kind in ("U", "O"):
        return _resolve_categorical(values, fallback, cmap_fn)
    return _resolve_numerical(values, fallback, cmap_fn, colour_range)


def resolve_atom_colours(
    species: list[str],
    atom_styles: dict[str, "AtomStyle"],
    atom_data: dict[str, np.ndarray],
    colour_by: str | list[str] | None = None,
    cmap: str | Callable[[float], tuple[float, float, float]] | object | list = "viridis",
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
    """
    if colour_by is None:
        return _species_colours(species, atom_styles)

    fallback = _species_colours(species, atom_styles)

    # --- Single key (common case) ---
    if isinstance(colour_by, str):
        cr = colour_range if not isinstance(colour_range, list) else None
        return _resolve_single_layer(
            atom_data, colour_by, fallback, cmap, cr,
        )

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

    # Resolve each layer independently.
    layers = [
        _resolve_single_layer(atom_data, key, fallback, cm, cr)
        for key, cm, cr in zip(colour_by, cmaps, ranges)
    ]

    # Merge: first layer with a non-fallback colour wins.
    n_atoms = len(species)
    result: list[tuple[float, float, float]] = list(fallback)
    for i in range(n_atoms):
        for layer in layers:
            if layer[i] != fallback[i]:
                result[i] = layer[i]
                break

    return result


def _resolve_numerical(
    values: np.ndarray,
    fallback: list[tuple[float, float, float]],
    cmap_fn: Callable[[float], tuple[float, float, float]],
    colour_range: tuple[float, float] | None,
) -> list[tuple[float, float, float]]:
    """Map numerical values through a colourmap."""
    mask = np.isnan(values)

    if colour_range is not None:
        vmin, vmax = colour_range
    else:
        valid = values[~mask]
        if len(valid) == 0:
            return list(fallback)
        vmin, vmax = float(np.min(valid)), float(np.max(valid))

    if vmin == vmax:
        normalised = np.where(mask, np.nan, 0.5)
    else:
        normalised = (values - vmin) / (vmax - vmin)
        normalised = np.clip(normalised, 0.0, 1.0)

    colours: list[tuple[float, float, float]] = []
    for i, val in enumerate(normalised):
        if np.isnan(val):
            colours.append(fallback[i])
        else:
            colours.append(cmap_fn(float(val)))
    return colours


def _resolve_categorical(
    values: np.ndarray,
    fallback: list[tuple[float, float, float]],
    cmap_fn: Callable[[float], tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Map categorical labels through a colourmap."""
    # Find unique non-empty labels, preserving first-occurrence order.
    seen: dict[str, int] = {}
    for v in values:
        s = str(v)
        if s and s not in seen:
            seen[s] = len(seen)

    n_labels = len(seen)
    if n_labels == 0:
        return list(fallback)

    # Space labels evenly across [0, 1].
    if n_labels == 1:
        positions = {label: 0.5 for label in seen}
    else:
        positions = {
            label: idx / (n_labels - 1) for label, idx in seen.items()
        }

    colours: list[tuple[float, float, float]] = []
    for i, v in enumerate(values):
        s = str(v)
        if s and s in positions:
            colours.append(cmap_fn(positions[s]))
        else:
            colours.append(fallback[i])
    return colours


class SlabClipMode(StrEnum):
    """How slab clipping interacts with coordination polyhedra.

    Controls whether polyhedra at the slab boundary are drawn partially,
    dropped entirely, or forced to be complete.

    Attributes:
        PER_FACE: Drop individual faces whose vertices lie outside the
            slab.  May produce partial polyhedron fragments.
        CLIP_WHOLE: If any vertex of a polyhedron is outside the slab,
            skip the entire polyhedron and its centre-to-vertex bonds.
        INCLUDE_WHOLE: If the centre atom is inside the slab, force
            all vertices and bonds of the polyhedron to be visible
            regardless of slab depth.
    """

    PER_FACE = "per_face"
    CLIP_WHOLE = "clip_whole"
    INCLUDE_WHOLE = "include_whole"


class WidgetCorner(StrEnum):
    """Which corner of the viewport to place the axes widget.

    Attributes:
        BOTTOM_LEFT: Bottom-left corner (default).
        BOTTOM_RIGHT: Bottom-right corner.
        TOP_LEFT: Top-left corner.
        TOP_RIGHT: Top-right corner.
    """

    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"


class PolyhedraVertexMode(StrEnum):
    """How polyhedron vertex atoms are ordered relative to faces.

    Attributes:
        IN_FRONT: Draw each vertex atom in front of all faces it
            belongs to.  Each vertex is deferred until after its
            last connected face has been painted.  Non-vertex atoms
            (e.g. the centre) draw at their natural depth.  Best
            for opaque polyhedra.
        DEPTH_SORTED: Draw front vertices (closer to the viewer than
            the centroid) on top of faces, and back vertices in their
            natural depth position behind front-facing faces.  Correct
            for transparent polyhedra but may produce minor painter's-
            algorithm artefacts at silhouette edges.
    """

    IN_FRONT = "in_front"
    DEPTH_SORTED = "depth_sorted"


_VALID_LINESTYLES = frozenset({"solid", "dashed", "dotted", "dashdot"})


@dataclass(frozen=True)
class CellEdgeStyle:
    """Visual style for unit cell edges.

    Attributes:
        colour: Edge colour.  Accepts any format understood by
            :func:`normalise_colour`.
        line_width: Width of the edge line in display units.
        linestyle: Line pattern: ``"solid"``, ``"dashed"``,
            ``"dotted"``, or ``"dashdot"``.
    """

    colour: Colour = (0.3, 0.3, 0.3)
    line_width: float = 0.8
    linestyle: str = "solid"

    def __post_init__(self) -> None:
        if self.line_width < 0:
            raise ValueError(
                f"line_width must be non-negative, got {self.line_width}"
            )
        if self.linestyle not in _VALID_LINESTYLES:
            raise ValueError(
                f"linestyle must be one of {sorted(_VALID_LINESTYLES)}, "
                f"got {self.linestyle!r}"
            )


@dataclass(frozen=True)
class AxesStyle:
    """Visual style for the crystallographic axes orientation widget.

    The widget draws three axis lines (a, b, c lattice directions)
    from a common origin in a corner of the viewport.  Lines rotate
    in sync with the structure, with italic labels at the tips.

    Attributes:
        colours: Tuple of three colours for the (a, b, c) axes.
            Each element accepts any format understood by
            :func:`normalise_colour`.  Defaults to uniform dark grey.
            Pass distinct colours for per-axis colouring.
        labels: Tuple of three label strings for the axes.
        font_size: Font size for axis labels in points.
        italic: Whether to render labels in italic (crystallographic
            convention).
        arrow_length: Axis line length as a fraction of the viewport
            half-extent.
        line_width: Width of the axis lines in points.
        corner: Widget origin position.  Pass a :class:`WidgetCorner`
            (or its string value) for automatic placement in one of
            the four viewport corners, offset by *margin*.  Pass an
            ``(x, y)`` tuple of fractional viewport coordinates
            (0.0 = left/bottom, 1.0 = right/top) for an explicit
            position; *margin* is ignored in this case.
        margin: Offset from the corner as a fraction of the viewport
            half-extent.  Only used when *corner* is a
            :class:`WidgetCorner`.
    """

    colours: tuple[Colour, Colour, Colour] = (
        (0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3),
        (0.3, 0.3, 0.3),
    )
    labels: tuple[str, str, str] = ("a", "b", "c")
    font_size: float = 10.0
    italic: bool = True
    arrow_length: float = 0.08
    line_width: float = 1.0
    corner: WidgetCorner | tuple[float, float] = WidgetCorner.BOTTOM_LEFT
    margin: float = 0.15

    def __post_init__(self) -> None:
        if isinstance(self.corner, tuple):
            if len(self.corner) != 2:
                raise ValueError(
                    f"corner tuple must have 2 elements, got {len(self.corner)}"
                )
        elif isinstance(self.corner, str):
            object.__setattr__(self, "corner", WidgetCorner(self.corner))
        if self.font_size <= 0:
            raise ValueError(
                f"font_size must be positive, got {self.font_size}"
            )
        if self.arrow_length <= 0:
            raise ValueError(
                f"arrow_length must be positive, got {self.arrow_length}"
            )
        if self.line_width < 0:
            raise ValueError(
                f"line_width must be non-negative, got {self.line_width}"
            )
        if self.margin < 0:
            raise ValueError(
                f"margin must be non-negative, got {self.margin}"
            )
        if len(self.colours) != 3:
            raise ValueError(
                f"colours must have exactly 3 elements, got {len(self.colours)}"
            )
        if len(self.labels) != 3:
            raise ValueError(
                f"labels must have exactly 3 elements, got {len(self.labels)}"
            )


@dataclass
class RenderStyle:
    """Visual style settings for rendering.

    Groups all appearance parameters that control how a scene is drawn,
    independent of the scene data itself.  A default ``RenderStyle()``
    gives the standard ball-and-stick look.

    Pass a style to :func:`~hofmann.render_mpl.render_mpl` via the
    *style* keyword, or override individual fields with convenience
    kwargs::

        style = RenderStyle(show_outlines=False, atom_scale=0.8)
        scene.render_mpl("out.svg", style=style)

        # Or override a single field:
        scene.render_mpl("out.svg", show_bonds=False)

    Attributes:
        atom_scale: Scale factor for atom display radii.  ``0.5`` gives
            ball-and-stick; ``1.0`` gives space-filling.
        bond_scale: Scale factor for bond cylinder radii.
        bond_colour: Override colour for all bonds, or ``None`` to use
            per-spec / half-bond colouring.
        half_bonds: Split each bond at the midpoint and colour halves
            to match the nearest atom.
        show_bonds: Whether to draw bonds at all.
        show_polyhedra: Whether to draw coordination polyhedra.
        show_outlines: Whether to draw outlines around atoms and bonds.
        outline_colour: Colour for outlines when *show_outlines* is
            ``True``.
        atom_outline_width: Line width for atom outlines (points).
        bond_outline_width: Line width for bond outlines (points).
        slab_clip_mode: How slab clipping affects polyhedra at the
            boundary.  ``"per_face"`` drops individual faces with
            out-of-slab vertices (default), ``"clip_whole"`` hides
            the entire polyhedron if any vertex is clipped, and
            ``"include_whole"`` forces the complete polyhedron to be
            visible when its centre atom is in the slab.
        circle_segments: Number of line segments used to approximate
            atom circles in static output.  Higher values give
            smoother circles in vector output (PDF/SVG).  The default
            (``72``) gives publication-quality output.
        arc_segments: Number of line segments per semicircular bond
            end-cap in static output.  Higher values give smoother
            bond ends in vector output.  The default (``12``) gives
            publication-quality output.
        interactive_circle_segments: Number of line segments for atom
            circles in the interactive viewer.  Lower values give
            faster redraws.  The default (``24``) balances quality
            and responsiveness.
        interactive_arc_segments: Number of line segments per bond
            end-cap in the interactive viewer.  Lower values give
            faster redraws.  The default (``5``) balances quality
            and responsiveness.
        polyhedra_shading: Strength of diffuse shading on polyhedra
            faces.  ``0.0`` gives flat colouring (no shading);
            ``1.0`` (the default) gives full Lambertian-style shading
            where faces pointing at the viewer are bright and edge-on
            faces are dimmed.
        polyhedra_vertex_mode: How vertex atoms are ordered relative
            to polyhedral faces.  ``"in_front"`` (the default) draws
            each vertex on top of the faces it belongs to.
            ``"depth_sorted"`` draws front vertices on top but back
            vertices behind front-facing faces — an alternative for
            transparent polyhedra.
        polyhedra_outline_width: Global override for polyhedra outline
            line width (points).  When ``None`` (the default), each
            polyhedron uses its own ``PolyhedronSpec.edge_width``.
            When set, overrides all per-spec values.
        show_cell: Whether to draw unit cell edges.  ``None``
            (the default) auto-detects: edges are drawn when the
            scene has a lattice.  ``True`` forces drawing (raises
            :class:`ValueError` at render time if no lattice is
            available).  ``False`` suppresses drawing.
        cell_style: Visual style for unit cell edges.  See
            :class:`CellEdgeStyle`.
        show_axes: Whether to draw the crystallographic axes
            orientation widget.  ``None`` (the default) auto-detects:
            the widget is drawn when the scene has a lattice.
            ``True`` forces drawing (raises :class:`ValueError` at
            render time if no lattice is available).  ``False``
            suppresses drawing.
        axes_style: Visual style for the axes widget.  See
            :class:`AxesStyle`.

    Raises:
        ValueError: If *atom_scale* or *bond_scale* are not positive,
            *atom_outline_width* or *bond_outline_width* are negative,
            *circle_segments* or *interactive_circle_segments* < 3,
            *arc_segments* or *interactive_arc_segments* < 2,
            *polyhedra_shading* is outside ``[0, 1]``, or
            *polyhedra_outline_width* is negative.
    """

    atom_scale: float = 0.5
    bond_scale: float = 1.0
    bond_colour: Colour | None = None
    half_bonds: bool = True
    show_bonds: bool = True
    show_polyhedra: bool = True
    show_outlines: bool = True
    outline_colour: Colour = (0.15, 0.15, 0.15)
    atom_outline_width: float = 1.0
    bond_outline_width: float = 1.0
    slab_clip_mode: SlabClipMode = SlabClipMode.PER_FACE
    circle_segments: int = 72
    arc_segments: int = 12
    interactive_circle_segments: int = 24
    interactive_arc_segments: int = 5
    polyhedra_shading: float = 1.0
    polyhedra_vertex_mode: PolyhedraVertexMode = PolyhedraVertexMode.IN_FRONT
    polyhedra_outline_width: float | None = None
    show_cell: bool | None = None
    cell_style: CellEdgeStyle = field(default_factory=CellEdgeStyle)
    show_axes: bool | None = None
    axes_style: AxesStyle = field(default_factory=AxesStyle)

    def __post_init__(self) -> None:
        if isinstance(self.slab_clip_mode, str):
            self.slab_clip_mode = SlabClipMode(self.slab_clip_mode)
        if isinstance(self.polyhedra_vertex_mode, str):
            self.polyhedra_vertex_mode = PolyhedraVertexMode(
                self.polyhedra_vertex_mode
            )
        if self.atom_scale <= 0:
            raise ValueError(f"atom_scale must be positive, got {self.atom_scale}")
        if self.bond_scale <= 0:
            raise ValueError(f"bond_scale must be positive, got {self.bond_scale}")
        if self.atom_outline_width < 0:
            raise ValueError(
                f"atom_outline_width must be non-negative, got {self.atom_outline_width}"
            )
        if self.bond_outline_width < 0:
            raise ValueError(
                f"bond_outline_width must be non-negative, got {self.bond_outline_width}"
            )
        if self.circle_segments < 3:
            raise ValueError(
                f"circle_segments must be >= 3, got {self.circle_segments}"
            )
        if self.arc_segments < 2:
            raise ValueError(
                f"arc_segments must be >= 2, got {self.arc_segments}"
            )
        if self.interactive_circle_segments < 3:
            raise ValueError(
                f"interactive_circle_segments must be >= 3, "
                f"got {self.interactive_circle_segments}"
            )
        if self.interactive_arc_segments < 2:
            raise ValueError(
                f"interactive_arc_segments must be >= 2, "
                f"got {self.interactive_arc_segments}"
            )
        if not 0.0 <= self.polyhedra_shading <= 1.0:
            raise ValueError(
                f"polyhedra_shading must be between 0.0 and 1.0, "
                f"got {self.polyhedra_shading}"
            )
        if self.polyhedra_outline_width is not None and self.polyhedra_outline_width < 0:
            raise ValueError(
                f"polyhedra_outline_width must be non-negative, "
                f"got {self.polyhedra_outline_width}"
            )


@dataclass
class AtomStyle:
    """Visual style for an atomic species.

    Attributes:
        radius: Display radius in angstroms.  Typical values range
            from about 0.5 (hydrogen) to 2.0 (heavy metals).
            See :data:`COVALENT_RADII` for physically motivated
            starting points.
        colour: Fill colour specification (CSS name, hex string, grey
            float, or RGB tuple/list).  See :data:`Colour`.
    """

    radius: float
    colour: Colour


@dataclass
class BondSpec:
    """Declarative rule for bond detection between species pairs.

    The *species* pair is stored in sorted order so that the data
    structure is invariant under exchange of the two labels.

    Species names support fnmatch-style wildcards (``*``, ``?``).

    Attributes:
        species: Sorted pair of species patterns.
        min_length: Minimum bond length threshold.
        max_length: Maximum bond length threshold.
        radius: Visual radius of the bond cylinder.
        colour: Bond colour used when ``half_bonds`` is disabled on
            the render style.  When ``half_bonds`` is ``True`` (the
            default), each half of the bond is coloured to match the
            nearest atom and this field is ignored.
    """

    species: tuple[str, str]
    min_length: float
    max_length: float
    radius: float
    colour: Colour

    def __post_init__(self) -> None:
        self.species = tuple(sorted(self.species))  # type: ignore[assignment]

    def matches(self, species_1: str, species_2: str) -> bool:
        """Check whether this spec matches a given species pair.

        Matching is symmetric: ``BondSpec(("C", "H"), ...).matches("H", "C")``
        returns ``True``.

        Args:
            species_1: First species label.
            species_2: Second species label.

        Returns:
            ``True`` if the pair matches in either order.
        """
        a, b = self.species
        forward = fnmatch(species_1, a) and fnmatch(species_2, b)
        reverse = fnmatch(species_1, b) and fnmatch(species_2, a)
        return forward or reverse


@dataclass(frozen=True)
class Bond:
    """A computed bond between two atoms.

    Attributes:
        index_a: Index of the first atom.
        index_b: Index of the second atom.
        length: Interatomic distance.
        spec: The BondSpec rule that produced this bond.
    """

    index_a: int
    index_b: int
    length: float
    spec: BondSpec


@dataclass
class PolyhedronSpec:
    """Declarative rule for rendering coordination polyhedra.

    A polyhedron is drawn around each atom whose species matches
    *centre*, using its bonded neighbours as vertices of a convex
    hull.  Species names support fnmatch-style wildcards.

    Attributes:
        centre: Species pattern for the centre atom (e.g. ``"Ti"``).
        colour: Face colour, or ``None`` to inherit from the centre
            atom's style colour.
        alpha: Face transparency (0 = fully transparent, 1 = opaque).
        edge_colour: Colour for face wireframe edges.
        edge_width: Line width for face wireframe edges (points).
        hide_centre: Whether to hide the centre atom circle when
            a polyhedron is drawn.
        hide_bonds: Whether to hide bonds from the centre atom to
            its coordinating neighbours when a polyhedron is drawn.
        hide_vertices: Whether to hide the vertex atom circles.
            An atom is only hidden if *every* polyhedron it
            participates in has ``hide_vertices=True``.
        min_vertices: Minimum number of bonded neighbours required
            to draw a polyhedron.  Centre atoms with fewer neighbours
            are skipped.  ``None`` uses the default minimum of 3.
    """

    centre: str
    colour: Colour | None = None
    alpha: float = 0.4
    edge_colour: Colour = (0.15, 0.15, 0.15)
    edge_width: float = 1.0
    hide_centre: bool = False
    hide_bonds: bool = False
    hide_vertices: bool = False
    min_vertices: int | None = None


@dataclass(frozen=True)
class Polyhedron:
    """A computed coordination polyhedron.

    Attributes:
        centre_index: Index of the centre atom.
        neighbour_indices: Indices of the coordinating atoms.
        faces: List of faces, each a 1-D array of vertex indices
            into *neighbour_indices*.  Triangular faces have length 3;
            merged coplanar faces may have 4 or more vertices.
        spec: The PolyhedronSpec that produced this polyhedron.
    """

    centre_index: int
    neighbour_indices: tuple[int, ...]
    faces: list[np.ndarray]
    spec: PolyhedronSpec


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


@dataclass
class ViewState:
    """Camera state for 3D-to-2D projection.

    Encapsulates rotation, zoom, centring, and optional perspective
    projection. Renderers consume the projected 2D coordinates and
    depth values produced by :meth:`project`.

    Depth-slab clipping is controlled by :attr:`slab_near`,
    :attr:`slab_far`, and :attr:`slab_origin`.  When set, only atoms
    whose depth (along the viewing direction) falls within the range
    ``[origin_depth + slab_near, origin_depth + slab_far]`` are
    rendered.  If *slab_origin* is ``None``, the slab is centred on
    :attr:`centre`.

    Attributes:
        rotation: 3x3 rotation matrix.
        zoom: Magnification factor.
        centre: 3D point about which to centre the view.
        perspective: Perspective strength (0 = orthographic).
        view_distance: Distance from camera to scene centre.
        slab_origin: 3D point defining the slab reference depth, or
            ``None`` to use *centre*.
        slab_near: Near offset from the slab origin depth (negative =
            further from camera), or ``None`` for no near limit.
        slab_far: Far offset from the slab origin depth (positive =
            closer to camera), or ``None`` for no far limit.
    """

    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )
    zoom: float = 1.0
    centre: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    perspective: float = 0.0
    view_distance: float = 10.0
    slab_origin: np.ndarray | None = None
    slab_near: float | None = None
    slab_far: float | None = None

    def project(
        self, coords: np.ndarray, radii: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D coordinates to 2D with depth information.

        The eye sits at ``[0, 0, view_distance]`` and each sphere's
        visible silhouette is projected onto the z=0 plane.

        Args:
            coords: Array of shape ``(n, 3)``.
            radii: Optional array of shape ``(n,)`` giving 3D sphere
                radii.  When provided the returned *projected_radii*
                are the screen-space silhouette radii; otherwise zeros.

        Returns:
            Tuple of ``(xy, depth, projected_radii)`` where:

            - *xy*: ``(n, 2)`` projected 2D coordinates.
            - *depth*: ``(n,)`` depth values (larger = closer to viewer).
            - *projected_radii*: ``(n,)`` screen-space sphere radii.
        """
        coords = np.asarray(coords, dtype=float)
        centred = coords - self.centre
        rotated = centred @ self.rotation.T
        depth = rotated[:, 2]

        if self.perspective > 0:
            # Eye-to-atom distance along z.
            d = self.view_distance - depth * self.perspective
            scale = self.view_distance / d
            xy = rotated[:, :2] * scale[:, np.newaxis] * self.zoom

            if radii is not None:
                radii = np.asarray(radii, dtype=float)
                # Silhouette radius: r * D / sqrt(d^2 - r^2).
                denom = np.sqrt(np.maximum(d**2 - radii**2, 1e-12))
                projected_radii = radii * self.view_distance / denom * self.zoom
            else:
                projected_radii = np.zeros(len(depth))
        else:
            xy = rotated[:, :2] * self.zoom
            if radii is not None:
                projected_radii = np.asarray(radii, dtype=float) * self.zoom
            else:
                projected_radii = np.zeros(len(depth))

        return xy, depth, projected_radii

    def slab_mask(self, coords: np.ndarray) -> np.ndarray:
        """Return a boolean mask selecting atoms within the depth slab.

        If neither :attr:`slab_near` nor :attr:`slab_far` is set, all
        atoms are selected.  The depth of each atom is measured along
        the current viewing direction, relative to the slab origin
        (or :attr:`centre` if no origin is set).

        Args:
            coords: World-space coordinates, shape ``(n, 3)``.

        Returns:
            Boolean array of shape ``(n,)``.
        """
        if self.slab_near is None and self.slab_far is None:
            return np.ones(len(coords), dtype=bool)

        coords = np.asarray(coords, dtype=float)
        centred = coords - self.centre
        # Depth is the z-component in camera space.
        depth = centred @ self.rotation[2]

        # Compute the reference depth from slab_origin.
        if self.slab_origin is not None:
            origin_centred = np.asarray(self.slab_origin, dtype=float) - self.centre
            ref_depth = np.dot(origin_centred, self.rotation[2])
        else:
            ref_depth = 0.0

        relative_depth = depth - ref_depth

        mask = np.ones(len(coords), dtype=bool)
        if self.slab_near is not None:
            mask &= relative_depth >= self.slab_near
        if self.slab_far is not None:
            mask &= relative_depth <= self.slab_far
        return mask

    def look_along(
        self,
        direction: np.ndarray | list[float] | tuple[float, ...],
        *,
        up: np.ndarray | list[float] | tuple[float, ...] = (0.0, 1.0, 0.0),
    ) -> None:
        """Set the rotation so the camera looks along *direction*.

        The view is oriented so that *direction* points into the screen
        (along +z in camera space).  The *up* vector determines which
        way is "up" on screen.

        This is equivalent to placing the camera at a point along
        *direction* looking back towards the origin.

        Args:
            direction: 3D vector giving the viewing direction (from
                the camera towards the scene).  Need not be normalised.
            up: 3D vector indicating the upward direction in screen
                space.  Defaults to ``[0, 1, 0]``.

        Raises:
            ValueError: If *direction* is zero-length or *up* is
                parallel to *direction*.
        """
        d = np.asarray(direction, dtype=float)
        u = np.asarray(up, dtype=float)

        d_len = np.linalg.norm(d)
        if d_len < 1e-12:
            raise ValueError("direction must be non-zero")
        fwd = d / d_len                     # camera z-axis (into screen)

        right = np.cross(u, fwd)
        right_len = np.linalg.norm(right)
        if right_len < 1e-12:
            # Up is parallel to direction.  If the caller explicitly
            # provided an up vector, that is an error.  Otherwise
            # fall back to [0, 0, 1] as the up hint.
            default_up = (0.0, 1.0, 0.0)
            if tuple(float(x) for x in up) != default_up:
                raise ValueError(
                    "up vector is parallel to the viewing direction"
                )
            u = np.array([0.0, 0.0, 1.0])
            right = np.cross(u, fwd)
            right_len = np.linalg.norm(right)
        right /= right_len                  # camera x-axis

        up_actual = np.cross(fwd, right)     # camera y-axis

        # Rotation matrix: rows are the camera basis vectors.
        # R maps world coords to camera coords: rotated = R @ world.
        self.rotation = np.array([right, up_actual, fwd])


@dataclass
class StructureScene:
    """Top-level scene holding atoms, frames, styles, bond rules, and view.

    Attributes:
        species: One label per atom.
        frames: List of coordinate snapshots.
        atom_styles: Mapping from species label to visual style.
        bond_specs: Declarative bond detection rules.
        polyhedra: Declarative polyhedron rendering rules.
        view: Camera / projection state.
        title: Scene title for display.
        lattice: Unit cell lattice matrix, shape ``(3, 3)`` with rows
            as lattice vectors, or ``None`` for non-periodic structures.
        atom_data: Per-atom metadata arrays, keyed by name.  Each value
            must be a 1-D array of length ``n_atoms``.  Use
            :meth:`set_atom_data` to populate this and ``colour_by``
            on the render methods to visualise it.
    """

    species: list[str]
    frames: list[Frame]
    atom_styles: dict[str, AtomStyle] = field(default_factory=dict)
    bond_specs: list[BondSpec] = field(default_factory=list)
    polyhedra: list[PolyhedronSpec] = field(default_factory=list)
    view: ViewState = field(default_factory=ViewState)
    title: str = ""
    lattice: np.ndarray | None = None
    atom_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lattice is not None:
            self.lattice = np.asarray(self.lattice, dtype=float)
            if self.lattice.shape != (3, 3):
                raise ValueError(
                    f"lattice must have shape (3, 3), got {self.lattice.shape}"
                )
        n_atoms = len(self.species)
        for key, arr in self.atom_data.items():
            arr = np.asarray(arr)
            if arr.ndim != 1 or len(arr) != n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length {n_atoms}, "
                    f"got shape {arr.shape}"
                )
            self.atom_data[key] = arr

    @classmethod
    def from_xbs(
        cls,
        bs_path: str | Path,
        mv_path: str | Path | None = None,
    ) -> "StructureScene":
        """Create a StructureScene from XBS ``.bs`` (and optional ``.mv``) files.

        Args:
            bs_path: Path to the ``.bs`` structure file.
            mv_path: Optional path to a ``.mv`` trajectory file.  When
                provided, the scene will contain multiple frames.

        Returns:
            A fully configured StructureScene with styles and bond
            specs parsed from the file.

        See Also:
            :func:`hofmann.scene.from_xbs`
        """
        from hofmann.scene import from_xbs

        return from_xbs(bs_path, mv_path)

    @classmethod
    def from_pymatgen(
        cls,
        structure: "Structure | Sequence[Structure]",
        bond_specs: list[BondSpec] | None = None,
        *,
        polyhedra: list[PolyhedronSpec] | None = None,
        pbc: bool = True,
        pbc_padding: float | None = 0.1,
        centre_atom: int | None = None,
    ) -> "StructureScene":
        """Create a StructureScene from pymatgen ``Structure`` object(s).

        Args:
            structure: A single pymatgen ``Structure`` or a sequence of
                structures (e.g. from an MD trajectory).
            bond_specs: Bond detection rules.  ``None`` generates
                sensible defaults from covalent radii; pass an empty
                list to disable bonds.
            polyhedra: Polyhedron rendering rules.  ``None`` disables
                polyhedra.
            pbc: If ``True`` (the default), add periodic image atoms
                at cell boundaries so that bonds crossing periodic
                boundaries are drawn.  Set to ``False`` to disable
                all PBC expansion.
            pbc_padding: Cartesian margin (angstroms) around the unit
                cell for placing periodic image atoms.  The default of
                0.1 angstroms captures atoms on cell boundaries.
                ``None`` falls back to the maximum bond length from
                *bond_specs* for wider geometric expansion.
            centre_atom: Index of the atom to centre the unit cell on.
                Fractional coordinates are shifted so this atom sits
                at (0.5, 0.5, 0.5) before PBC expansion.

        Returns:
            A StructureScene with default element styles.

        Raises:
            ImportError: If pymatgen is not installed.

        See Also:
            :func:`hofmann.scene.from_pymatgen`
        """
        from hofmann.scene import from_pymatgen

        return from_pymatgen(
            structure, bond_specs, polyhedra=polyhedra,
            pbc=pbc, pbc_padding=pbc_padding, centre_atom=centre_atom,
        )

    def centre_on(self, atom_index: int, *, frame: int = 0) -> None:
        """Centre the view on a specific atom.

        Sets :attr:`view.centre` to the Cartesian position of the atom
        at *atom_index* in the given frame.

        Args:
            atom_index: Index of the atom to centre on.
            frame: Frame index to read coordinates from.
        """
        self.view.centre = self.frames[frame].coords[atom_index].copy()

    def set_atom_data(
        self,
        key: str,
        values: np.ndarray | Sequence[float] | Sequence[str] | dict[int, object],
    ) -> None:
        """Set per-atom metadata for colourmap-based rendering.

        Args:
            key: Name for this metadata (e.g. ``"charge"``,
                ``"site"``).
            values: Either an array-like of length ``n_atoms``, or a
                dict mapping atom indices to values.  When a dict is
                given, missing atoms are filled with ``NaN`` for
                numeric values or ``""`` for string values.

        Raises:
            ValueError: If an array-like has the wrong length, or a
                dict contains indices outside the valid range.
        """
        n_atoms = len(self.species)

        if isinstance(values, dict):
            if not values:
                raise ValueError("values dict must not be empty")
            for idx in values:
                if not 0 <= idx < n_atoms:
                    raise ValueError(
                        f"atom index {idx} out of range for "
                        f"{n_atoms} atoms"
                    )
            sample = next(iter(values.values()))
            if isinstance(sample, str):
                arr = np.array([""] * n_atoms, dtype=object)
                for idx, val in values.items():
                    arr[idx] = val
            else:
                arr = np.full(n_atoms, np.nan)
                for idx, val in values.items():
                    arr[idx] = val
        else:
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) != n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length {n_atoms}, "
                    f"got shape {arr.shape}"
                )

        self.atom_data[key] = arr

    def render_mpl(
        self,
        output: str | Path | None = None,
        *,
        style: RenderStyle | None = None,
        frame_index: int = 0,
        figsize: tuple[float, float] = (5.0, 5.0),
        dpi: int = 150,
        background: Colour = "white",
        show: bool | None = None,
        colour_by: str | list[str] | None = None,
        cmap: str | Callable[[float], tuple[float, float, float]] | object | list = "viridis",
        colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
        **style_kwargs: object,
    ) -> Figure:
        """Render the scene as a static matplotlib figure.

        Args:
            output: Optional file path to save the figure.  The format
                is inferred from the extension (``.svg``, ``.pdf``,
                ``.png``).
            style: A :class:`RenderStyle` controlling visual appearance.
                Any :class:`RenderStyle` field name may also be passed
                as a keyword argument to override individual fields.
            frame_index: Which frame to render (default 0).
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution for raster output formats.
            background: Background colour.
            show: Whether to call ``plt.show()``.  Defaults to
                ``True`` when *output* is ``None``, ``False`` when
                saving to a file.
            colour_by: Key into :attr:`atom_data` to colour atoms by.
                When ``None`` (the default), species-based colouring
                is used.
            cmap: Matplotlib colourmap name (e.g. ``"viridis"``),
                ``Colormap`` object, or callable mapping a float in
                ``[0, 1]`` to an ``(r, g, b)`` tuple.
            colour_range: Explicit ``(vmin, vmax)`` for normalising
                numerical data.  ``None`` auto-ranges from the data.
            **style_kwargs: Any :class:`RenderStyle` field name as a
                keyword argument (e.g. ``show_bonds=False``).

        Returns:
            The matplotlib :class:`~matplotlib.figure.Figure`.

        See Also:
            :func:`hofmann.render_mpl.render_mpl`
        """
        from hofmann.render_mpl import render_mpl

        return render_mpl(
            self, output, style=style, frame_index=frame_index,
            figsize=figsize, dpi=dpi, background=background,
            show=show, colour_by=colour_by, cmap=cmap,
            colour_range=colour_range, **style_kwargs,
        )

    def render_mpl_interactive(
        self,
        *,
        style: RenderStyle | None = None,
        frame_index: int = 0,
        figsize: tuple[float, float] = (5.0, 5.0),
        dpi: int = 150,
        background: Colour = "white",
        colour_by: str | list[str] | None = None,
        cmap: str | Callable[[float], tuple[float, float, float]] | object | list = "viridis",
        colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
        **style_kwargs: object,
    ) -> tuple[ViewState, RenderStyle]:
        """Open an interactive matplotlib viewer with mouse and keyboard controls.

        Left-drag rotates, scroll zooms, and keyboard shortcuts control
        rotation, pan, perspective, display toggles, and frame navigation.
        Press **h** to show a help overlay listing all keybindings.

        When the window is closed the updated :class:`ViewState` and
        :class:`RenderStyle` are returned so they can be reused for
        static rendering::

            view, style = scene.render_mpl_interactive()
            scene.view = view
            scene.render_mpl("output.svg", style=style)

        Args:
            style: A :class:`RenderStyle` controlling visual appearance.
                Any :class:`RenderStyle` field name may also be passed
                as a keyword argument to override individual fields.
            frame_index: Which frame to render initially.
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution.
            background: Background colour.
            colour_by: Key into :attr:`atom_data` to colour atoms by.
            cmap: Matplotlib colourmap name, object, or callable.
            colour_range: Explicit ``(vmin, vmax)`` for numerical data.
            **style_kwargs: Any :class:`RenderStyle` field name as a
                keyword argument (e.g. ``show_bonds=False``).

        Returns:
            A ``(ViewState, RenderStyle)`` tuple reflecting any view
            and style changes applied during the interactive session.

        See Also:
            :func:`hofmann.render_mpl.render_mpl_interactive`
        """
        from hofmann.render_mpl import render_mpl_interactive

        return render_mpl_interactive(
            self, style=style, frame_index=frame_index,
            figsize=figsize, dpi=dpi, background=background,
            colour_by=colour_by, cmap=cmap, colour_range=colour_range,
            **style_kwargs,
        )
