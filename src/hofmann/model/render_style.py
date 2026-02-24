from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from hofmann.model._util import _field_defaults
from hofmann.model.colour import Colour, normalise_colour


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

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted.
        """
        defaults = _field_defaults(type(self))
        d: dict = {}
        if normalise_colour(self.colour) != normalise_colour(defaults["colour"]):
            d["colour"] = list(normalise_colour(self.colour))
        if self.line_width != defaults["line_width"]:
            d["line_width"] = self.line_width
        if self.linestyle != defaults["linestyle"]:
            d["linestyle"] = self.linestyle
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CellEdgeStyle:
        """Deserialise from a dictionary."""
        kwargs: dict = {}
        for key in _field_defaults(cls):
            if key in d:
                val = d[key]
                if key == "colour" and isinstance(val, list):
                    val = tuple(val)
                kwargs[key] = val
        return cls(**kwargs)


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
    arrow_length: float = 0.12
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

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted.
        """
        _SPECIAL = frozenset({"colours", "labels", "corner"})
        defaults = _field_defaults(type(self), exclude=_SPECIAL)
        d: dict = {}
        default_colours = tuple(
            normalise_colour(c) for c in type(self).colours
        )
        actual_colours = tuple(normalise_colour(c) for c in self.colours)
        if actual_colours != default_colours:
            d["colours"] = [list(c) for c in actual_colours]
        if self.labels != type(self).labels:
            d["labels"] = list(self.labels)
        for field_name, default in defaults.items():
            val = getattr(self, field_name)
            if val != default:
                d[field_name] = val
        if isinstance(self.corner, WidgetCorner):
            if self.corner != type(self).corner:
                d["corner"] = self.corner.value
        else:
            d["corner"] = list(self.corner)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AxesStyle:
        """Deserialise from a dictionary."""
        _SPECIAL = frozenset({"colours", "labels", "corner"})
        defaults = _field_defaults(cls, exclude=_SPECIAL)
        kwargs: dict = {}
        if "colours" in d:
            kwargs["colours"] = tuple(
                tuple(c) if isinstance(c, list) else c for c in d["colours"]
            )
        if "labels" in d:
            kwargs["labels"] = tuple(d["labels"])
        for field_name in defaults:
            if field_name in d:
                kwargs[field_name] = d[field_name]
        if "corner" in d:
            val = d["corner"]
            if isinstance(val, list):
                kwargs["corner"] = tuple(val)
            else:
                kwargs["corner"] = val  # str -> WidgetCorner in __post_init__
        return cls(**kwargs)


_DEFAULT_CIRCLE_RADIUS: float = 5.0
"""Default legend circle radius in points."""

_DEFAULT_SPACING: float = 2.5
"""Default vertical spacing between legend entries in points."""


class LegendItem:
    """A single entry in the species legend.

    Each item carries a *key* (used as an identifier and fallback
    display label), a *colour* for the legend circle, an optional
    *label* override, and an optional *radius* in points.

    The class follows the same validated-property pattern as
    :class:`~hofmann.model.BondSpec`: ``colour`` and ``radius`` are
    backed by private fields with setters that validate on every
    assignment.

    Args:
        key: Identifier for this legend entry.  Also used as the
            default display label when *label* is ``None``.
        colour: Fill colour for the legend marker.  Accepts any
            format understood by :func:`normalise_colour`; the value
            is normalised to an ``(R, G, B)`` tuple on assignment.
        label: Display label text.  ``None`` falls back to *key*.
            Common chemical notation is auto-formatted at render
            time: trailing charges become superscripts, embedded
            digits become subscripts.  Labels containing ``$`` are
            passed through as explicit matplotlib mathtext.
        radius: Marker radius in points (before display-space
            scaling).  ``None`` falls back to
            ``LegendStyle.circle_radius`` when that is a plain float,
            or to its default value otherwise (the proportional and
            per-species dict modes do not apply to individual items).
        sides: Number of sides for a regular-polygon marker, or
            ``None`` for a circle (the default).  Must be at least 3.
        rotation: Rotation of the polygon marker in degrees
            (default 0.0).  Ignored when *sides* is ``None``.
        gap_after: Vertical gap in points between this entry and the
            next one.  ``None`` falls back to
            ``LegendStyle.spacing``.  Must be non-negative.  Ignored
            for the final entry in the list.
        alpha: Opacity of the marker face, from 0.0 (fully transparent)
            to 1.0 (fully opaque, the default).  Marker outlines are
            unaffected and remain fully opaque.
    """

    @staticmethod
    def _validate_radius(value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(
                f"radius must be positive, got {value}"
            )

    @staticmethod
    def _validate_sides(value: int | None) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"sides must be an int >= 3, got {value!r} "
                f"(type {type(value).__name__})"
            )
        if value < 3:
            raise ValueError(
                f"sides must be >= 3, got {value}"
            )

    @staticmethod
    def _validate_alpha(value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"alpha must be between 0.0 and 1.0, got {value}"
            )

    @staticmethod
    def _validate_gap_after(value: float | None) -> None:
        if value is not None and value < 0:
            raise ValueError(
                f"gap_after must be non-negative, got {value}"
            )

    def __init__(
        self,
        key: str,
        colour: Colour,
        label: str | None = None,
        radius: float | None = None,
        sides: int | None = None,
        rotation: float = 0.0,
        gap_after: float | None = None,
        alpha: float = 1.0,
    ) -> None:
        if not key:
            raise ValueError("key must be non-empty")
        self.key = key
        self._colour = normalise_colour(colour)
        self.label = label
        self._radius = radius
        self._sides = sides
        self._rotation = float(rotation)
        self._gap_after = gap_after
        self._alpha = float(alpha)
        self._validate()

    def _validate(self) -> None:
        self._validate_radius(self._radius)
        self._validate_sides(self._sides)
        self._validate_gap_after(self._gap_after)
        self._validate_alpha(self._alpha)

    @property
    def colour(self) -> tuple[float, float, float]:
        """Fill colour for the legend circle (normalised RGB)."""
        return self._colour

    @colour.setter
    def colour(self, value: Colour) -> None:
        self._colour = normalise_colour(value)

    @property
    def radius(self) -> float | None:
        """Circle radius in points, or ``None`` for the style default."""
        return self._radius

    @radius.setter
    def radius(self, value: float | None) -> None:
        self._validate_radius(value)
        self._radius = value

    @property
    def sides(self) -> int | None:
        """Number of polygon sides, or ``None`` for a circle."""
        return self._sides

    @sides.setter
    def sides(self, value: int | None) -> None:
        self._validate_sides(value)
        self._sides = value

    @property
    def rotation(self) -> float:
        """Rotation of the polygon marker in degrees."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        self._rotation = float(value)

    @property
    def gap_after(self) -> float | None:
        """Gap in points after this entry, or ``None`` for the style default."""
        return self._gap_after

    @gap_after.setter
    def gap_after(self, value: float | None) -> None:
        self._validate_gap_after(value)
        self._gap_after = value

    @property
    def alpha(self) -> float:
        """Opacity of the marker face (0.0 = transparent, 1.0 = opaque)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._validate_alpha(value)
        self._alpha = float(value)

    @property
    def marker(self) -> str | tuple[int, int, float]:
        """Matplotlib marker specification derived from *sides* and *rotation*.

        Returns ``"o"`` for circles, or ``(sides, 0, rotation)`` for
        regular polygons.
        """
        if self._sides is None:
            return "o"
        return (self._sides, 0, self._rotation)

    @property
    def display_label(self) -> str:
        """The label to display, falling back to *key*."""
        return self.key if self.label is None else self.label

    def __repr__(self) -> str:
        parts = [
            f"key={self.key!r}",
            f"colour={self._colour!r}",
        ]
        if self.label is not None:
            parts.append(f"label={self.label!r}")
        if self._radius is not None:
            parts.append(f"radius={self._radius!r}")
        if self._sides is not None:
            parts.append(f"sides={self._sides!r}")
        if self._rotation != 0.0:
            parts.append(f"rotation={self._rotation!r}")
        if self._gap_after is not None:
            parts.append(f"gap_after={self._gap_after!r}")
        if self._alpha != 1.0:
            parts.append(f"alpha={self._alpha!r}")
        return f"LegendItem({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LegendItem):
            return NotImplemented
        return (
            self.key == other.key
            and self._colour == other._colour
            and self.label == other.label
            and self._radius == other._radius
            and self._sides == other._sides
            and self._rotation == other._rotation
            and self._gap_after == other._gap_after
            and self._alpha == other._alpha
        )

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        d: dict = {
            "key": self.key,
            "colour": list(self._colour),
        }
        if self.label is not None:
            d["label"] = self.label
        if self._radius is not None:
            d["radius"] = self._radius
        if self._sides is not None:
            d["sides"] = self._sides
        if self._rotation != 0.0:
            d["rotation"] = self._rotation
        if self._gap_after is not None:
            d["gap_after"] = self._gap_after
        if self._alpha != 1.0:
            d["alpha"] = self._alpha
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LegendItem:
        """Deserialise from a dictionary."""
        colour = d["colour"]
        if isinstance(colour, list):
            colour = tuple(colour)
        kwargs: dict = {"key": d["key"], "colour": colour}
        if "label" in d:
            kwargs["label"] = d["label"]
        if "radius" in d:
            kwargs["radius"] = d["radius"]
        if "sides" in d:
            kwargs["sides"] = d["sides"]
        if "rotation" in d:
            kwargs["rotation"] = d["rotation"]
        if "gap_after" in d:
            kwargs["gap_after"] = d["gap_after"]
        if "alpha" in d:
            kwargs["alpha"] = d["alpha"]
        return cls(**kwargs)


@dataclass(frozen=True)
class LegendStyle:
    """Visual style for the species legend widget.

    The widget draws a vertical column of coloured circles with labels
    beside them.  By default, entries are auto-generated from the
    scene's species and atom styles.  To display a fully custom legend
    (e.g. for ``colour_by`` data), pass a tuple of
    :class:`LegendItem` instances via the *items* parameter — this
    bypasses species auto-generation entirely.

    Attributes:
        corner: Widget position.  Pass a :class:`WidgetCorner` (or its
            string value) for automatic placement in one of the four
            viewport corners, offset by *margin*.  Pass an ``(x, y)``
            tuple of fractional viewport coordinates
            (0.0 = left/bottom, 1.0 = right/top) for an explicit
            position; *margin* is ignored in this case.
        margin: Offset from the corner as a fraction of the viewport
            half-extent.  Only used when *corner* is a
            :class:`WidgetCorner`.
        font_size: Font size for species labels in points.
        circle_radius: Controls the size of the coloured circles in
            points.  Accepts three forms:

            * **float** — uniform radius for all entries (default 5.0).
            * **tuple (min, max)** — proportional sizing.  Each
              species' circle radius is linearly interpolated between
              *min* and *max* based on its ``AtomStyle.radius``
              relative to the smallest and largest radii in the legend.
              When all atom radii are equal, *max* is used.
            * **dict[str, float]** — explicit per-species radii.
              Species not present in the dict use the class default
              (5.0 points).
        spacing: Vertical gap between legend entries in points.
        label_gap: Horizontal gap between the circle edge and the
            species label in points.
        species: Explicit list of species to include, in display
            order.  ``None`` (the default) auto-detects from the
            scene: unique species in first-seen order, filtered to
            those with ``visible=True`` in their atom style.
            Ignored when *items* is provided.
        labels: Custom display labels for legend entries, mapping
            species name to label string.  Common chemical notation
            is auto-formatted: trailing charges become superscripts
            (``"Sr2+"``), embedded digits become subscripts
            (``"TiO6"``).  Labels containing ``$`` are passed
            through as explicit matplotlib mathtext.  ``None`` (the
            default) uses species names for all entries.  Ignored
            when *items* is provided.
        items: Explicit legend entries.  When provided, the legend
            displays these items instead of auto-generating from
            species.  *species*, *labels*, and the tuple/dict forms
            of *circle_radius* are all ignored.  Items with
            ``radius=None`` fall back to *circle_radius* when that
            is a plain float, or to 5.0 points otherwise.
    """

    corner: WidgetCorner | tuple[float, float] = WidgetCorner.BOTTOM_RIGHT
    margin: float = 0.15
    font_size: float = 10.0
    circle_radius: float | tuple[float, float] | dict[str, float] = _DEFAULT_CIRCLE_RADIUS
    spacing: float = _DEFAULT_SPACING
    label_gap: float = 5.0
    species: tuple[str, ...] | None = None
    labels: dict[str, str] | None = None
    items: tuple[LegendItem, ...] | None = None

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
        if isinstance(self.circle_radius, dict):
            if len(self.circle_radius) == 0:
                raise ValueError("circle_radius dict must be non-empty")
            for v in self.circle_radius.values():
                if v <= 0:
                    raise ValueError(
                        f"circle_radius dict values must be positive, got {v}"
                    )
        elif isinstance(self.circle_radius, tuple):
            lo, hi = self.circle_radius
            if lo <= 0 or hi <= 0:
                raise ValueError(
                    f"circle_radius range values must be positive, "
                    f"got ({lo}, {hi})"
                )
            if lo > hi:
                raise ValueError(
                    f"circle_radius min must not exceed max, "
                    f"got ({lo}, {hi})"
                )
        else:
            if self.circle_radius <= 0:
                raise ValueError(
                    f"circle_radius must be positive, got {self.circle_radius}"
                )
        if self.spacing < 0:
            raise ValueError(
                f"spacing must be non-negative, got {self.spacing}"
            )
        if self.label_gap < 0:
            raise ValueError(
                f"label_gap must be non-negative, got {self.label_gap}"
            )
        if self.margin < 0:
            raise ValueError(
                f"margin must be non-negative, got {self.margin}"
            )
        if self.species is not None and len(self.species) == 0:
            raise ValueError("species must be non-empty when provided")
        if self.items is not None:
            if len(self.items) == 0:
                raise ValueError("items must be non-empty when provided")
            for i, entry in enumerate(self.items):
                if not isinstance(entry, LegendItem):
                    raise TypeError(
                        f"items[{i}] must be a LegendItem, got "
                        f"{type(entry).__name__}"
                    )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted.
        """
        _SPECIAL = frozenset({
            "corner", "species", "circle_radius", "labels", "items",
        })
        defaults = _field_defaults(type(self), exclude=_SPECIAL)
        d: dict = {}
        for field_name, default in defaults.items():
            val = getattr(self, field_name)
            if val != default:
                d[field_name] = val
        # circle_radius: tuple → list, dict → dict, float → omit if default.
        if isinstance(self.circle_radius, tuple):
            d["circle_radius"] = list(self.circle_radius)
        elif isinstance(self.circle_radius, dict):
            d["circle_radius"] = dict(self.circle_radius)
        elif self.circle_radius != _DEFAULT_CIRCLE_RADIUS:
            d["circle_radius"] = self.circle_radius
        if isinstance(self.corner, WidgetCorner):
            if self.corner != type(self).corner:
                d["corner"] = self.corner.value
        else:
            d["corner"] = list(self.corner)
        if self.species is not None:
            d["species"] = list(self.species)
        if self.labels is not None:
            d["labels"] = dict(self.labels)
        if self.items is not None:
            d["items"] = [item.to_dict() for item in self.items]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LegendStyle:
        """Deserialise from a dictionary."""
        _SPECIAL = frozenset({
            "corner", "species", "circle_radius", "labels", "items",
        })
        defaults = _field_defaults(cls, exclude=_SPECIAL)
        kwargs: dict = {}
        for field_name in defaults:
            if field_name in d:
                kwargs[field_name] = d[field_name]
        if "circle_radius" in d:
            val = d["circle_radius"]
            if isinstance(val, list):
                kwargs["circle_radius"] = tuple(val)
            else:
                kwargs["circle_radius"] = val  # float or dict
        if "corner" in d:
            val = d["corner"]
            if isinstance(val, list):
                kwargs["corner"] = tuple(val)
            else:
                kwargs["corner"] = val  # str -> WidgetCorner in __post_init__
        if "species" in d:
            kwargs["species"] = tuple(d["species"])
        if "labels" in d:
            kwargs["labels"] = d["labels"]
        if "items" in d:
            kwargs["items"] = tuple(
                LegendItem.from_dict(item) for item in d["items"]
            )
        return cls(**kwargs)


@dataclass
class RenderStyle:
    """Visual style settings for rendering.

    Groups all appearance parameters that control how a scene is drawn,
    independent of the scene data itself.  A default ``RenderStyle()``
    gives the standard ball-and-stick look.

    Pass a style to :func:`~hofmann.rendering.static.render_mpl` via the
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
        show_legend: Whether to draw the species legend.  ``False``
            (the default) suppresses drawing.  ``True`` draws a
            legend showing each visible species with its colour.
        legend_style: Visual style for the legend widget.  See
            :class:`LegendStyle`.
        pbc: Whether to use the lattice for periodic bond
            computation and image-atom expansion.  Only meaningful
            when the scene has a lattice.  Set to ``False`` to
            disable all periodic boundary handling and render only
            the physical atoms with Euclidean bond detection.
        pbc_padding: Cartesian margin (angstroms) for geometric
            cell-face expansion.  Atoms within this distance of a
            unit cell face are duplicated on the opposite side,
            producing an expanded view of the structure.  ``None``
            disables geometric expansion.  The default of ``0.1``
            angstroms gives a thin shell that catches atoms sitting
            exactly on cell edges.
        max_recursive_depth: Maximum iterations for recursive bond
            expansion.  Only relevant when one or more *bond_specs*
            have ``recursive=True``.  Must be >= 1.
        deduplicate_molecules: Whether to remove duplicate molecular
            fragments that span cell boundaries.  When ``True``,
            each molecule appears only once, keeping the largest
            connected cluster.

    Raises:
        ValueError: If *atom_scale* or *bond_scale* are not positive,
            *max_recursive_depth* is less than 1,
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
    polyhedra_outline_width: float | None = None
    show_cell: bool | None = None
    cell_style: CellEdgeStyle = field(default_factory=CellEdgeStyle)
    show_axes: bool | None = None
    axes_style: AxesStyle = field(default_factory=AxesStyle)
    show_legend: bool = False
    legend_style: LegendStyle = field(default_factory=LegendStyle)
    pbc: bool = True
    pbc_padding: float | None = 0.1
    max_recursive_depth: int = 5
    deduplicate_molecules: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.slab_clip_mode, str):
            self.slab_clip_mode = SlabClipMode(self.slab_clip_mode)
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
        if self.max_recursive_depth < 1:
            raise ValueError(
                f"max_recursive_depth must be >= 1, "
                f"got {self.max_recursive_depth}"
            )
        if self.pbc_padding is not None and self.pbc_padding < 0:
            raise ValueError(
                f"pbc_padding must be non-negative, "
                f"got {self.pbc_padding}"
            )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted.  Nested
        ``cell_style``, ``axes_style``, and ``legend_style`` are
        serialised as sub-dicts (omitted entirely when they equal
        their own defaults).
        """
        defaults = _field_defaults(type(self))
        d: dict = {}
        for field_name, default in defaults.items():
            val = getattr(self, field_name)
            if field_name == "outline_colour":
                if normalise_colour(val) != normalise_colour(default):
                    d[field_name] = list(normalise_colour(val))
            elif field_name == "bond_colour":
                if val is not None:
                    d[field_name] = list(normalise_colour(val))
            elif field_name == "slab_clip_mode":
                if val != default:
                    d[field_name] = val.value
            else:
                if val != default:
                    d[field_name] = val

        cell_d = self.cell_style.to_dict()
        if cell_d:
            d["cell_style"] = cell_d
        axes_d = self.axes_style.to_dict()
        if axes_d:
            d["axes_style"] = axes_d
        legend_d = self.legend_style.to_dict()
        if legend_d:
            d["legend_style"] = legend_d
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RenderStyle:
        """Deserialise from a dictionary.

        Missing fields use their defaults.  The ``slab_clip_mode``
        string is coerced to :class:`SlabClipMode` and ``bond_colour``
        lists are converted to tuples for type consistency.
        """
        defaults = _field_defaults(cls)
        kwargs: dict = {}
        for field_name in defaults:
            if field_name in d:
                val = d[field_name]
                if field_name == "slab_clip_mode" and isinstance(val, str):
                    val = SlabClipMode(val)
                elif field_name == "bond_colour" and isinstance(val, list):
                    val = tuple(val)
                elif field_name == "outline_colour" and isinstance(val, list):
                    val = tuple(val)
                kwargs[field_name] = val
        if "cell_style" in d:
            kwargs["cell_style"] = CellEdgeStyle.from_dict(d["cell_style"])
        if "axes_style" in d:
            kwargs["axes_style"] = AxesStyle.from_dict(d["axes_style"])
        if "legend_style" in d:
            kwargs["legend_style"] = LegendStyle.from_dict(d["legend_style"])
        return cls(**kwargs)
