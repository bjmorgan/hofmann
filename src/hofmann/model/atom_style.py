from __future__ import annotations

from hofmann.model.colour import Colour, normalise_colour


class AtomStyle:
    """Visual style for an atomic species.

    For visible atoms, *radius* and *colour* are required.  For hidden
    atoms (``visible=False``), they may be omitted — or provided to
    preserve styling when toggling visibility later.

    Args:
        radius: Display radius in angstroms.  Required when
            ``visible=True``.
        colour: Fill colour specification (CSS name, hex string, grey
            float, or RGB tuple/list).  Required when ``visible=True``.
        visible: Whether atoms of this species are drawn.  Set to
            ``False`` to hide atoms without removing them from the
            scene.  Bonds to hidden atoms are also suppressed.

    Raises:
        ValueError: If ``visible=True`` and *radius* or *colour* is
            not provided, or if *radius* is not positive.
    """

    def __init__(
        self,
        radius: float | None = None,
        colour: Colour | None = None,
        *,
        visible: bool = True,
    ) -> None:
        if visible:
            if radius is None:
                raise ValueError(
                    "radius is required for visible atoms"
                )
            if colour is None:
                raise ValueError(
                    "colour is required for visible atoms"
                )
        if radius is not None and radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._radius = radius
        self._colour = colour
        self._visible = visible

    @property
    def radius(self) -> float | None:
        """Display radius in angstroms, or ``None`` when not set."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"radius must be a number, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"radius must be positive, got {value}")
        self._radius = value

    @property
    def colour(self) -> Colour | None:
        """Fill colour, or ``None`` when not set."""
        return self._colour

    @colour.setter
    def colour(self, value: Colour) -> None:
        if value is None:
            raise TypeError("colour cannot be set to None")
        self._colour = value

    @property
    def visible(self) -> bool:
        """Whether atoms of this species are drawn."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if value and (self._radius is None or self._colour is None):
            raise ValueError(
                "cannot make atom visible without radius and colour; "
                "set radius and colour first"
            )
        self._visible = value

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Colours are normalised to ``[r, g, b]`` lists.  Fields that
        are ``None`` (hidden atoms without explicit style) are omitted.
        The ``visible`` field is omitted when ``True`` (the default).
        """
        d: dict = {}
        if self._radius is not None:
            d["radius"] = self._radius
        if self._colour is not None:
            d["colour"] = list(normalise_colour(self._colour))
        if not self._visible:
            d["visible"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AtomStyle:
        """Deserialise from a dictionary.

        Accepts any colour format understood by
        :func:`normalise_colour`.
        """
        return cls(
            radius=d.get("radius"),
            colour=d.get("colour"),
            visible=d.get("visible", True),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomStyle):
            return NotImplemented
        return (
            self._radius == other._radius
            and self._colour == other._colour
            and self._visible == other._visible
        )

    __hash__ = None  # type: ignore[assignment]  # mutable

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._radius is not None:
            parts.append(f"radius={self._radius!r}")
        if self._colour is not None:
            parts.append(f"colour={self._colour!r}")
        if not self._visible:
            parts.append("visible=False")
        return f"AtomStyle({', '.join(parts)})"
