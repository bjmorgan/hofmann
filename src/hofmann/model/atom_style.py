from __future__ import annotations

from dataclasses import dataclass

from hofmann.model.colour import Colour, normalise_colour


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
        visible: Whether atoms of this species are drawn.  Set to
            ``False`` to hide atoms without removing them from the
            scene.  Bonds to hidden atoms are also suppressed.
    """

    radius: float
    colour: Colour
    visible: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Colours are normalised to ``[r, g, b]`` lists.  The
        ``visible`` field is omitted when ``True`` (the default).
        """
        d: dict = {
            "radius": self.radius,
            "colour": list(normalise_colour(self.colour)),
        }
        if not self.visible:
            d["visible"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AtomStyle:
        """Deserialise from a dictionary.

        Accepts any colour format understood by
        :func:`normalise_colour`.
        """
        return cls(
            radius=d["radius"],
            colour=d["colour"],
            visible=d.get("visible", True),
        )
