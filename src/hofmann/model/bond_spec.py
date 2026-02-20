from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import ClassVar

from hofmann.model.colour import Colour, normalise_colour


class BondSpec:
    """Declarative rule for bond detection between species pairs.

    The *species* pair is stored in sorted order so that the data
    structure is invariant under exchange of the two labels.

    Species names support fnmatch-style wildcards (``*``, ``?``).

    Only *species* and *max_length* are required.  *radius* and
    *colour* default to ``None``, meaning "use the class-level
    default" (``BondSpec.default_radius`` and
    ``BondSpec.default_colour``).  The resolved value is returned
    by the corresponding property; ``repr()`` shows
    ``radius=<default 0.1>`` when unset so the intent is visible.

    To change the defaults for all specs that have not been
    explicitly set::

        BondSpec.default_radius = 0.15
        BondSpec.default_colour = "grey"

    Attributes:
        species: Sorted pair of species patterns.
        max_length: Maximum bond length threshold.
        min_length: Minimum bond length threshold.  Defaults to
            ``0.0``.
        radius: Visual radius of the bond cylinder.  Defaults to
            ``BondSpec.default_radius`` (``0.1``) when not set
            explicitly.
        colour: Bond colour used when ``half_bonds`` is disabled on
            the render style.  When ``half_bonds`` is ``True`` (the
            default), each half of the bond is coloured to match the
            nearest atom and this field is ignored.  Defaults to
            ``BondSpec.default_colour`` (``0.5``, grey) when not set
            explicitly.
        complete: Controls single-pass bond completion across periodic
            boundaries.  A species string (e.g. ``"Zr"``) adds
            missing partners around visible atoms of that species.
            ``"*"`` completes around both species in the pair.
            ``False`` (default) disables completion.  Unlike
            *recursive*, newly added atoms are **not** themselves
            searched.
        recursive: If ``True``, atoms bonded via this spec are
            searched recursively across periodic boundaries.  When
            an image atom is materialised, its own bonded partners
            are checked on the next iteration, extending the
            expansion outward.  Useful for molecules that span
            periodic boundaries.
    """

    default_radius: ClassVar[float] = 0.1
    """Class-level default for *radius* when not set explicitly."""

    default_colour: ClassVar[Colour] = 0.5
    """Class-level default for *colour* when not set explicitly."""

    def __init__(
        self,
        species: tuple[str, str],
        max_length: float,
        min_length: float = 0.0,
        radius: float | None = None,
        colour: Colour | None = None,
        complete: bool | str = False,
        recursive: bool = False,
    ) -> None:
        a, b = sorted(species)
        self.species = (a, b)
        self.max_length = max_length
        self.min_length = min_length
        self._radius = radius
        self._colour = colour
        self.complete = complete
        self.recursive = recursive

        if self.max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {self.max_length}"
            )
        if self.min_length < 0:
            raise ValueError(
                f"min_length must be non-negative, got {self.min_length}"
            )
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must not exceed "
                f"max_length ({self.max_length})"
            )
        if self._radius is not None and self._radius < 0:
            raise ValueError(
                f"radius must be non-negative, got {self._radius}"
            )

        if self.complete is True:
            raise ValueError(
                "complete=True is not supported; use a species name "
                "(e.g. complete='Zr') or complete='*' for both directions"
            )
        if self.complete is not False and not isinstance(self.complete, str):
            raise ValueError(
                "complete must be False, a species name string "
                "(e.g. 'Zr'), or '*' for both directions"
            )
        if isinstance(self.complete, str) and self.complete != "*":
            if self.complete == "":
                raise ValueError("complete must not be an empty string")
            if not any(
                fnmatch(sp, self.complete) for sp in self.species
            ):
                raise ValueError(
                    f"complete={self.complete!r} does not match either "
                    f"species in the pair {self.species}"
                )

    @property
    def radius(self) -> float:
        """Visual radius of the bond cylinder."""
        return self.default_radius if self._radius is None else self._radius

    @radius.setter
    def radius(self, value: float | None) -> None:
        if value is not None and value < 0:
            raise ValueError(f"radius must be non-negative, got {value}")
        self._radius = value

    @property
    def colour(self) -> Colour:
        """Bond colour (resolved from class default when not set explicitly)."""
        return self.default_colour if self._colour is None else self._colour

    @colour.setter
    def colour(self, value: Colour | None) -> None:
        self._colour = value

    def __repr__(self) -> str:
        radius_repr = (
            f"<default {self.default_radius!r}>"
            if self._radius is None
            else repr(self._radius)
        )
        colour_repr = (
            f"<default {self.default_colour!r}>"
            if self._colour is None
            else repr(self._colour)
        )
        parts = [
            f"species={self.species!r}",
            f"max_length={self.max_length!r}",
            f"min_length={self.min_length!r}",
            f"radius={radius_repr}",
            f"colour={colour_repr}",
        ]
        if self.complete is not False:
            parts.append(f"complete={self.complete!r}")
        if self.recursive:
            parts.append(f"recursive={self.recursive!r}")
        return f"BondSpec({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BondSpec):
            return NotImplemented
        return (
            self.species == other.species
            and self.max_length == other.max_length
            and self.min_length == other.min_length
            and self._radius == other._radius
            and self._colour == other._colour
            and self.complete == other.complete
            and self.recursive == other.recursive
        )

    __hash__ = None  # type: ignore[assignment]

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

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Fields at their default values are omitted (``min_length=0``,
        ``complete=False``, ``recursive=False``).  When ``radius`` or
        ``colour`` use the class-level default (i.e. were not set
        explicitly), they are omitted too.
        """
        d: dict = {
            "species": list(self.species),
            "max_length": self.max_length,
        }
        if self.min_length != 0.0:
            d["min_length"] = self.min_length
        if self._radius is not None:
            d["radius"] = self._radius
        if self._colour is not None:
            d["colour"] = list(normalise_colour(self._colour))
        if self.complete is not False:
            d["complete"] = self.complete
        if self.recursive:
            d["recursive"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BondSpec:
        """Deserialise from a dictionary.

        Missing optional fields use their defaults.
        """
        return cls(
            species=tuple(d["species"]),
            max_length=d["max_length"],
            min_length=d.get("min_length", 0.0),
            radius=d.get("radius"),
            colour=d.get("colour"),
            complete=d.get("complete", False),
            recursive=d.get("recursive", False),
        )


@dataclass(frozen=True)
class Bond:
    """A computed bond between two atoms.

    Attributes:
        index_a: Index of the first atom.
        index_b: Index of the second atom.
        length: Interatomic distance.
        spec: The BondSpec rule that produced this bond.
        image: Lattice translation applied to atom b to form the bond.
            ``(0, 0, 0)`` means a direct bond within the cell.
            ``(1, 0, 0)`` means atom b is shifted by +1 along lattice
            vector **a**.  Always ``(0, 0, 0)`` for non-periodic scenes.
    """

    index_a: int
    index_b: int
    length: float
    spec: BondSpec
    image: tuple[int, int, int] = (0, 0, 0)

    def __hash__(self) -> int:
        return hash((self.index_a, self.index_b, self.image))
