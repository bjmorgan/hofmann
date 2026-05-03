"""Site composition value type for partial / mixed occupancy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Composition(Mapping[str, float]):
    """Species-to-occupancy mapping for a (possibly mixed) site.

    See the design doc and module docstring for detailed semantics.
    """

    occupancies: Mapping[str, float]

    def __post_init__(self) -> None:
        # Wrap the supplied mapping in an immutable view so dataclass
        # frozenness is honoured at the value level too.
        object.__setattr__(
            self, "occupancies", MappingProxyType(dict(self.occupancies)),
        )

    def __getitem__(self, key: str) -> float:
        return self.occupancies[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.occupancies)

    def __len__(self) -> int:
        return len(self.occupancies)
