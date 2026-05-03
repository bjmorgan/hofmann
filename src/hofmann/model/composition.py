"""Site composition value type for partial / mixed occupancy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

_OCCUPANCY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Composition(Mapping[str, float]):
    """Species-to-occupancy mapping for a (possibly mixed) site."""

    occupancies: Mapping[str, float]

    def __post_init__(self) -> None:
        raw = dict(self.occupancies)

        # Type-check keys.
        for key in raw:
            if not isinstance(key, str):
                raise TypeError(
                    f"Composition keys must be strings, got "
                    f"{type(key).__name__}: {key!r}"
                )

        # Validate values, dropping zeros.
        cleaned: dict[str, float] = {}
        for key, value in raw.items():
            v = float(value)
            if v < 0:
                raise ValueError(
                    f"occupancy for {key!r} must be positive, got {v}"
                )
            if v > 1.0 + _OCCUPANCY_TOLERANCE:
                raise ValueError(
                    f"occupancy for {key!r} must be in (0, 1], got {v}"
                )
            if v == 0.0:
                continue
            cleaned[key] = v

        if not cleaned:
            raise ValueError("Composition must not be empty")

        total = sum(cleaned.values())
        if total > 1.0 + _OCCUPANCY_TOLERANCE:
            raise ValueError(
                f"Composition occupancies sum to {total}, must be <= 1.0"
            )

        object.__setattr__(self, "occupancies", MappingProxyType(cleaned))

    def __getitem__(self, key: str) -> float:
        return self.occupancies[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.occupancies)

    def __len__(self) -> int:
        return len(self.occupancies)
