"""Site composition value type for partial / mixed occupancy."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

_OCCUPANCY_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class Composition(Mapping[str, float]):
    """Species-to-occupancy mapping for a (possibly mixed) site.

    A frozen, hashable ``Mapping[str, float]`` describing how a site is
    occupied: a single species at full occupancy (a "pure" site, also
    expressible as a plain ``str``), a weighted mix of species (a
    "mixed" site), or any of the above with an implicit vacancy fraction
    (``1 - sum(occupancies)``).

    Validated at construction:

    - All occupancy values must be finite and lie in ``[0, 1]``.
      Zero values are dropped before further validation; negative
      or non-finite values raise.
    - The occupancy sum must not exceed ``1.0`` (within a tolerance of
      ``1e-9``).  Any deficit is interpreted as a vacancy fraction.
    - Keys must be non-empty strings.

    Iteration order is canonical: descending occupancy, with alphabetical
    tiebreak.  This ordering determines wedge layout in the renderer, so
    visual reproducibility is guaranteed across runs.

    Args:
        occupancies: A mapping of species labels to occupancy fractions.

    Raises:
        ValueError: If any value is non-finite or outside ``[0, 1]``;
            if the sum exceeds 1.0; if the resulting mapping is empty
            (all values zero, or the input was empty).
        TypeError: If any key is not a string.
    """

    occupancies: Mapping[str, float]

    def __post_init__(self) -> None:
        raw = dict(self.occupancies)

        for key in raw:
            if not isinstance(key, str):
                raise TypeError(
                    f"Composition keys must be strings, got "
                    f"{type(key).__name__}: {key!r}"
                )
            if not key.strip():
                raise ValueError(
                    "Composition keys must be non-empty, non-whitespace species labels"
                )

        cleaned: dict[str, float] = {}
        for key, value in raw.items():
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(
                    f"occupancy for {key!r} must be finite, got {v}"
                )
            if v < 0:
                raise ValueError(
                    f"occupancy for {key!r} must be non-negative, got {v}"
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

        # Canonical order: descending occupancy, then alphabetical.
        ordered = dict(sorted(
            cleaned.items(), key=lambda kv: (-kv[1], kv[0]),
        ))
        object.__setattr__(self, "occupancies", MappingProxyType(ordered))

    def __getitem__(self, key: str) -> float:
        return self.occupancies[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.occupancies)

    def __len__(self) -> int:
        return len(self.occupancies)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Composition):
            return NotImplemented
        return dict(self.occupancies) == dict(other.occupancies)

    def __hash__(self) -> int:
        return hash(tuple(self.occupancies.items()))

    @property
    def species(self) -> frozenset[str]:
        """Set of constituent species (vacancy excluded)."""
        return frozenset(self.occupancies.keys())

    @property
    def dominant_species(self) -> str:
        """Species with the highest occupancy.  Tiebreak: alphabetical."""
        # Iteration order is already canonical (descending occupancy,
        # then alphabetical), so the first key is the dominant species.
        return next(iter(self.occupancies))

    @property
    def vacancy(self) -> float:
        """Vacancy fraction: ``1 - sum(occupancies)``, clamped to [0, 1)."""
        deficit = 1.0 - sum(self.occupancies.values())
        return max(0.0, deficit)
