"""Projection modes for the camera: parallel and perspective."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


#: Stand-in eye distance for parallel projections: far enough that all
#: view rays are effectively parallel (matching XBS pmode == 0).
_PARALLEL_EYE_DISTANCE = 1e6


def _sqrt_difference_of_squares(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sqrt(a**2 - b**2)`` for non-negative *a*, *b*, without overflow.

    Evaluated as ``sqrt(a - b) * sqrt(a + b)``: forming ``a**2 - b**2``
    first overflows to ``inf`` for *a* beyond about 1.3e154, and the
    caller's division by that ``inf`` then drives the silhouette to zero.
    ``a < b`` clamps to zero (the eye is inside the sphere; the caller warns).
    """
    return np.sqrt(np.maximum(a - b, 0.0)) * np.sqrt(a + b)


class Projection(ABC):
    """A camera projection mode.

    Concrete variants map camera-space geometry to the screen; see
    :class:`Orthographic` and :class:`Perspective`.
    """

    # Empty slots keep the ABC slots-friendly: without it, subclasses
    # declaring ``slots=True`` would still gain a ``__dict__`` from this
    # base, so a mistyped attribute would silently stick.
    __slots__ = ()

    @abstractmethod
    def to_screen(self, camera: np.ndarray) -> np.ndarray:
        """Map camera-space ``(n, 3)`` to screen-space ``(n, 2)``, before zoom."""

    @abstractmethod
    def silhouette_radius(
        self, depth: np.ndarray, radii: np.ndarray,
    ) -> np.ndarray:
        """Screen-space sphere silhouette radii, before zoom."""

    @abstractmethod
    def max_magnification(self, worst_depth: float) -> float:
        """Worst-case screen magnification for a point at that depth."""

    @property
    @abstractmethod
    def eye_distance(self) -> float:
        """Reference eye distance for bond-cap foreshortening."""

    @abstractmethod
    def reaches_eye_plane(self, depth: np.ndarray) -> bool:
        """Whether any point is at or behind the eye (a degenerate view)."""


@dataclass(frozen=True, slots=True)
class Orthographic(Projection):
    """Parallel projection: depth is not foreshortened."""

    def to_screen(self, camera: np.ndarray) -> np.ndarray:
        return camera[:, :2]

    def silhouette_radius(
        self, depth: np.ndarray, radii: np.ndarray,
    ) -> np.ndarray:
        return radii

    def max_magnification(self, worst_depth: float) -> float:
        return 1.0

    @property
    def eye_distance(self) -> float:
        return _PARALLEL_EYE_DISTANCE

    def reaches_eye_plane(self, depth: np.ndarray) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Perspective(Projection):
    """Perspective projection with the eye on the camera's +z axis.

    Screen positions are scaled by ``D / (D - z * s)`` for an atom at
    camera depth *z*, writing *D* for :attr:`view_distance` and *s*
    for :attr:`strength`.  That places the eye at ``D / s``, so a
    *strength* of ``1.0`` is a true pinhole camera at
    :attr:`view_distance`; smaller values move the eye further out,
    weakening the foreshortening.

    Attributes:
        strength: Perspective strength.  Must be positive; use
            :class:`Orthographic` for a parallel projection.
        view_distance: Reference distance from the scene centre,
            equal to the eye distance at ``strength = 1``.
    """

    strength: float = 0.5
    view_distance: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or self.strength <= 0:
            raise ValueError(
                "strength must be finite and positive (use Orthographic "
                f"for a parallel projection), got {self.strength}"
            )
        if not math.isfinite(self.view_distance) or self.view_distance <= 0:
            raise ValueError(
                "view_distance must be finite and positive, got "
                f"{self.view_distance}"
            )

    def to_screen(self, camera: np.ndarray) -> np.ndarray:
        d = self.view_distance - camera[:, 2] * self.strength
        # errstate: at or behind the eye plane the divisor is 0 or
        # negative, producing inf/negative on purpose
        # (ViewState.project_camera warns), so numpy's divide/invalid
        # warnings are silenced.
        with np.errstate(divide="ignore", invalid="ignore"):
            return camera[:, :2] * (self.view_distance / d)[:, np.newaxis]

    def silhouette_radius(
        self, depth: np.ndarray, radii: np.ndarray,
    ) -> np.ndarray:
        d = self.view_distance - depth * self.strength
        rs = radii * self.strength
        if np.any(np.abs(d) <= rs):
            warnings.warn(
                "a sphere contains the perspective eye and is drawn "
                "with an unbounded silhouette; increase view_distance "
                "or reduce strength.",
                UserWarning,
                stacklevel=2,
            )
        # Silhouette radius r*D/sqrt(d^2 - (r*s)^2), the eye at D/s.
        # abs(d) keeps a real denominator for atoms behind the eye
        # (d < 0); the 1e-6 floor gives a huge finite radius when the
        # eye is inside a sphere.
        denom = np.maximum(_sqrt_difference_of_squares(np.abs(d), rs), 1e-6)
        return radii * self.view_distance / denom

    def max_magnification(self, worst_depth: float) -> float:
        denom = self.view_distance - worst_depth * self.strength
        return self.view_distance / (denom if denom > 0 else 1e-6)

    @property
    def eye_distance(self) -> float:
        return self.view_distance / self.strength

    def reaches_eye_plane(self, depth: np.ndarray) -> bool:
        return bool(np.any(self.view_distance - depth * self.strength <= 0))
