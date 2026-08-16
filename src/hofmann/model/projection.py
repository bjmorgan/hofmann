"""Projection modes for the camera: parallel and perspective."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


#: Stand-in eye distance for parallel projections: far enough that all
#: view rays are effectively parallel (matching XBS pmode == 0).
_PARALLEL_EYE_DISTANCE = 1e6


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
        # negative, producing inf/negative on purpose (project_camera
        # warns), so numpy's divide/invalid warnings are silenced.
        with np.errstate(divide="ignore", invalid="ignore"):
            return camera[:, :2] * (self.view_distance / d)[:, np.newaxis]

    def silhouette_radius(
        self, depth: np.ndarray, radii: np.ndarray,
    ) -> np.ndarray:
        d = self.view_distance - depth * self.strength
        # Silhouette radius r*D/sqrt(d^2 - r^2), approximate: the eye is
        # at D/strength, for which the exact form carries (r*strength)^2.
        # Bond end caps use the same reference distance D (bond_geometry),
        # so the two share an eye.
        denom = np.sqrt(np.maximum(d**2 - radii**2, 1e-12))
        return radii * self.view_distance / denom

    def max_magnification(self, worst_depth: float) -> float:
        denom = self.view_distance - worst_depth * self.strength
        return self.view_distance / (denom if denom > 0 else 1e-6)

    @property
    def eye_distance(self) -> float:
        return self.view_distance

    def reaches_eye_plane(self, depth: np.ndarray) -> bool:
        return bool(np.any(self.view_distance - depth * self.strength <= 0))
