"""Core data model for hofmann: dataclasses, colour handling, and projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Colour can be a CSS name / hex string, a grey float, or an RGB tuple.
Colour = str | float | tuple[float, float, float]


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

    if isinstance(colour, tuple):
        if len(colour) != 3:
            raise ValueError(
                f"RGB tuple must have 3 elements, got {len(colour)}"
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


@dataclass
class AtomStyle:
    """Visual style for an atomic species.

    Attributes:
        radius: Display radius for atoms of this species.
        colour: Fill colour specification.
    """

    radius: float
    colour: Colour


@dataclass
class BondSpec:
    """Declarative rule for bond detection between species pairs.

    Species names support fnmatch-style wildcards (``*``, ``?``).

    Attributes:
        species_a: First species pattern.
        species_b: Second species pattern.
        min_length: Minimum bond length threshold.
        max_length: Maximum bond length threshold.
        radius: Visual radius of the bond cylinder.
        colour: Bond colour specification.
    """

    species_a: str
    species_b: str
    min_length: float
    max_length: float
    radius: float
    colour: Colour

    def matches(self, species_1: str, species_2: str) -> bool:
        """Check whether this spec matches a given species pair.

        Matching is symmetric: ``BondSpec("C", "H").matches("H", "C")``
        returns ``True``.

        Args:
            species_1: First species label.
            species_2: Second species label.

        Returns:
            ``True`` if the pair matches in either order.
        """
        forward = fnmatch(species_1, self.species_a) and fnmatch(
            species_2, self.species_b
        )
        reverse = fnmatch(species_1, self.species_b) and fnmatch(
            species_2, self.species_a
        )
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
class Frame:
    """A single snapshot of atomic coordinates.

    Attributes:
        coords: Cartesian coordinates, shape ``(n_atoms, 3)``.
        label: Optional frame label or identifier.
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

    Attributes:
        rotation: 3x3 rotation matrix.
        zoom: Magnification factor.
        centre: 3D point about which to centre the view.
        perspective: Perspective strength (0 = orthographic).
        view_distance: Distance from camera to scene centre.
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


@dataclass
class StructureScene:
    """Top-level scene holding atoms, frames, styles, bond rules, and view.

    Attributes:
        species: One label per atom.
        frames: List of coordinate snapshots.
        atom_styles: Mapping from species label to visual style.
        bond_specs: Declarative bond detection rules.
        view: Camera / projection state.
        title: Scene title for display.
    """

    species: list[str]
    frames: list[Frame]
    atom_styles: dict[str, AtomStyle] = field(default_factory=dict)
    bond_specs: list[BondSpec] = field(default_factory=list)
    view: ViewState = field(default_factory=ViewState)
    title: str = ""

    @classmethod
    def from_xbs(cls, bs_path, mv_path=None):
        """Create from XBS files. See ``hofmann.scene.from_xbs``."""
        from hofmann.scene import from_xbs

        return from_xbs(bs_path, mv_path)

    @classmethod
    def from_pymatgen(cls, structure, bond_specs=None):
        """Create from pymatgen Structure(s). See ``hofmann.scene.from_pymatgen``."""
        from hofmann.scene import from_pymatgen

        return from_pymatgen(structure, bond_specs)

    def render_mpl(self, output=None, **kwargs):
        """Render with matplotlib. See ``hofmann.render_mpl.render_mpl``."""
        from hofmann.render_mpl import render_mpl

        return render_mpl(self, output, **kwargs)

    def render_plotly(self, **kwargs):
        """Render with plotly. See ``hofmann.render_plotly.render_plotly``."""
        from hofmann.render_plotly import render_plotly

        return render_plotly(self, **kwargs)
