"""Core data model for hofmann: dataclasses, colour handling, and projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
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


@dataclass
class RenderStyle:
    """Visual style settings for rendering.

    Groups all appearance parameters that control how a scene is drawn,
    independent of the scene data itself.  A default ``RenderStyle()``
    gives the standard ball-and-stick look.

    Pass a style to :func:`~hofmann.render_mpl.render_mpl` via the
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
            atom circles.  Higher values give smoother circles in
            vector output (PDF/SVG).  ``24`` is fine for screen;
            ``72`` is recommended for publication.
        arc_segments: Number of line segments per semicircular bond
            end-cap.  Higher values give smoother bond ends in vector
            output.  ``5`` is fine for screen; ``12`` is recommended
            for publication.
    """

    atom_scale: float = 0.5
    bond_scale: float = 1.0
    bond_colour: Colour | None = None
    half_bonds: bool = True
    show_bonds: bool = True
    show_polyhedra: bool = True
    show_outlines: bool = True
    outline_colour: Colour = (0.15, 0.15, 0.15)
    atom_outline_width: float = 1.5
    bond_outline_width: float = 1.0
    slab_clip_mode: SlabClipMode = SlabClipMode.PER_FACE
    circle_segments: int = 24
    arc_segments: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.slab_clip_mode, str):
            self.slab_clip_mode = SlabClipMode(self.slab_clip_mode)


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

    The *species* pair is stored in sorted order so that the data
    structure is invariant under exchange of the two labels.

    Species names support fnmatch-style wildcards (``*``, ``?``).

    Attributes:
        species: Sorted pair of species patterns.
        min_length: Minimum bond length threshold.
        max_length: Maximum bond length threshold.
        radius: Visual radius of the bond cylinder.
        colour: Bond colour used when ``half_bonds`` is disabled on
            the render style.  When ``half_bonds`` is ``True`` (the
            default), each half of the bond is coloured to match the
            nearest atom and this field is ignored.
    """

    species: tuple[str, str]
    min_length: float
    max_length: float
    radius: float
    colour: Colour

    def __post_init__(self) -> None:
        self.species = tuple(sorted(self.species))  # type: ignore[assignment]

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
class PolyhedronSpec:
    """Declarative rule for rendering coordination polyhedra.

    A polyhedron is drawn around each atom whose species matches
    *centre*, using its bonded neighbours as vertices of a convex
    hull.  Species names support fnmatch-style wildcards.

    Attributes:
        centre: Species pattern for the centre atom (e.g. ``"Ti"``).
        colour: Face colour, or ``None`` to inherit from the centre
            atom's style colour.
        alpha: Face transparency (0 = fully transparent, 1 = opaque).
        edge_colour: Colour for face wireframe edges.
        edge_width: Line width for face wireframe edges (points).
        hide_centre: Whether to hide the centre atom circle when
            a polyhedron is drawn.
        hide_bonds: Whether to hide bonds from the centre atom to
            its coordinating neighbours when a polyhedron is drawn.
        hide_vertices: Whether to hide the vertex atom circles.
            An atom is only hidden if *every* polyhedron it
            participates in has ``hide_vertices=True``.
        min_vertices: Minimum number of bonded neighbours required
            to draw a polyhedron.  Centre atoms with fewer neighbours
            are skipped.  ``None`` uses the default minimum of 3.
    """

    centre: str
    colour: Colour | None = None
    alpha: float = 0.4
    edge_colour: Colour = (0.15, 0.15, 0.15)
    edge_width: float = 0.8
    hide_centre: bool = False
    hide_bonds: bool = False
    hide_vertices: bool = False
    min_vertices: int | None = None


@dataclass(frozen=True)
class Polyhedron:
    """A computed coordination polyhedron.

    Attributes:
        centre_index: Index of the centre atom.
        neighbour_indices: Indices of the coordinating atoms.
        faces: List of faces, each a 1-D array of vertex indices
            into *neighbour_indices*.  Triangular faces have length 3;
            merged coplanar faces may have 4 or more vertices.
        spec: The PolyhedronSpec that produced this polyhedron.
    """

    centre_index: int
    neighbour_indices: tuple[int, ...]
    faces: list[np.ndarray]
    spec: PolyhedronSpec


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

    Depth-slab clipping is controlled by :attr:`slab_near`,
    :attr:`slab_far`, and :attr:`slab_origin`.  When set, only atoms
    whose depth (along the viewing direction) falls within the range
    ``[origin_depth + slab_near, origin_depth + slab_far]`` are
    rendered.  If *slab_origin* is ``None``, the slab is centred on
    :attr:`centre`.

    Attributes:
        rotation: 3x3 rotation matrix.
        zoom: Magnification factor.
        centre: 3D point about which to centre the view.
        perspective: Perspective strength (0 = orthographic).
        view_distance: Distance from camera to scene centre.
        slab_origin: 3D point defining the slab reference depth, or
            ``None`` to use *centre*.
        slab_near: Near offset from the slab origin depth (negative =
            further from camera), or ``None`` for no near limit.
        slab_far: Far offset from the slab origin depth (positive =
            closer to camera), or ``None`` for no far limit.
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
    slab_origin: np.ndarray | None = None
    slab_near: float | None = None
    slab_far: float | None = None

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

    def slab_mask(self, coords: np.ndarray) -> np.ndarray:
        """Return a boolean mask selecting atoms within the depth slab.

        If neither :attr:`slab_near` nor :attr:`slab_far` is set, all
        atoms are selected.  The depth of each atom is measured along
        the current viewing direction, relative to the slab origin
        (or :attr:`centre` if no origin is set).

        Args:
            coords: World-space coordinates, shape ``(n, 3)``.

        Returns:
            Boolean array of shape ``(n,)``.
        """
        if self.slab_near is None and self.slab_far is None:
            return np.ones(len(coords), dtype=bool)

        coords = np.asarray(coords, dtype=float)
        centred = coords - self.centre
        # Depth is the z-component in camera space.
        depth = centred @ self.rotation[2]

        # Compute the reference depth from slab_origin.
        if self.slab_origin is not None:
            origin_centred = np.asarray(self.slab_origin, dtype=float) - self.centre
            ref_depth = np.dot(origin_centred, self.rotation[2])
        else:
            ref_depth = 0.0

        relative_depth = depth - ref_depth

        mask = np.ones(len(coords), dtype=bool)
        if self.slab_near is not None:
            mask &= relative_depth >= self.slab_near
        if self.slab_far is not None:
            mask &= relative_depth <= self.slab_far
        return mask

    def look_along(
        self,
        direction: np.ndarray | list[float] | tuple[float, ...],
        *,
        up: np.ndarray | list[float] | tuple[float, ...] = (0.0, 1.0, 0.0),
    ) -> None:
        """Set the rotation so the camera looks along *direction*.

        The view is oriented so that *direction* points into the screen
        (along +z in camera space).  The *up* vector determines which
        way is "up" on screen.

        This is equivalent to placing the camera at a point along
        *direction* looking back towards the origin.

        Args:
            direction: 3D vector giving the viewing direction (from
                the camera towards the scene).  Need not be normalised.
            up: 3D vector indicating the upward direction in screen
                space.  Defaults to ``[0, 1, 0]``.

        Raises:
            ValueError: If *direction* is zero-length or *up* is
                parallel to *direction*.
        """
        d = np.asarray(direction, dtype=float)
        u = np.asarray(up, dtype=float)

        d_len = np.linalg.norm(d)
        if d_len < 1e-12:
            raise ValueError("direction must be non-zero")
        fwd = d / d_len                     # camera z-axis (into screen)

        right = np.cross(u, fwd)
        right_len = np.linalg.norm(right)
        if right_len < 1e-12:
            # Up is parallel to direction.  If the caller explicitly
            # provided an up vector, that is an error.  Otherwise
            # fall back to [0, 0, 1] as the up hint.
            default_up = (0.0, 1.0, 0.0)
            if tuple(float(x) for x in up) != default_up:
                raise ValueError(
                    "up vector is parallel to the viewing direction"
                )
            u = np.array([0.0, 0.0, 1.0])
            right = np.cross(u, fwd)
            right_len = np.linalg.norm(right)
        right /= right_len                  # camera x-axis

        up_actual = np.cross(fwd, right)     # camera y-axis

        # Rotation matrix: rows are the camera basis vectors.
        # R maps world coords to camera coords: rotated = R @ world.
        self.rotation = np.array([right, up_actual, fwd])


@dataclass
class StructureScene:
    """Top-level scene holding atoms, frames, styles, bond rules, and view.

    Attributes:
        species: One label per atom.
        frames: List of coordinate snapshots.
        atom_styles: Mapping from species label to visual style.
        bond_specs: Declarative bond detection rules.
        polyhedra: Declarative polyhedron rendering rules.
        view: Camera / projection state.
        title: Scene title for display.
        n_unit_cell_atoms: Number of atoms in the original unit cell
            before PBC expansion.  When set, only atoms with index
            below this value are considered as polyhedron centres.
            ``None`` means all atoms are candidates.
    """

    species: list[str]
    frames: list[Frame]
    atom_styles: dict[str, AtomStyle] = field(default_factory=dict)
    bond_specs: list[BondSpec] = field(default_factory=list)
    polyhedra: list[PolyhedronSpec] = field(default_factory=list)
    view: ViewState = field(default_factory=ViewState)
    title: str = ""
    n_unit_cell_atoms: int | None = None

    @classmethod
    def from_xbs(cls, bs_path, mv_path=None):
        """Create from XBS files. See ``hofmann.scene.from_xbs``."""
        from hofmann.scene import from_xbs

        return from_xbs(bs_path, mv_path)

    @classmethod
    def from_pymatgen(
        cls, structure, bond_specs=None, *, polyhedra=None,
        pbc=False, pbc_cutoff=None, centre_atom=None,
    ):
        """Create from pymatgen Structure(s). See ``hofmann.scene.from_pymatgen``."""
        from hofmann.scene import from_pymatgen

        return from_pymatgen(
            structure, bond_specs, polyhedra=polyhedra,
            pbc=pbc, pbc_cutoff=pbc_cutoff, centre_atom=centre_atom,
        )

    def centre_on(self, atom_index: int, *, frame: int = 0) -> None:
        """Centre the view on a specific atom.

        Sets :attr:`view.centre` to the Cartesian position of the atom
        at *atom_index* in the given frame.

        Args:
            atom_index: Index of the atom to centre on.
            frame: Frame index to read coordinates from.
        """
        self.view.centre = self.frames[frame].coords[atom_index].copy()

    def render_mpl(self, output=None, **kwargs):
        """Render with matplotlib. See ``hofmann.render_mpl.render_mpl``."""
        from hofmann.render_mpl import render_mpl

        return render_mpl(self, output, **kwargs)

    def render_mpl_interactive(self, **kwargs) -> ViewState:
        """Interactive matplotlib viewer. See ``hofmann.render_mpl.render_mpl_interactive``."""
        from hofmann.render_mpl import render_mpl_interactive

        return render_mpl_interactive(self, **kwargs)
