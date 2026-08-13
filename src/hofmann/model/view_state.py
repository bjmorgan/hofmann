from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Orthographic:
    """Parallel projection with no foreshortening of depth."""


@dataclass(frozen=True, slots=True)
class Perspective:
    """Perspective projection parameters.

    Attributes:
        strength: Perspective strength.  ``1.0`` is a true pinhole
            camera at *view_distance*; smaller values weaken the
            convergence (equivalent to a pinhole eye at
            ``view_distance / strength``).  Must be finite and
            positive; zero strength is not representable —
            orthographic projection is :class:`Orthographic`.
        view_distance: Distance from camera to scene centre.  Must be
            finite and positive.
    """

    strength: float = 0.5
    view_distance: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError(f"strength must be finite, got {self.strength}")
        if self.strength <= 0:
            raise ValueError(
                f"strength must be positive, got {self.strength}"
            )
        if not math.isfinite(self.view_distance):
            raise ValueError(
                f"view_distance must be finite, got {self.view_distance}"
            )
        if self.view_distance <= 0:
            raise ValueError(
                f"view_distance must be positive, got {self.view_distance}"
            )


@dataclass(frozen=True, slots=True)
class Oblique:
    """Direction and foreshortening of an oblique projection's receding axis.

    An oblique parallel projection draws two axes undistorted in the
    plane of the page and the third receding at an angle,
    foreshortened.

    Attributes:
        angle: On-screen direction of the receding axis, in degrees
            anticlockwise from screen +x.
        foreshortening: Scale factor applied to the receding axis.
            Zero recovers the orthographic projection exactly.
            Negative values are equivalent to ``angle + 180``.
    """

    angle: float
    foreshortening: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.angle):
            raise ValueError(f"angle must be finite, got {self.angle}")
        if not math.isfinite(self.foreshortening):
            raise ValueError(
                f"foreshortening must be finite, got {self.foreshortening}"
            )


@dataclass(slots=True)
class ViewState:
    """Camera state for 3D-to-2D projection.

    Encapsulates rotation, zoom, centring, and projection mode.
    Renderers consume the projected 2D coordinates and depth values
    produced by :meth:`project`.

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
        projection: Projection mode: :class:`Orthographic` (default),
            :class:`Perspective`, or :class:`Oblique`.
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
    projection: Orthographic | Perspective | Oblique = field(
        default_factory=Orthographic
    )
    slab_origin: np.ndarray | None = None
    slab_near: float | None = None
    slab_far: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.zoom) or self.zoom <= 0:
            raise ValueError(
                f"zoom must be finite and positive, got {self.zoom}"
            )
        if not isinstance(
            self.projection, Orthographic | Perspective | Oblique
        ):
            raise TypeError(
                "projection must be Orthographic, Perspective, or Oblique, "
                f"got {type(self.projection).__name__}"
            )

    @property
    def screen_matrix(self) -> np.ndarray:
        """The ``(2, 3)`` linear map from camera space to screen space.

        Identity on x and y unless :attr:`projection` is
        :class:`Oblique`, in which case the third column displaces
        screen positions in proportion to depth, so that the receding
        axis is drawn at :attr:`Oblique.angle` with length scaled by
        :attr:`Oblique.foreshortening`.  Zoom-free and
        perspective-free, so it can also map bare direction vectors,
        as the axes orientation widget requires.
        """
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        match self.projection:
            case Oblique() as ob:
                th = np.radians(ob.angle)
                m[0, 2] = -ob.foreshortening * np.cos(th)
                m[1, 2] = -ob.foreshortening * np.sin(th)
        return m

    @property
    def screen_scale_bound(self) -> float:
        """Largest factor by which :attr:`screen_matrix` can stretch a vector.

        The largest singular value of the screen matrix:
        ``sqrt(1 + f**2)`` for foreshortening ``f`` when
        :attr:`projection` is :class:`Oblique`, exactly ``1.0``
        otherwise.  Used for viewport sizing.
        """
        match self.projection:
            case Oblique() as ob:
                return float(np.sqrt(1.0 + ob.foreshortening**2))
            case _:
                return 1.0

    def project_camera(
        self, camera: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map camera-space coordinates to screen coordinates.

        The single source of truth for the camera-space-to-screen
        mapping: applies :attr:`screen_matrix`, then perspective
        scaling, then zoom.  Every rendered object must obtain its
        screen positions through this method (directly or via
        :meth:`project`) so that all drawn geometry projects
        consistently.

        Args:
            camera: Array of shape ``(n, 3)`` in camera space, i.e.
                after centring and rotation.

        Returns:
            Tuple of ``(xy, scale)`` where *xy* has shape ``(n, 2)``
            and *scale* has shape ``(n,)``, the perspective scale
            factor at each depth (all ones for the parallel
            projections, :class:`Orthographic` and :class:`Oblique`).

        Raises:
            ValueError: If *camera* does not have shape ``(n, 3)`` —
                a single ``(3,)`` point would otherwise broadcast into
                silently duplicated rows.

        Warns:
            UserWarning: Under :class:`Perspective`, if any point lies
                at or behind the eye plane, where the projection
                formula is singular or sign-reversing; the returned
                *scale* is still computed (inf or negative) rather
                than clamped.
        """
        camera = np.asarray(camera, dtype=float)
        if camera.ndim != 2 or camera.shape[1] != 3:
            raise ValueError(
                f"camera must have shape (n, 3), got {camera.shape}"
            )
        xy = camera @ self.screen_matrix.T
        match self.projection:
            case Orthographic() | Oblique():
                scale = np.ones(len(camera))
            case Perspective() as p:
                denom = p.view_distance - camera[:, 2] * p.strength
                if np.any(denom <= 0):
                    warnings.warn(
                        "one or more points lie at or behind the "
                        "perspective eye plane and will be drawn "
                        f"unreliably (view_distance={p.view_distance:.3g}, "
                        f"strength={p.strength:.3g}).  Increase "
                        "view_distance or reduce strength.",
                        UserWarning,
                        stacklevel=2,
                    )
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = p.view_distance / denom
            case _:
                raise TypeError(
                    "projection must be Orthographic, Perspective, or "
                    f"Oblique, got {type(self.projection).__name__}"
                )
        return xy * scale[:, np.newaxis] * self.zoom, scale

    def screen_frame(self, camera: np.ndarray) -> np.ndarray:
        """Map camera-space coordinates to the screen-aligned frame.

        The 3D frame in which this view's parallel projection is a
        plain drop of z: x and y are the pre-zoom screen coordinates
        (:attr:`screen_matrix` applied) and z is the unchanged depth.
        Geometry that must agree with drawn screen positions — bond
        junction offsets against atom silhouettes, for example — must
        be computed in this frame: under an oblique projection the
        camera frame's z axis is not the projection ray, but this
        frame's z axis is.

        An exact passthrough whenever :attr:`projection` is not
        :class:`Oblique`.

        Args:
            camera: Array of shape ``(n, 3)`` in camera space.

        Returns:
            Array of shape ``(n, 3)``.

        Raises:
            ValueError: If *camera* does not have shape ``(n, 3)``.
        """
        camera = np.asarray(camera, dtype=float)
        if camera.ndim != 2 or camera.shape[1] != 3:
            raise ValueError(
                f"camera must have shape (n, 3), got {camera.shape}"
            )
        return np.column_stack(
            [camera @ self.screen_matrix.T, camera[:, 2]]
        )

    def project(
        self, coords: np.ndarray, radii: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D coordinates to 2D with depth information.

        Under a :class:`Perspective` projection, screen positions are
        scaled by ``view_distance / (view_distance - depth * strength)``
        and each sphere's visible silhouette is projected onto the
        z = 0 plane.  At ``strength = 1`` this is a pinhole eye at
        ``[0, 0, view_distance]``; smaller strengths weaken the
        convergence, equivalent to moving the eye out to
        ``view_distance / strength``.

        With an oblique projection the screen coordinates gain a
        depth-proportional offset; *depth* and *projected_radii* are
        unchanged.

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

        xy, _ = self.project_camera(rotated)

        if radii is not None:
            radii = np.asarray(radii, dtype=float)
            match self.projection:
                case Perspective() as p:
                    # Recomputed directly (not view_distance / scale): the
                    # division round-trip is not bit-exact and output must
                    # be byte-identical to the two-field implementation.
                    # Eye-to-atom distance along z.
                    d = p.view_distance - depth * p.strength
                    # Silhouette radius: r * D / sqrt(d^2 - r^2).
                    denom = np.sqrt(np.maximum(d**2 - radii**2, 1e-12))
                    projected_radii = (
                        radii * p.view_distance / denom * self.zoom
                    )
                case _:
                    projected_radii = radii * self.zoom
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
    ) -> ViewState:
        """Set the rotation so the camera views the scene from *direction*.

        The camera is placed at a point along *direction*, looking
        back towards the origin: *direction* maps to +z in camera
        space and points out of the screen, towards the viewer.  The
        *up* vector determines which way is "up" on screen.

        Returns ``self`` so callers can chain, e.g.::

            scene.view = ViewState(centre=centroid).look_along([1, 1, 1])

        Args:
            direction: 3D vector from the scene towards the camera.
                Need not be normalised.
            up: 3D vector indicating the upward direction in screen
                space.  Defaults to ``[0, 1, 0]``.

        Returns:
            ``self``, with the rotation updated in place.

        Raises:
            ValueError: If *direction* is zero-length or *up* is
                parallel to *direction*.
        """
        d = np.asarray(direction, dtype=float)
        u = np.asarray(up, dtype=float)

        d_len = np.linalg.norm(d)
        if d_len < 1e-12:
            raise ValueError("direction must be non-zero")
        fwd = d / d_len                     # camera z-axis (towards the viewer)

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
        return self
