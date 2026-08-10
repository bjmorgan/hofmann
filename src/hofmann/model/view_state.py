from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Oblique:
    """Direction and foreshortening of an oblique projection's receding axis.

    An oblique parallel projection draws two axes undistorted in the
    plane of the page and the third receding at an angle,
    foreshortened.

    Attributes:
        angle: On-screen direction of the receding axis, in degrees
            anticlockwise from screen +x.
        foreshortening: Scale factor applied to the receding axis
            (cavalier 1.0, cabinet 0.5).  Zero recovers the
            orthographic projection exactly.  Negative values are
            equivalent to ``angle + 180``.
    """

    angle: float = 45.0
    foreshortening: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.angle):
            raise ValueError(f"angle must be finite, got {self.angle}")
        if not math.isfinite(self.foreshortening):
            raise ValueError(
                f"foreshortening must be finite, got {self.foreshortening}"
            )


#: Cavalier projection: receding axis at 45 degrees, full length.
CAVALIER = Oblique(45.0, 1.0)

#: Cabinet projection: receding axis at 45 degrees, half length.
CABINET = Oblique(45.0, 0.5)


def _check_oblique_perspective_exclusive(
    oblique: Oblique | None, perspective: float,
) -> None:
    """Raise if an oblique projection is combined with perspective."""
    if oblique is not None and perspective > 0:
        raise ValueError(
            "oblique and perspective projections are mutually exclusive"
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
        oblique: Oblique projection parameters, or ``None`` for the
            standard orthographic / perspective projection.  Mutually
            exclusive with *perspective*.
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
    oblique: Oblique | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.zoom) or self.zoom <= 0:
            raise ValueError(
                f"zoom must be finite and positive, got {self.zoom}"
            )
        if not math.isfinite(self.view_distance) or self.view_distance <= 0:
            raise ValueError(
                f"view_distance must be finite and positive, got "
                f"{self.view_distance}"
            )
        if not math.isfinite(self.perspective):
            raise ValueError(
                f"perspective must be finite, got {self.perspective}"
            )
        _check_oblique_perspective_exclusive(self.oblique, self.perspective)

    @property
    def screen_matrix(self) -> np.ndarray:
        """The ``(2, 3)`` linear map from camera space to screen space.

        Identity on x and y when :attr:`oblique` is ``None``.  With an
        oblique projection the third column displaces screen positions
        in proportion to depth, so that the receding axis is drawn at
        :attr:`Oblique.angle` with length scaled by
        :attr:`Oblique.foreshortening`.  Zoom-free and
        perspective-free, so it can also map bare direction vectors,
        as the axes orientation widget requires.
        """
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        if self.oblique is not None:
            th = np.radians(self.oblique.angle)
            f = self.oblique.foreshortening
            m[0, 2] = -f * np.cos(th)
            m[1, 2] = -f * np.sin(th)
        return m

    @property
    def screen_scale_bound(self) -> float:
        """Largest factor by which :attr:`screen_matrix` can stretch a vector.

        The largest singular value of the screen matrix:
        ``sqrt(1 + f**2)`` for foreshortening ``f``, exactly ``1.0``
        when :attr:`oblique` is ``None``.  Used for viewport sizing.
        """
        if self.oblique is None:
            return 1.0
        f = self.oblique.foreshortening
        return float(np.sqrt(1.0 + f * f))

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
            factor at each depth (all ones when orthographic).

        Raises:
            ValueError: If :attr:`oblique` is set while
                :attr:`perspective` is positive.  Construction and
                :meth:`with_oblique` also reject this combination;
                this backstop closes the direct-assignment path.
        """
        _check_oblique_perspective_exclusive(self.oblique, self.perspective)
        camera = np.asarray(camera, dtype=float)
        xy = camera @ self.screen_matrix.T
        if self.perspective > 0:
            scale = self.view_distance / (
                self.view_distance - camera[:, 2] * self.perspective
            )
        else:
            scale = np.ones(len(camera))
        return xy * scale[:, np.newaxis] * self.zoom, scale

    def project(
        self, coords: np.ndarray, radii: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D coordinates to 2D with depth information.

        The eye sits at ``[0, 0, view_distance]`` and each sphere's
        visible silhouette is projected onto the z=0 plane.

        With an oblique projection the screen coordinates gain a
        depth-proportional offset; *depth* and *projected_radii* are
        unchanged — spheres keep their circular outlines by drawing
        convention.

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
            if self.perspective > 0:
                # Recomputed directly (not view_distance / scale): the
                # division round-trip is not bit-exact and output must
                # be byte-identical to the pre-oblique implementation.
                # Eye-to-atom distance along z.
                d = self.view_distance - depth * self.perspective
                # Silhouette radius: r * D / sqrt(d^2 - r^2).
                denom = np.sqrt(np.maximum(d**2 - radii**2, 1e-12))
                projected_radii = radii * self.view_distance / denom * self.zoom
            else:
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
        return self

    def with_oblique(self, oblique: Oblique = CABINET) -> ViewState:
        """Enable an oblique projection and return ``self`` for chaining.

        Mirrors :meth:`look_along`::

            scene.view = ViewState().look_along([0, -1, 0]).with_oblique(CAVALIER)

        To return to an orthographic projection, assign
        ``view.oblique = None`` directly; this method deliberately
        does not accept ``None``.

        Args:
            oblique: Oblique projection parameters.  Defaults to
                :data:`CABINET`.

        Returns:
            ``self``, with :attr:`oblique` set.

        Raises:
            ValueError: If *perspective* is positive — oblique and
                perspective projections are mutually exclusive.
        """
        _check_oblique_perspective_exclusive(oblique, self.perspective)
        self.oblique = oblique
        return self
