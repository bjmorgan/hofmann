from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from hofmann.model.projection import Orthographic, Perspective, Projection


_DEFAULT_PERSPECTIVE = Perspective()


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
        projection: Projection mode, :class:`Orthographic` (the
            default) or :class:`Perspective`.
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
    projection: Projection = field(
        default_factory=Orthographic
    )
    slab_origin: np.ndarray | None = None
    slab_near: float | None = None
    slab_far: float | None = None

    def __post_init__(self) -> None:
        if self.zoom <= 0:
            raise ValueError(f"zoom must be positive, got {self.zoom}")

    def project(
        self, coords: np.ndarray, radii: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D coordinates to 2D with depth information.

        Under :class:`Perspective` the eye sits on the camera's +z
        axis and each sphere's visible silhouette is projected onto
        the z=0 plane.

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
        xy = self.project_camera(rotated)

        if radii is None:
            return xy, depth, np.zeros(len(depth))

        radii = np.asarray(radii, dtype=float)
        silhouette = self.projection.silhouette_radius(depth, radii)
        return xy, depth, silhouette * self.zoom

    def project_camera(self, camera: np.ndarray) -> np.ndarray:
        """Map camera-space positions to screen positions.

        The single camera-to-screen mapping for scene geometry: atoms,
        bonds, and cell edges all obtain their screen positions here.

        Args:
            camera: Array of shape ``(n, 3)``, already centred and
                rotated into camera space.

        Returns:
            *xy* of shape ``(n, 2)`` — screen positions with zoom applied.
        """
        camera = np.asarray(camera, dtype=float)
        if self.projection.reaches_eye_plane(camera[:, 2]):
            warnings.warn(
                "one or more points lie at or behind the perspective "
                "eye plane; they are drawn mirrored through the origin "
                "and sorted as if nearest the viewer.  Increase "
                "view_distance or reduce strength.",
                UserWarning,
                stacklevel=2,
            )
        return self.projection.to_screen(camera) * self.zoom

    def screen_frame(self, camera: np.ndarray) -> np.ndarray:
        """Camera coordinates with the screen shear applied to xy, depth kept.

        Bond junction geometry is resolved in this frame, so tangent
        offsets agree with the screen circles they are drawn against.  An
        exact passthrough (``== camera``) unless the projection shears
        (:class:`Oblique`); zoom is not applied (unlike
        :meth:`project_camera`).

        Args:
            camera: Array of shape ``(n, 3)``, already centred and
                rotated into camera space.

        Returns:
            Array of shape ``(n, 3)``: the sheared screen ``xy`` with the
            camera depth in the third column.
        """
        camera = np.asarray(camera, dtype=float)
        xy = camera @ self.projection.screen_matrix.T
        return np.column_stack([xy, camera[:, 2]])

    def set_orthographic(self) -> ViewState:
        """Draw without perspective foreshortening.

        Returns:
            ``self``, so the call can be chained with
            :meth:`look_along`.
        """
        self.projection = Orthographic()
        return self

    def set_perspective(
        self,
        strength: float = _DEFAULT_PERSPECTIVE.strength,
        view_distance: float = _DEFAULT_PERSPECTIVE.view_distance,
    ) -> ViewState:
        """Draw with perspective foreshortening.

        Args:
            strength: Perspective strength.  ``1.0`` is a true pinhole
                camera at *view_distance*; smaller values move the eye
                further out and weaken the effect.
            view_distance: Reference distance from the scene centre.

        Returns:
            ``self``, so the call can be chained with
            :meth:`look_along`.
        """
        self.projection = Perspective(strength, view_distance)
        return self

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
        """Set the rotation to view along the *direction* axis.

        The camera sits on the ``+direction`` side, looking back towards
        the origin, so *direction* points out of the screen towards the
        viewer (the camera's +z axis).  The *up* vector determines which
        way is "up" on screen.

        Returns ``self`` so callers can chain, e.g.::

            scene.view = ViewState(centre=centroid).look_along([1, 1, 1])

        Args:
            direction: 3D vector giving the axis to view along; the
                camera is placed on the ``+direction`` side, looking
                back towards the origin.  Need not be normalised.
            up: 3D vector indicating the upward direction in screen
                space.  Defaults to ``[0, 1, 0]``.

        Returns:
            ``self``, with the rotation updated in place.

        Raises:
            ValueError: If *direction* or *up* is zero or has a
                non-finite length, or a caller-supplied *up* is parallel
                to *direction*.
        """
        d = np.asarray(direction, dtype=float)
        u = np.asarray(up, dtype=float)

        # Norms overflow to inf for a huge vector; the finiteness checks
        # below reject that (and NaN, and zero) in place of a bare numpy
        # warning or a silent degenerate rotation.
        with np.errstate(over="ignore"):
            d_len = np.linalg.norm(d)
            u_len = np.linalg.norm(u)
        if not np.isfinite(d_len) or d_len < 1e-12:
            raise ValueError("direction must be finite and non-zero")
        if not np.isfinite(u_len) or u_len < 1e-12:
            raise ValueError("up must be finite and non-zero")
        fwd = d / d_len                     # camera z-axis (out of screen)

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
