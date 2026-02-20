from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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

    def __post_init__(self) -> None:
        if self.zoom <= 0:
            raise ValueError(f"zoom must be positive, got {self.zoom}")
        if self.view_distance <= 0:
            raise ValueError(
                f"view_distance must be positive, got {self.view_distance}"
            )

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
    ) -> ViewState:
        """Set the rotation so the camera looks along *direction*.

        The view is oriented so that *direction* points into the screen
        (along +z in camera space).  The *up* vector determines which
        way is "up" on screen.

        This is equivalent to placing the camera at a point along
        *direction* looking back towards the origin.

        Returns ``self`` so callers can chain, e.g.::

            scene.view = ViewState(centre=centroid).look_along([1, 1, 1])

        Args:
            direction: 3D vector giving the viewing direction (from
                the camera towards the scene).  Need not be normalised.
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
