"""Tests for the _legend_polyhedra module — vertex registry, faces, rotation, shading."""

import numpy as np
import pytest

from hofmann._constants import VALID_POLYHEDRA
from hofmann.rendering._legend_polyhedra import (
    CANONICAL_VERTICES,
    LEGEND_ROTATION,
    _get_faces,
    shade_face,
)


class TestCanonicalVertices:
    """Tests for the vertex registry."""

    def test_octahedron_shape(self):
        verts = CANONICAL_VERTICES["octahedron"]
        assert verts.shape == (6, 3)

    def test_tetrahedron_shape(self):
        verts = CANONICAL_VERTICES["tetrahedron"]
        assert verts.shape == (4, 3)

    def test_octahedron_unit_radius(self):
        verts = CANONICAL_VERTICES["octahedron"]
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 1.0)

    def test_tetrahedron_unit_radius(self):
        verts = CANONICAL_VERTICES["tetrahedron"]
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 1.0)

    def test_octahedron_centred_at_origin(self):
        verts = CANONICAL_VERTICES["octahedron"]
        np.testing.assert_allclose(verts.mean(axis=0), 0.0, atol=1e-12)

    def test_tetrahedron_centred_at_origin(self):
        verts = CANONICAL_VERTICES["tetrahedron"]
        np.testing.assert_allclose(verts.mean(axis=0), 0.0, atol=1e-12)

    def test_cuboctahedron_shape(self):
        verts = CANONICAL_VERTICES["cuboctahedron"]
        assert verts.shape == (12, 3)

    def test_cuboctahedron_unit_radius(self):
        verts = CANONICAL_VERTICES["cuboctahedron"]
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 1.0)

    def test_cuboctahedron_centred_at_origin(self):
        verts = CANONICAL_VERTICES["cuboctahedron"]
        np.testing.assert_allclose(verts.mean(axis=0), 0.0, atol=1e-12)

    def test_canonical_vertices_matches_valid_polyhedra(self):
        """CANONICAL_VERTICES keys must match the shared VALID_POLYHEDRA constant."""
        assert frozenset(CANONICAL_VERTICES) == VALID_POLYHEDRA


class TestGetFaces:
    """Tests for the face computation and caching."""

    def test_octahedron_face_count(self):
        faces = _get_faces("octahedron")
        assert len(faces) == 8

    def test_tetrahedron_face_count(self):
        faces = _get_faces("tetrahedron")
        assert len(faces) == 4

    def test_octahedron_faces_are_triangles(self):
        faces = _get_faces("octahedron")
        for face in faces:
            assert len(face) == 3

    def test_tetrahedron_faces_are_triangles(self):
        faces = _get_faces("tetrahedron")
        for face in faces:
            assert len(face) == 3

    def test_cuboctahedron_face_count(self):
        faces = _get_faces("cuboctahedron")
        assert len(faces) == 14

    def test_cuboctahedron_face_sizes(self):
        """Cuboctahedron has 8 triangular and 6 square faces."""
        faces = _get_faces("cuboctahedron")
        sizes = sorted(len(f) for f in faces)
        assert sizes == [3] * 8 + [4] * 6

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown polyhedron shape"):
            _get_faces("cube")

    def test_cache_identity(self):
        """Repeated calls return the same object (cached)."""
        a = _get_faces("octahedron")
        b = _get_faces("octahedron")
        assert a is b


class TestLegendRotation:
    """Tests for the fixed rotation matrix."""

    def test_shape(self):
        assert LEGEND_ROTATION.shape == (3, 3)

    def test_orthogonal(self):
        """Rotation matrix should be orthogonal: R @ R^T = I."""
        product = LEGEND_ROTATION @ LEGEND_ROTATION.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_determinant_one(self):
        """Proper rotation has determinant +1."""
        np.testing.assert_allclose(np.linalg.det(LEGEND_ROTATION), 1.0, atol=1e-12)


class TestShadeFace:
    """Tests for the per-face shading helper."""

    _LIGHT_Z = np.array([0.0, 0.0, 1.0])

    def test_viewer_facing_face_brightest(self):
        """A face whose normal points at the viewer (z) gets shading = 1.0."""
        # Triangle in the xy-plane: normal is along z.
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
        rgb = shade_face(verts, (0.8, 0.6, 0.4), polyhedra_shading=1.0, light_direction=self._LIGHT_Z)
        assert rgb == pytest.approx((0.8, 0.6, 0.4))

    def test_edge_on_face_dimmed(self):
        """A face whose normal is perpendicular to z gets maximum dimming."""
        # Triangle in the xz-plane: normal is along y.
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (1.0, 1.0, 1.0), polyhedra_shading=1.0, light_direction=self._LIGHT_Z)
        # shading = 1.0 - 1.0 * 0.6 * (1.0 - 0.0) = 0.4
        assert rgb == pytest.approx((0.4, 0.4, 0.4))

    def test_zero_shading_gives_flat_colour(self):
        """When polyhedra_shading=0, the base colour is returned unchanged."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (0.5, 0.7, 0.3), polyhedra_shading=0.0, light_direction=self._LIGHT_Z)
        assert rgb == pytest.approx((0.5, 0.7, 0.3))

    def test_half_shading(self):
        """Intermediate shading strength produces intermediate colours."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (1.0, 1.0, 1.0), polyhedra_shading=0.5, light_direction=self._LIGHT_Z)
        # shading = 1.0 - 0.5 * 0.6 * (1.0 - 0.0) = 0.7
        assert rgb == pytest.approx((0.7, 0.7, 0.7))


class TestShadeFaceLightDirection:
    """shade_face uses the light_direction parameter."""

    def _make_triangle(self, normal_direction: tuple[float, float, float]) -> np.ndarray:
        """Build a triangle whose face normal points in the given direction.

        Constructs three vertices in a plane perpendicular to
        *normal_direction*.
        """
        n = np.asarray(normal_direction, dtype=float)
        n = n / np.linalg.norm(n)
        # Find two vectors perpendicular to n.
        if abs(n[0]) < 0.9:
            perp1 = np.cross(n, [1, 0, 0])
        else:
            perp1 = np.cross(n, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(n, perp1)
        return np.array([
            [0.0, 0.0, 0.0],
            perp1,
            perp2,
        ])

    def test_face_aligned_with_light_is_brightest(self):
        """A face normal aligned with the light gets maximum brightness."""
        light_dir = np.array([0.0, 0.0, 1.0])
        tri = self._make_triangle((0.0, 0.0, 1.0))
        base = (0.8, 0.6, 0.4)
        shaded = shade_face(tri, base, 1.0, light_dir)
        # cos_angle = 1.0 → shading = 1.0 → output == base
        assert shaded == pytest.approx(base)

    def test_face_perpendicular_to_light_is_dimmest(self):
        """A face normal perpendicular to the light gets minimum brightness."""
        light_dir = np.array([0.0, 0.0, 1.0])
        tri = self._make_triangle((1.0, 0.0, 0.0))
        base = (0.8, 0.6, 0.4)
        shaded = shade_face(tri, base, 1.0, light_dir)
        # cos_angle = 0.0 → shading = 0.4 → output = base * 0.4
        expected = tuple(c * 0.4 for c in base)
        assert shaded == pytest.approx(expected)

    def test_upper_left_light_shades_differently_from_default_z(self):
        """A non-z-axis light gives different shading for z-facing faces."""
        light_dir_z = np.array([0.0, 0.0, 1.0])
        light_dir_ul = np.array([-1.0, 1.0, 1.0])
        light_dir_ul = light_dir_ul / np.linalg.norm(light_dir_ul)
        tri = self._make_triangle((0.0, 0.0, 1.0))
        base = (0.8, 0.6, 0.4)
        shaded_z = shade_face(tri, base, 1.0, light_dir_z)
        shaded_ul = shade_face(tri, base, 1.0, light_dir_ul)
        # z-facing face fully lit by z-light, partially lit by upper-left light
        assert shaded_z != pytest.approx(shaded_ul)

    def test_flat_shading_ignores_light_direction(self):
        """With polyhedra_shading=0, light direction has no effect."""
        light_a = np.array([0.0, 0.0, 1.0])
        light_b = np.array([1.0, 0.0, 0.0])
        tri = self._make_triangle((0.0, 1.0, 0.0))
        base = (0.8, 0.6, 0.4)
        shaded_a = shade_face(tri, base, 0.0, light_a)
        shaded_b = shade_face(tri, base, 0.0, light_b)
        assert shaded_a == pytest.approx(base)
        assert shaded_b == pytest.approx(base)
