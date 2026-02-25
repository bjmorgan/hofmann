"""Tests for the _legend_polyhedra module — vertex registry, faces, rotation, shading."""

import numpy as np
import pytest

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

    def test_viewer_facing_face_brightest(self):
        """A face whose normal points at the viewer (z) gets shading = 1.0."""
        # Triangle in the xy-plane: normal is along z.
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)
        rgb = shade_face(verts, (0.8, 0.6, 0.4), polyhedra_shading=1.0)
        assert rgb == pytest.approx((0.8, 0.6, 0.4))

    def test_edge_on_face_dimmed(self):
        """A face whose normal is perpendicular to z gets maximum dimming."""
        # Triangle in the xz-plane: normal is along y.
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (1.0, 1.0, 1.0), polyhedra_shading=1.0)
        # shading = 1.0 - 1.0 * 0.6 * (1.0 - 0.0) = 0.4
        assert rgb == pytest.approx((0.4, 0.4, 0.4))

    def test_zero_shading_gives_flat_colour(self):
        """When polyhedra_shading=0, the base colour is returned unchanged."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (0.5, 0.7, 0.3), polyhedra_shading=0.0)
        assert rgb == pytest.approx((0.5, 0.7, 0.3))

    def test_half_shading(self):
        """Intermediate shading strength produces intermediate colours."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        rgb = shade_face(verts, (1.0, 1.0, 1.0), polyhedra_shading=0.5)
        # shading = 1.0 - 0.5 * 0.6 * (1.0 - 0.0) = 0.7
        assert rgb == pytest.approx((0.7, 0.7, 0.7))
