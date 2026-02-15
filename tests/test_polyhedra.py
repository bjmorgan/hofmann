"""Tests for hofmann.polyhedra — convex hull computation."""

import numpy as np
import pytest

from hofmann.model import Bond, BondSpec, Polyhedron, PolyhedronSpec
from hofmann.polyhedra import compute_polyhedra


def _make_bond(i: int, j: int, spec: BondSpec) -> Bond:
    """Create a Bond with a dummy length."""
    return Bond(index_a=i, index_b=j, length=2.0, spec=spec)


# A bond spec that matches everything.
_ANY_SPEC = BondSpec(
    species=("*", "*"), min_length=0.0, max_length=10.0,
    radius=0.1, colour=0.5,
)


class TestComputePolyhedra:
    def test_regular_tetrahedron(self):
        """4 neighbours at tetrahedral positions produce 4 triangular faces."""
        # Centre at index 0, 4 neighbours at indices 1-4.
        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
        specs = [PolyhedronSpec(centre="C")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1
        poly = result[0]
        assert poly.centre_index == 0
        assert set(poly.neighbour_indices) == {1, 2, 3, 4}
        assert len(poly.faces) == 4
        assert all(len(f) == 3 for f in poly.faces)

    def test_regular_octahedron(self):
        """6 neighbours at octahedral positions produce 8 triangular faces."""
        species = ["Ti"] + ["O"] * 6
        coords = np.array([
            [0.0, 0.0, 0.0],   # Ti centre
            [1.0, 0.0, 0.0],   # +x
            [-1.0, 0.0, 0.0],  # -x
            [0.0, 1.0, 0.0],   # +y
            [0.0, -1.0, 0.0],  # -y
            [0.0, 0.0, 1.0],   # +z
            [0.0, 0.0, -1.0],  # -z
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 7)]
        specs = [PolyhedronSpec(centre="Ti")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1
        poly = result[0]
        assert poly.centre_index == 0
        assert len(poly.faces) == 8
        assert all(len(f) == 3 for f in poly.faces)

    def test_three_neighbours_single_face(self):
        """3 neighbours produce a single triangular face."""
        species = ["B", "O", "O", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 4)]
        specs = [PolyhedronSpec(centre="B")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1
        assert len(result[0].faces) == 1
        assert len(result[0].faces[0]) == 3

    def test_two_neighbours_skipped(self):
        """2 neighbours cannot form a polyhedron."""
        species = ["A", "B", "B"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        bonds = [_make_bond(0, 1, _ANY_SPEC), _make_bond(0, 2, _ANY_SPEC)]
        specs = [PolyhedronSpec(centre="A")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 0

    def test_one_neighbour_skipped(self):
        """1 neighbour cannot form a polyhedron."""
        species = ["A", "B"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bonds = [_make_bond(0, 1, _ANY_SPEC)]
        specs = [PolyhedronSpec(centre="A")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 0

    def test_no_bonds_no_polyhedra(self):
        """A centre atom with no bonds produces no polyhedron."""
        species = ["Ti", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        specs = [PolyhedronSpec(centre="Ti")]
        result = compute_polyhedra(species, coords, [], specs)
        assert len(result) == 0

    def test_no_specs_no_polyhedra(self):
        """Empty polyhedra_specs produces no polyhedra."""
        species = ["Ti", "O"]
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bonds = [_make_bond(0, 1, _ANY_SPEC)]
        result = compute_polyhedra(species, coords, bonds, [])
        assert len(result) == 0

    def test_wildcard_centre(self):
        """Wildcard centre pattern matches multiple species."""
        species = ["Ti", "O", "O", "O", "O",
                   "Si", "O", "O", "O", "O"]
        coords = np.zeros((10, 3))
        # Give each centre 4 neighbours for tetrahedra.
        coords[1] = [1, 1, 1]
        coords[2] = [-1, -1, 1]
        coords[3] = [1, -1, -1]
        coords[4] = [-1, 1, -1]
        coords[6] = [6, 1, 1]
        coords[7] = [4, -1, 1]
        coords[8] = [6, -1, -1]
        coords[9] = [4, 1, -1]
        coords[5] = [5, 0, 0]
        bonds = (
            [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
            + [_make_bond(5, i, _ANY_SPEC) for i in range(6, 10)]
        )
        specs = [PolyhedronSpec(centre="*")]
        result = compute_polyhedra(species, coords, bonds, specs)
        # Both Ti and Si should get polyhedra.
        assert len(result) == 2

    def test_first_spec_wins(self):
        """First matching PolyhedronSpec takes precedence."""
        species = ["Ti", "O", "O", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 4)]
        spec_a = PolyhedronSpec(centre="Ti", alpha=0.3)
        spec_b = PolyhedronSpec(centre="*", alpha=0.8)
        result = compute_polyhedra(species, coords, bonds, [spec_a, spec_b])
        assert len(result) == 1
        assert result[0].spec is spec_a

    def test_face_indices_are_local(self):
        """Face indices should index into neighbour_indices, not global atoms."""
        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
        specs = [PolyhedronSpec(centre="C")]
        result = compute_polyhedra(species, coords, bonds, specs)
        poly = result[0]
        # All face indices should be in range [0, len(neighbour_indices)).
        assert min(f.min() for f in poly.faces) >= 0
        assert max(f.max() for f in poly.faces) < len(poly.neighbour_indices)

    def test_min_vertices_skips_small_polyhedron(self):
        """min_vertices=5 should skip a 4-neighbour tetrahedron."""
        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
        specs = [PolyhedronSpec(centre="C", min_vertices=5)]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 0

    def test_min_vertices_allows_sufficient_neighbours(self):
        """min_vertices=4 should allow a 4-neighbour tetrahedron."""
        species = ["C", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
        specs = [PolyhedronSpec(centre="C", min_vertices=4)]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1

    def test_cube_merges_to_six_faces(self):
        """8 vertices of a cube should produce 6 quadrilateral faces."""
        species = ["A"] + ["B"] * 8
        coords = np.array([
            [0.0, 0.0, 0.0],  # centre (unused for hull)
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ], dtype=float)
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 9)]
        specs = [PolyhedronSpec(centre="A")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1
        poly = result[0]
        # 12 triangles from ConvexHull should merge to 6 quads.
        assert len(poly.faces) == 6
        assert all(len(f) == 4 for f in poly.faces)

    def test_coplanar_four_neighbours(self):
        """4 coplanar neighbours should produce faces via fallback."""
        species = ["A", "B", "B", "B", "B"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        bonds = [_make_bond(0, i, _ANY_SPEC) for i in range(1, 5)]
        specs = [PolyhedronSpec(centre="A")]
        result = compute_polyhedra(species, coords, bonds, specs)
        assert len(result) == 1
        # Should have triangulated the planar polygon into faces.
        assert len(result[0].faces) >= 1
