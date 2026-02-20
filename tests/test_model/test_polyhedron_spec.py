"""Tests for PolyhedronSpec, PolyhedronSpec validation, and Polyhedron."""

import numpy as np
import pytest

from hofmann.model.polyhedron_spec import Polyhedron, PolyhedronSpec


class TestPolyhedronSpec:
    def test_defaults(self):
        spec = PolyhedronSpec(centre="Ti")
        assert spec.centre == "Ti"
        assert spec.colour is None
        assert spec.alpha == 0.4
        assert spec.edge_colour == (0.15, 0.15, 0.15)
        assert spec.edge_width == 1.0
        assert spec.hide_centre is False
        assert spec.hide_bonds is False
        assert spec.hide_vertices is False
        assert spec.min_vertices is None

    def test_custom_values(self):
        spec = PolyhedronSpec(
            centre="Si",
            colour=(0.2, 0.4, 0.8),
            alpha=0.6,
            edge_colour="black",
            edge_width=1.5,
            hide_centre=True,
            hide_bonds=True,
            hide_vertices=True,
            min_vertices=6,
        )
        assert spec.centre == "Si"
        assert spec.colour == (0.2, 0.4, 0.8)
        assert spec.alpha == 0.6
        assert spec.hide_centre is True
        assert spec.hide_bonds is True
        assert spec.hide_vertices is True
        assert spec.min_vertices == 6


class TestPolyhedronSpecValidation:
    def test_alpha_below_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            PolyhedronSpec(centre="Ti", alpha=-0.1)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            PolyhedronSpec(centre="Ti", alpha=1.5)

    def test_negative_edge_width_raises(self):
        with pytest.raises(ValueError, match="edge_width"):
            PolyhedronSpec(centre="Ti", edge_width=-1.0)

    def test_min_vertices_below_three_raises(self):
        with pytest.raises(ValueError, match="min_vertices"):
            PolyhedronSpec(centre="Ti", min_vertices=2)

    def test_valid_spec_accepted(self):
        spec = PolyhedronSpec(centre="Ti", alpha=0.5, edge_width=1.0, min_vertices=4)
        assert spec.centre == "Ti"


class TestPolyhedron:
    def test_is_frozen(self):
        faces = [np.array([0, 1, 2])]
        spec = PolyhedronSpec(centre="Ti")
        poly = Polyhedron(
            centre_index=0,
            neighbour_indices=(1, 2, 3),
            faces=faces,
            spec=spec,
        )
        with pytest.raises(AttributeError):
            poly.centre_index = 5  # type: ignore[misc]

    def test_fields(self):
        faces = [np.array([0, 1, 2]), np.array([0, 2, 3])]
        spec = PolyhedronSpec(centre="Ti")
        poly = Polyhedron(
            centre_index=0,
            neighbour_indices=(1, 2, 3, 4),
            faces=faces,
            spec=spec,
        )
        assert poly.centre_index == 0
        assert poly.neighbour_indices == (1, 2, 3, 4)
        assert len(poly.faces) == 2
        assert all(len(f) == 3 for f in poly.faces)
        assert poly.spec is spec
