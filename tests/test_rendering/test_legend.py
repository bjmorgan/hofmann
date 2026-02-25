"""Tests for the legend module — label formatting and item building."""

import numpy as np

from hofmann.model import (
    AtomStyle,
    Frame,
    LegendItem,
    LegendStyle,
    StructureScene,
    _DEFAULT_CIRCLE_RADIUS,
    normalise_colour,
)


def _minimal_scene():
    """Create a minimal two-species scene for legend tests."""
    return StructureScene(
        species=["A", "B"],
        frames=[Frame(coords=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]))],
        atom_styles={
            "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
            "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
        },
    )


class TestFormatLegendLabel:
    """Tests for automatic chemical notation formatting."""

    def test_plain_text_unchanged(self):
        from hofmann.rendering.legend import _format_legend_label
        assert _format_legend_label("Sr") == "Sr"

    def test_charge_superscript(self):
        from hofmann.rendering.legend import _format_legend_label
        assert _format_legend_label("Sr2+") == r"Sr$^{2\!+}$"
        assert _format_legend_label("O2-") == r"O$^{2\!-}$"
        assert _format_legend_label("Fe3+") == r"Fe$^{3\!+}$"

    def test_subscript(self):
        from hofmann.rendering.legend import _format_legend_label
        assert _format_legend_label("TiO6") == "TiO$_{6}$"
        assert _format_legend_label("H2O") == "H$_{2}$O"

    def test_explicit_mathtext_unchanged(self):
        from hofmann.rendering.legend import _format_legend_label
        raw = r"Sr$^{2\!+}$"
        assert _format_legend_label(raw) == raw


class TestBuildLegendItems:
    """Tests for the _build_legend_items helper."""

    def _build(self, scene, style=None):
        from hofmann.rendering.legend import _build_legend_items
        return _build_legend_items(scene, style or LegendStyle())

    def test_auto_detect_species(self):
        """Items follow unique species in first-seen order."""
        scene = StructureScene(
            species=["B", "A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.8, (0.8, 0.2, 0.2)),
            },
        )
        items = self._build(scene)
        assert [item.key for item in items] == ["B", "A"]
        assert normalise_colour(items[0].colour) == normalise_colour((0.8, 0.2, 0.2))
        assert normalise_colour(items[1].colour) == normalise_colour((0.5, 0.5, 0.5))

    def test_explicit_species(self):
        """Explicit species list controls inclusion and order."""
        scene = _minimal_scene()
        items = self._build(scene, LegendStyle(species=("B",)))
        assert [item.key for item in items] == ["B"]

    def test_invisible_species_filtered(self):
        """Auto-detect excludes species with visible=False."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(0.8, (0.8, 0.2, 0.2), visible=False),
            },
        )
        items = self._build(scene)
        assert [item.key for item in items] == ["A"]

    def test_custom_labels(self):
        """Labels from style are attached to items."""
        scene = _minimal_scene()
        items = self._build(scene, LegendStyle(labels={"A": "Alpha"}))
        a_item = next(item for item in items if item.key == "A")
        b_item = next(item for item in items if item.key == "B")
        assert a_item.label == "Alpha"
        assert a_item.display_label == "Alpha"
        assert b_item.label is None
        assert b_item.display_label == "B"

    def test_proportional_radius(self):
        """Proportional circle_radius sets different radii on items."""
        scene = StructureScene(
            species=["A", "B"],
            frames=[Frame(coords=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]))],
            atom_styles={
                "A": AtomStyle(1.0, (0.5, 0.5, 0.5)),
                "B": AtomStyle(2.0, (0.8, 0.2, 0.2)),
            },
        )
        items = self._build(scene, LegendStyle(circle_radius=(3.0, 9.0)))
        a_item = next(item for item in items if item.key == "A")
        b_item = next(item for item in items if item.key == "B")
        assert a_item.radius == 3.0
        assert b_item.radius == 9.0

    def test_uniform_radius_is_none(self):
        """Uniform circle_radius leaves item radii as None."""
        scene = _minimal_scene()
        items = self._build(scene, LegendStyle(circle_radius=7.0))
        assert all(item.radius is None for item in items)

    def test_dict_radius(self):
        """Dict circle_radius sets per-species radii on items."""
        scene = _minimal_scene()
        items = self._build(
            scene, LegendStyle(circle_radius={"A": 4.0, "B": 8.0}),
        )
        a_item = next(item for item in items if item.key == "A")
        b_item = next(item for item in items if item.key == "B")
        assert a_item.radius == 4.0
        assert b_item.radius == 8.0

    def test_unknown_species_gets_grey(self):
        """Species not in atom_styles get grey fallback colour."""
        scene = StructureScene(
            species=["X"],
            frames=[Frame(coords=np.array([[0.0, 0.0, 0.0]]))],
            atom_styles={},
        )
        items = self._build(scene)
        assert len(items) == 1
        assert normalise_colour(items[0].colour) == (0.5, 0.5, 0.5)

    def test_empty_scene_returns_empty(self):
        """Scene with no species returns an empty list."""
        scene = StructureScene(
            species=[],
            frames=[Frame(coords=np.zeros((0, 3)))],
            atom_styles={},
        )
        items = self._build(scene)
        assert items == []


class TestResolveItemRadius:
    """Tests for _resolve_item_radius — polyhedron vs flat defaults."""

    def _resolve(self, item, style):
        from hofmann.rendering.legend import _resolve_item_radius
        return _resolve_item_radius(item, style)

    def test_explicit_radius_wins(self):
        item = LegendItem(key="a", colour="red", radius=7.0)
        assert self._resolve(item, LegendStyle()) == 7.0

    def test_explicit_radius_wins_for_polyhedron(self):
        item = LegendItem(
            key="a", colour="red", polyhedron="octahedron", radius=20.0,
        )
        assert self._resolve(item, LegendStyle()) == 20.0

    def test_flat_item_uses_circle_radius(self):
        item = LegendItem(key="a", colour="red")
        assert self._resolve(item, LegendStyle(circle_radius=8.0)) == 8.0

    def test_polyhedron_defaults_to_twice_circle_radius(self):
        """Polyhedra default to 2x the flat-marker radius."""
        item = LegendItem(key="a", colour="red", polyhedron="octahedron")
        assert self._resolve(item, LegendStyle()) == 2.0 * _DEFAULT_CIRCLE_RADIUS

    def test_polyhedron_scales_with_circle_radius(self):
        """Polyhedra scale to 2x when circle_radius is set."""
        item = LegendItem(key="a", colour="red", polyhedron="tetrahedron")
        assert self._resolve(item, LegendStyle(circle_radius=3.0)) == 6.0
