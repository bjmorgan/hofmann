"""Tests for style serialisation and file I/O."""

import json

import numpy as np
import pytest

from hofmann.model import (
    AtomStyle,
    AxesStyle,
    BondSpec,
    CellEdgeStyle,
    Frame,
    LegendItem,
    LegendStyle,
    PolyhedronSpec,
    RenderStyle,
    SlabClipMode,
    StructureScene,
    WidgetCorner,
    normalise_colour,
)
from hofmann.construction.styles import StyleSet, load_styles, save_styles


# -- AtomStyle ---------------------------------------------------------------

class TestAtomStyleDict:
    def test_round_trip(self):
        style = AtomStyle(radius=1.4, colour=(0.5, 1.0, 0.5))
        d = style.to_dict()
        restored = AtomStyle.from_dict(d)
        assert restored.radius == style.radius
        assert normalise_colour(restored.colour) == normalise_colour(style.colour)
        assert restored.visible is True

    def test_visible_false_included(self):
        style = AtomStyle(radius=1.0, colour="red", visible=False)
        d = style.to_dict()
        assert d["visible"] is False
        restored = AtomStyle.from_dict(d)
        assert restored.visible is False

    def test_visible_true_omitted(self):
        style = AtomStyle(radius=1.0, colour="blue")
        d = style.to_dict()
        assert "visible" not in d

    def test_colour_normalised_to_list(self):
        style = AtomStyle(radius=1.0, colour="red")
        d = style.to_dict()
        assert isinstance(d["colour"], list)
        assert len(d["colour"]) == 3

    def test_from_dict_accepts_colour_name(self):
        d = {"radius": 1.0, "colour": "green"}
        style = AtomStyle.from_dict(d)
        assert style.colour == "green"


# -- BondSpec -----------------------------------------------------------------

class TestBondSpecDict:
    def test_round_trip_minimal(self):
        spec = BondSpec(species=("Ti", "O"), max_length=2.5)
        d = spec.to_dict()
        restored = BondSpec.from_dict(d)
        assert restored.species == spec.species
        assert restored.max_length == spec.max_length
        assert restored.min_length == 0.0
        assert restored._radius is None
        assert restored._colour is None

    def test_round_trip_full(self):
        spec = BondSpec(
            species=("Na", "Cl"), max_length=3.0, min_length=0.5,
            radius=0.12, colour=(0.4, 0.4, 0.4),
            complete="*", recursive=True,
        )
        d = spec.to_dict()
        restored = BondSpec.from_dict(d)
        assert restored.species == spec.species
        assert restored.max_length == spec.max_length
        assert restored.min_length == 0.5
        assert restored._radius == 0.12
        assert normalise_colour(restored.colour) == normalise_colour(spec.colour)
        assert restored.complete == "*"
        assert restored.recursive is True

    def test_defaults_omitted(self):
        spec = BondSpec(species=("C", "H"), max_length=1.2)
        d = spec.to_dict()
        assert "min_length" not in d
        assert "radius" not in d
        assert "colour" not in d
        assert "complete" not in d
        assert "recursive" not in d

    def test_explicit_radius_preserved(self):
        spec = BondSpec(species=("C", "H"), max_length=1.2, radius=0.15)
        d = spec.to_dict()
        assert d["radius"] == 0.15

    def test_none_radius_not_serialised(self):
        """Specs using the class default should not serialise radius."""
        spec = BondSpec(species=("C", "H"), max_length=1.2)
        d = spec.to_dict()
        assert "radius" not in d
        restored = BondSpec.from_dict(d)
        assert restored._radius is None
        assert restored.radius == BondSpec.default_radius

    def test_from_dict_sorts_species(self):
        """from_dict should sort the species pair alphabetically."""
        d = {"species": ["Ti", "O"], "max_length": 2.5}
        restored = BondSpec.from_dict(d)
        assert restored.species == ("O", "Ti")


# -- PolyhedronSpec -----------------------------------------------------------

class TestPolyhedronSpecDict:
    def test_round_trip_minimal(self):
        spec = PolyhedronSpec(centre="Ti")
        d = spec.to_dict()
        assert d == {"centre": "Ti"}
        restored = PolyhedronSpec.from_dict(d)
        assert restored.centre == "Ti"
        assert restored.colour is None
        assert restored.alpha == 0.4

    def test_round_trip_full(self):
        spec = PolyhedronSpec(
            centre="Ti", colour=(0.2, 0.4, 0.9), alpha=0.6,
            edge_colour="black", edge_width=2.0,
            hide_centre=True, hide_bonds=True,
            hide_vertices=True, min_vertices=4,
        )
        d = spec.to_dict()
        restored = PolyhedronSpec.from_dict(d)
        assert restored.centre == "Ti"
        assert normalise_colour(restored.colour) == normalise_colour(spec.colour)
        assert restored.alpha == 0.6
        assert restored.hide_centre is True
        assert restored.min_vertices == 4

    def test_defaults_omitted(self):
        spec = PolyhedronSpec(centre="O")
        d = spec.to_dict()
        assert set(d.keys()) == {"centre"}


# -- CellEdgeStyle ------------------------------------------------------------

class TestCellEdgeStyleDict:
    def test_default_is_empty(self):
        style = CellEdgeStyle()
        assert style.to_dict() == {}

    def test_round_trip(self):
        style = CellEdgeStyle(colour="red", line_width=2.0, linestyle="dashed")
        d = style.to_dict()
        restored = CellEdgeStyle.from_dict(d)
        assert normalise_colour(restored.colour) == normalise_colour("red")
        assert restored.line_width == 2.0
        assert restored.linestyle == "dashed"


# -- AxesStyle ----------------------------------------------------------------

class TestAxesStyleDict:
    def test_default_is_empty(self):
        style = AxesStyle()
        assert style.to_dict() == {}

    def test_round_trip(self):
        style = AxesStyle(
            colours=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            labels=("x", "y", "z"),
            font_size=12.0,
            corner=WidgetCorner.TOP_RIGHT,
        )
        d = style.to_dict()
        restored = AxesStyle.from_dict(d)
        assert restored.labels == ("x", "y", "z")
        assert restored.font_size == 12.0
        assert restored.corner == WidgetCorner.TOP_RIGHT

    def test_explicit_corner_tuple(self):
        style = AxesStyle(corner=(0.1, 0.9))
        d = style.to_dict()
        assert d["corner"] == [0.1, 0.9]
        restored = AxesStyle.from_dict(d)
        assert restored.corner == (0.1, 0.9)


# -- LegendItem ---------------------------------------------------------------

class TestLegendItemDict:
    def test_minimal_round_trip(self):
        item = LegendItem(key="Na", colour="blue")
        d = item.to_dict()
        restored = LegendItem.from_dict(d)
        assert restored.key == "Na"
        assert restored.colour == normalise_colour("blue")

    def test_full_round_trip(self):
        item = LegendItem(key="Na", colour=(0.2, 0.4, 0.8), label="Sodium", radius=6.0)
        d = item.to_dict()
        restored = LegendItem.from_dict(d)
        assert restored.key == "Na"
        assert restored.colour == (0.2, 0.4, 0.8)
        assert restored.label == "Sodium"
        assert restored.radius == 6.0

    def test_label_none_omitted(self):
        item = LegendItem(key="Na", colour="blue")
        d = item.to_dict()
        assert "label" not in d

    def test_radius_none_omitted(self):
        item = LegendItem(key="Na", colour="blue")
        d = item.to_dict()
        assert "radius" not in d

    def test_colour_normalised_in_dict(self):
        item = LegendItem(key="Na", colour="blue")
        d = item.to_dict()
        assert d["colour"] == list(normalise_colour("blue"))


# -- LegendStyle --------------------------------------------------------------

class TestLegendStyleDict:
    def test_default_is_empty(self):
        style = LegendStyle()
        assert style.to_dict() == {}

    def test_round_trip(self):
        style = LegendStyle(
            corner=WidgetCorner.TOP_LEFT,
            font_size=12.0,
            circle_radius=8.0,
            spacing=4.0,
        )
        d = style.to_dict()
        restored = LegendStyle.from_dict(d)
        assert restored.corner is WidgetCorner.TOP_LEFT
        assert restored.font_size == 12.0
        assert restored.circle_radius == 8.0
        assert restored.spacing == 4.0

    def test_species_round_trip(self):
        style = LegendStyle(species=("Na", "Cl"))
        d = style.to_dict()
        assert d["species"] == ["Na", "Cl"]
        restored = LegendStyle.from_dict(d)
        assert restored.species == ("Na", "Cl")

    def test_explicit_corner_tuple(self):
        style = LegendStyle(corner=(0.1, 0.9))
        d = style.to_dict()
        assert d["corner"] == [0.1, 0.9]
        restored = LegendStyle.from_dict(d)
        assert restored.corner == (0.1, 0.9)

    def test_species_none_omitted(self):
        style = LegendStyle()
        d = style.to_dict()
        assert "species" not in d

    def test_circle_radius_range_round_trip(self):
        style = LegendStyle(circle_radius=(3.0, 8.0))
        d = style.to_dict()
        assert d["circle_radius"] == [3.0, 8.0]
        restored = LegendStyle.from_dict(d)
        assert restored.circle_radius == (3.0, 8.0)

    def test_circle_radius_dict_round_trip(self):
        style = LegendStyle(circle_radius={"Na": 5.0, "Cl": 8.0})
        d = style.to_dict()
        assert d["circle_radius"] == {"Na": 5.0, "Cl": 8.0}
        restored = LegendStyle.from_dict(d)
        assert restored.circle_radius == {"Na": 5.0, "Cl": 8.0}

    def test_circle_radius_float_default_omitted(self):
        style = LegendStyle(circle_radius=5.0)
        d = style.to_dict()
        assert "circle_radius" not in d

    def test_circle_radius_float_non_default_included(self):
        style = LegendStyle(circle_radius=7.0)
        d = style.to_dict()
        assert d["circle_radius"] == 7.0

    def test_label_gap_round_trip(self):
        style = LegendStyle(label_gap=8.0)
        d = style.to_dict()
        assert d["label_gap"] == 8.0
        restored = LegendStyle.from_dict(d)
        assert restored.label_gap == 8.0

    def test_label_gap_default_omitted(self):
        style = LegendStyle()
        d = style.to_dict()
        assert "label_gap" not in d

    def test_labels_round_trip(self):
        labels = {"Ti": "$\\mathrm{Ti^{4+}}$", "O": "oxygen"}
        style = LegendStyle(labels=labels)
        d = style.to_dict()
        assert d["labels"] == labels
        restored = LegendStyle.from_dict(d)
        assert restored.labels == labels

    def test_labels_none_omitted(self):
        style = LegendStyle()
        d = style.to_dict()
        assert "labels" not in d


# -- RenderStyle --------------------------------------------------------------

class TestRenderStyleDict:
    def test_default_is_empty(self):
        style = RenderStyle()
        assert style.to_dict() == {}

    def test_round_trip(self):
        style = RenderStyle(
            atom_scale=0.8,
            show_outlines=False,
            half_bonds=False,
            bond_colour="blue",
            slab_clip_mode=SlabClipMode.CLIP_WHOLE,
        )
        d = style.to_dict()
        restored = RenderStyle.from_dict(d)
        assert restored.atom_scale == 0.8
        assert restored.show_outlines is False
        assert restored.half_bonds is False
        assert normalise_colour(restored.bond_colour) == normalise_colour("blue")
        assert restored.slab_clip_mode == SlabClipMode.CLIP_WHOLE

    def test_nested_cell_style(self):
        style = RenderStyle(
            cell_style=CellEdgeStyle(linestyle="dashed"),
        )
        d = style.to_dict()
        assert "cell_style" in d
        assert d["cell_style"]["linestyle"] == "dashed"
        restored = RenderStyle.from_dict(d)
        assert restored.cell_style.linestyle == "dashed"

    def test_nested_axes_style(self):
        style = RenderStyle(
            axes_style=AxesStyle(font_size=14.0),
        )
        d = style.to_dict()
        assert "axes_style" in d
        restored = RenderStyle.from_dict(d)
        assert restored.axes_style.font_size == 14.0

    def test_nested_legend_style(self):
        style = RenderStyle(
            legend_style=LegendStyle(font_size=14.0),
        )
        d = style.to_dict()
        assert "legend_style" in d
        restored = RenderStyle.from_dict(d)
        assert restored.legend_style.font_size == 14.0

    def test_show_legend_round_trip(self):
        style = RenderStyle(show_legend=True)
        d = style.to_dict()
        assert d["show_legend"] is True
        restored = RenderStyle.from_dict(d)
        assert restored.show_legend is True

    def test_defaults_omitted(self):
        style = RenderStyle()
        d = style.to_dict()
        assert "atom_scale" not in d
        assert "cell_style" not in d
        assert "axes_style" not in d
        assert "legend_style" not in d
        assert "show_legend" not in d
        assert "pbc" not in d
        assert "pbc_padding" not in d
        assert "max_recursive_depth" not in d
        assert "deduplicate_molecules" not in d

    def test_pbc_fields_round_trip(self):
        style = RenderStyle(
            pbc=False, pbc_padding=0.5,
            max_recursive_depth=2, deduplicate_molecules=True,
        )
        d = style.to_dict()
        assert d["pbc"] is False
        assert d["pbc_padding"] == 0.5
        assert d["max_recursive_depth"] == 2
        assert d["deduplicate_molecules"] is True
        restored = RenderStyle.from_dict(d)
        assert restored.pbc is False
        assert restored.pbc_padding == 0.5
        assert restored.max_recursive_depth == 2
        assert restored.deduplicate_molecules is True

    def test_pbc_padding_none_round_trip(self):
        style = RenderStyle(pbc_padding=None)
        d = style.to_dict()
        assert d["pbc_padding"] is None
        restored = RenderStyle.from_dict(d)
        assert restored.pbc_padding is None


# -- save_styles / load_styles ------------------------------------------------

class TestSaveLoadStyles:
    def test_round_trip_all_sections(self, tmp_path):
        path = tmp_path / "styles.json"
        atom_styles = {
            "Ti": AtomStyle(radius=1.0, colour=(0.2, 0.4, 0.9)),
            "O": AtomStyle(radius=0.8, colour="red"),
        }
        bond_specs = [
            BondSpec(species=("Ti", "O"), max_length=2.5, radius=0.12),
        ]
        polyhedra = [PolyhedronSpec(centre="Ti")]
        render_style = RenderStyle(atom_scale=0.8)

        save_styles(
            path,
            atom_styles=atom_styles,
            bond_specs=bond_specs,
            polyhedra=polyhedra,
            render_style=render_style,
        )

        result = load_styles(path)
        assert result.atom_styles is not None
        assert result.atom_styles["Ti"].radius == 1.0
        assert result.atom_styles["O"].radius == 0.8
        assert result.bond_specs is not None
        assert len(result.bond_specs) == 1
        assert result.bond_specs[0].species == ("O", "Ti")
        assert result.polyhedra is not None
        assert len(result.polyhedra) == 1
        assert result.render_style is not None
        assert result.render_style.atom_scale == 0.8

    def test_partial_atom_styles_only(self, tmp_path):
        path = tmp_path / "atoms.json"
        save_styles(
            path,
            atom_styles={"Na": AtomStyle(radius=1.5, colour="green")},
        )
        result = load_styles(path)
        assert result.atom_styles is not None
        assert result.bond_specs is None
        assert result.polyhedra is None
        assert result.render_style is None

    def test_partial_render_style_only(self, tmp_path):
        path = tmp_path / "render.json"
        save_styles(path, render_style=RenderStyle(show_bonds=False))
        result = load_styles(path)
        assert result.atom_styles is None
        assert result.render_style is not None
        assert result.render_style.show_bonds is False

    def test_unknown_keys_raise(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"atom_styles": {}, "unknown_key": 42}')
        with pytest.raises(ValueError, match="unknown top-level keys"):
            load_styles(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}")
        result = load_styles(path)
        assert result.atom_styles is None
        assert result.bond_specs is None

    def test_json_is_human_readable(self, tmp_path):
        path = tmp_path / "pretty.json"
        save_styles(
            path,
            atom_styles={"Na": AtomStyle(radius=1.0, colour="green")},
        )
        text = path.read_text()
        # Should be indented, not a single line.
        assert "\n" in text
        assert "  " in text


# -- StructureScene convenience methods ---------------------------------------

class TestSceneStyleMethods:
    def _make_scene(self):
        return StructureScene(
            species=["Na", "Cl"],
            frames=[Frame(coords=np.array([[0, 0, 0], [1, 1, 1]], dtype=float))],
            atom_styles={
                "Na": AtomStyle(radius=1.0, colour=(0.3, 0.3, 0.8)),
                "Cl": AtomStyle(radius=1.2, colour=(0.1, 0.8, 0.1)),
            },
            bond_specs=[
                BondSpec(species=("Na", "Cl"), max_length=3.0),
            ],
        )

    def test_save_and_load_round_trip(self, tmp_path):
        scene = self._make_scene()
        path = tmp_path / "scene_styles.json"
        scene.save_styles(path)

        # Create a new scene with different styles.
        scene2 = StructureScene(
            species=["Na", "Cl"],
            frames=[Frame(coords=np.array([[0, 0, 0], [2, 2, 2]], dtype=float))],
            atom_styles={
                "Na": AtomStyle(radius=0.5, colour="grey"),
                "Cl": AtomStyle(radius=0.5, colour="grey"),
            },
        )
        scene2.load_styles(path)

        # Na style should be overridden from the file.
        assert scene2.atom_styles["Na"].radius == 1.0
        # Bond specs should be replaced.
        assert len(scene2.bond_specs) == 1

    def test_load_merges_atom_styles(self, tmp_path):
        """Loading atom styles merges rather than replacing."""
        path = tmp_path / "partial.json"
        # File only has Na style.
        save_styles(
            path,
            atom_styles={"Na": AtomStyle(radius=2.0, colour="red")},
        )
        scene = self._make_scene()
        original_cl = scene.atom_styles["Cl"]
        scene.load_styles(path)
        # Na overridden, Cl unchanged.
        assert scene.atom_styles["Na"].radius == 2.0
        assert scene.atom_styles["Cl"] is original_cl
