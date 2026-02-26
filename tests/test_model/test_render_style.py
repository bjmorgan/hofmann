"""Tests for RenderStyle, SlabClipMode, CellEdgeStyle, WidgetCorner, AxesStyle, LegendItem subclasses, and LegendStyle."""

import numpy as np
import pytest

from hofmann.model.render_style import (
    AtomLegendItem,
    AxesStyle,
    CellEdgeStyle,
    LegendItem,
    LegendStyle,
    PolygonLegendItem,
    PolyhedronLegendItem,
    RenderStyle,
    SlabClipMode,
    WidgetCorner,
)


class TestRenderStyle:
    def test_slab_clip_mode_default(self):
        style = RenderStyle()
        assert style.slab_clip_mode == SlabClipMode.PER_FACE

    def test_slab_clip_mode_string_coercion(self):
        style = RenderStyle(slab_clip_mode="include_whole")
        assert style.slab_clip_mode is SlabClipMode.INCLUDE_WHOLE

    def test_slab_clip_mode_enum_value(self):
        style = RenderStyle(slab_clip_mode=SlabClipMode.CLIP_WHOLE)
        assert style.slab_clip_mode is SlabClipMode.CLIP_WHOLE

    def test_polyhedra_outline_width_default_none(self):
        style = RenderStyle()
        assert style.polyhedra_outline_width is None

    def test_polyhedra_outline_width_override(self):
        style = RenderStyle(polyhedra_outline_width=2.0)
        assert style.polyhedra_outline_width == 2.0

    @pytest.mark.parametrize("field, value", [
        ("atom_scale", 0),
        ("atom_scale", -1.0),
        ("bond_scale", 0),
        ("bond_scale", -0.5),
    ])
    def test_positive_scale_required(self, field, value):
        with pytest.raises(ValueError, match="must be positive"):
            RenderStyle(**{field: value})

    @pytest.mark.parametrize("field, value", [
        ("atom_outline_width", -1.0),
        ("bond_outline_width", -0.1),
        ("polyhedra_outline_width", -0.5),
    ])
    def test_non_negative_width_required(self, field, value):
        with pytest.raises(ValueError, match="must be non-negative"):
            RenderStyle(**{field: value})

    def test_circle_segments_default(self):
        assert RenderStyle().circle_segments == 72

    def test_arc_segments_default(self):
        assert RenderStyle().arc_segments == 12

    def test_circle_segments_minimum(self):
        with pytest.raises(ValueError, match="circle_segments must be >= 3"):
            RenderStyle(circle_segments=2)

    def test_arc_segments_minimum(self):
        with pytest.raises(ValueError, match="arc_segments must be >= 2"):
            RenderStyle(arc_segments=1)

    def test_interactive_circle_segments_default(self):
        assert RenderStyle().interactive_circle_segments == 24

    def test_interactive_arc_segments_default(self):
        assert RenderStyle().interactive_arc_segments == 5

    def test_interactive_circle_segments_minimum(self):
        with pytest.raises(ValueError, match="interactive_circle_segments must be >= 3"):
            RenderStyle(interactive_circle_segments=2)

    def test_interactive_arc_segments_minimum(self):
        with pytest.raises(ValueError, match="interactive_arc_segments must be >= 2"):
            RenderStyle(interactive_arc_segments=1)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_polyhedra_shading_range(self, value):
        with pytest.raises(ValueError, match="polyhedra_shading must be between"):
            RenderStyle(polyhedra_shading=value)

    def test_valid_boundary_values_accepted(self):
        """Edge values at the boundaries should not raise."""
        RenderStyle(
            atom_scale=0.01,
            bond_scale=0.01,
            atom_outline_width=0.0,
            bond_outline_width=0.0,
            polyhedra_shading=0.0,
            polyhedra_outline_width=0.0,
        )
        RenderStyle(polyhedra_shading=1.0)

    def test_show_cell_default_none(self):
        style = RenderStyle()
        assert style.show_cell is None

    def test_cell_style_default(self):
        style = RenderStyle()
        assert isinstance(style.cell_style, CellEdgeStyle)
        assert style.cell_style.linestyle == "solid"

    def test_cell_style_override(self):
        cs = CellEdgeStyle(colour="red", line_width=2.0)
        style = RenderStyle(cell_style=cs)
        assert style.cell_style.colour == "red"

    def test_show_axes_default_none(self):
        style = RenderStyle()
        assert style.show_axes is None

    def test_axes_style_default(self):
        style = RenderStyle()
        assert isinstance(style.axes_style, AxesStyle)
        assert style.axes_style.corner is WidgetCorner.BOTTOM_LEFT

    def test_axes_style_override(self):
        ws = AxesStyle(corner="top_right", font_size=14.0)
        style = RenderStyle(axes_style=ws)
        assert style.axes_style.corner is WidgetCorner.TOP_RIGHT
        assert style.axes_style.font_size == 14.0

    def test_show_legend_default_false(self):
        style = RenderStyle()
        assert style.show_legend is False

    def test_legend_style_default(self):
        style = RenderStyle()
        assert isinstance(style.legend_style, LegendStyle)
        assert style.legend_style.corner is WidgetCorner.BOTTOM_RIGHT

    def test_legend_style_override(self):
        ls = LegendStyle(corner="top_left", font_size=14.0)
        style = RenderStyle(legend_style=ls)
        assert style.legend_style.corner is WidgetCorner.TOP_LEFT
        assert style.legend_style.font_size == 14.0

    # ---- PBC rendering fields ----

    def test_pbc_defaults(self):
        style = RenderStyle()
        assert style.pbc is True
        assert style.pbc_padding == 0.1
        assert style.max_recursive_depth == 5
        assert style.deduplicate_molecules is False

    def test_pbc_overrides(self):
        style = RenderStyle(
            pbc=False, pbc_padding=0.5,
            max_recursive_depth=2, deduplicate_molecules=True,
        )
        assert style.pbc is False
        assert style.pbc_padding == 0.5
        assert style.max_recursive_depth == 2
        assert style.deduplicate_molecules is True

    def test_pbc_padding_none_disables(self):
        style = RenderStyle(pbc_padding=None)
        assert style.pbc_padding is None

    def test_max_recursive_depth_minimum(self):
        with pytest.raises(ValueError, match="max_recursive_depth must be >= 1"):
            RenderStyle(max_recursive_depth=0)

    def test_negative_max_recursive_depth_raises(self):
        with pytest.raises(ValueError, match="max_recursive_depth must be >= 1"):
            RenderStyle(max_recursive_depth=-1)

    def test_negative_pbc_padding_raises(self):
        with pytest.raises(ValueError, match="pbc_padding must be non-negative"):
            RenderStyle(pbc_padding=-0.1)


class TestSlabClipMode:
    def test_values(self):
        assert SlabClipMode.PER_FACE == "per_face"
        assert SlabClipMode.CLIP_WHOLE == "clip_whole"
        assert SlabClipMode.INCLUDE_WHOLE == "include_whole"

    def test_string_construction(self):
        assert SlabClipMode("per_face") is SlabClipMode.PER_FACE
        assert SlabClipMode("clip_whole") is SlabClipMode.CLIP_WHOLE
        assert SlabClipMode("include_whole") is SlabClipMode.INCLUDE_WHOLE


class TestCellEdgeStyle:
    def test_defaults(self):
        style = CellEdgeStyle()
        assert style.colour == (0.3, 0.3, 0.3)
        assert style.line_width == 0.8
        assert style.linestyle == "solid"

    def test_custom_values(self):
        style = CellEdgeStyle(colour="blue", line_width=1.5, linestyle="dashed")
        assert style.colour == "blue"
        assert style.line_width == 1.5
        assert style.linestyle == "dashed"

    def test_negative_line_width_raises(self):
        with pytest.raises(ValueError, match="line_width must be non-negative"):
            CellEdgeStyle(line_width=-0.1)

    def test_invalid_linestyle_raises(self):
        with pytest.raises(ValueError, match="linestyle must be one of"):
            CellEdgeStyle(linestyle="wavy")

    def test_is_frozen(self):
        style = CellEdgeStyle()
        with pytest.raises(AttributeError):
            style.line_width = 2.0

    @pytest.mark.parametrize("ls", ["solid", "dashed", "dotted", "dashdot"])
    def test_valid_linestyles(self, ls):
        style = CellEdgeStyle(linestyle=ls)
        assert style.linestyle == ls


class TestWidgetCorner:
    def test_values(self):
        assert WidgetCorner.BOTTOM_LEFT == "bottom_left"
        assert WidgetCorner.BOTTOM_RIGHT == "bottom_right"
        assert WidgetCorner.TOP_LEFT == "top_left"
        assert WidgetCorner.TOP_RIGHT == "top_right"

    def test_string_construction(self):
        assert WidgetCorner("bottom_left") is WidgetCorner.BOTTOM_LEFT
        assert WidgetCorner("top_right") is WidgetCorner.TOP_RIGHT


class TestAxesStyle:
    def test_defaults(self):
        style = AxesStyle()
        assert len(style.colours) == 3
        assert all(c == (0.3, 0.3, 0.3) for c in style.colours)
        assert style.labels == ("a", "b", "c")
        assert style.font_size == 10.0
        assert style.italic is True
        assert style.arrow_length == 0.12
        assert style.line_width == 1.0
        assert style.corner is WidgetCorner.BOTTOM_LEFT
        assert style.margin == 0.15

    def test_custom_values(self):
        style = AxesStyle(
            colours=("red", "green", "blue"),
            labels=("x", "y", "z"),
            font_size=12.0,
            italic=False,
            arrow_length=0.1,
            corner="top_right",
        )
        assert style.colours == ("red", "green", "blue")
        assert style.labels == ("x", "y", "z")
        assert style.font_size == 12.0
        assert style.italic is False
        assert style.arrow_length == 0.1
        assert style.corner is WidgetCorner.TOP_RIGHT

    def test_is_frozen(self):
        style = AxesStyle()
        with pytest.raises(AttributeError):
            style.font_size = 14.0

    def test_corner_string_coercion(self):
        style = AxesStyle(corner="bottom_right")
        assert style.corner is WidgetCorner.BOTTOM_RIGHT

    def test_negative_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            AxesStyle(font_size=-1.0)

    def test_zero_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            AxesStyle(font_size=0)

    def test_negative_arrow_length_raises(self):
        with pytest.raises(ValueError, match="arrow_length must be positive"):
            AxesStyle(arrow_length=-0.1)

    def test_negative_line_width_raises(self):
        with pytest.raises(ValueError, match="line_width must be non-negative"):
            AxesStyle(line_width=-1.0)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin must be non-negative"):
            AxesStyle(margin=-0.1)

    def test_wrong_number_of_colours_raises(self):
        with pytest.raises(ValueError, match="colours must have exactly 3"):
            AxesStyle(colours=((1, 0, 0), (0, 1, 0)))

    def test_wrong_number_of_labels_raises(self):
        with pytest.raises(ValueError, match="labels must have exactly 3"):
            AxesStyle(labels=("a", "b"))

    def test_corner_tuple(self):
        style = AxesStyle(corner=(0.1, 0.9))
        assert style.corner == (0.1, 0.9)

    def test_corner_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="corner tuple must have 2 elements"):
            AxesStyle(corner=(0.1, 0.2, 0.3))


class TestLegendItemBase:
    """Tests for shared LegendItem behaviour (using AtomLegendItem)."""

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            LegendItem(key="Na", colour="blue")

    def test_required_fields(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.key == "Na"
        assert item.colour == (0.0, 0.0, 1.0)
        assert item.label is None
        assert item.radius is None

    def test_all_shared_fields(self):
        item = AtomLegendItem(
            key="Na", colour=(0.2, 0.4, 0.8), label="Sodium", radius=6.0,
        )
        assert item.key == "Na"
        assert item.colour == (0.2, 0.4, 0.8)
        assert item.label == "Sodium"
        assert item.radius == 6.0

    def test_colour_setter_valid(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.colour = "red"
        assert item.colour == (1.0, 0.0, 0.0)

    def test_colour_setter_invalid_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError):
            item.colour = "not_a_colour"

    def test_radius_setter_valid(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.radius = 3.0
        assert item.radius == 3.0

    def test_radius_setter_none(self):
        item = AtomLegendItem(key="Na", colour="blue", radius=5.0)
        item.radius = None
        assert item.radius is None

    def test_radius_setter_negative_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="radius must be positive"):
            item.radius = -1.0

    def test_radius_setter_zero_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="radius must be positive"):
            item.radius = 0.0

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key must be non-empty"):
            AtomLegendItem(key="", colour="blue")

    def test_invalid_colour_raises(self):
        with pytest.raises(ValueError):
            AtomLegendItem(key="Na", colour="not_a_colour")

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            AtomLegendItem(key="Na", colour="blue", radius=-1.0)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            AtomLegendItem(key="Na", colour="blue", radius=0.0)

    def test_display_label_falls_back_to_key(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.display_label == "Na"

    def test_display_label_returns_label(self):
        item = AtomLegendItem(key="Na", colour="blue", label="Sodium")
        assert item.display_label == "Sodium"

    def test_label_is_mutable(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.label = "Sodium"
        assert item.label == "Sodium"
        assert item.display_label == "Sodium"

    def test_not_hashable(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(TypeError):
            hash(item)

    # ---- gap_after ----

    def test_gap_after_defaults_to_none(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.gap_after is None

    def test_gap_after_construction(self):
        item = AtomLegendItem(key="Na", colour="blue", gap_after=10.0)
        assert item.gap_after == 10.0

    def test_gap_after_zero(self):
        item = AtomLegendItem(key="Na", colour="blue", gap_after=0.0)
        assert item.gap_after == 0.0

    def test_gap_after_setter_valid(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.gap_after = 5.0
        assert item.gap_after == 5.0

    def test_gap_after_setter_none(self):
        item = AtomLegendItem(key="Na", colour="blue", gap_after=5.0)
        item.gap_after = None
        assert item.gap_after is None

    def test_gap_after_negative_raises(self):
        with pytest.raises(ValueError, match="gap_after must be non-negative"):
            AtomLegendItem(key="Na", colour="blue", gap_after=-1.0)

    def test_gap_after_setter_negative_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="gap_after must be non-negative"):
            item.gap_after = -0.5

    # ---- alpha ----

    def test_alpha_defaults_to_one(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.alpha == 1.0

    def test_alpha_construction(self):
        item = AtomLegendItem(key="Na", colour="blue", alpha=0.5)
        assert item.alpha == 0.5

    def test_alpha_setter_valid(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.alpha = 0.3
        assert item.alpha == 0.3

    def test_alpha_zero(self):
        item = AtomLegendItem(key="Na", colour="blue", alpha=0.0)
        assert item.alpha == 0.0

    def test_alpha_coerced_to_float(self):
        item = AtomLegendItem(key="Na", colour="blue", alpha=1)
        assert item.alpha == 1.0
        assert isinstance(item.alpha, float)

    def test_alpha_below_zero_raises(self):
        with pytest.raises(ValueError, match="alpha must be between"):
            AtomLegendItem(key="Na", colour="blue", alpha=-0.1)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha must be between"):
            AtomLegendItem(key="Na", colour="blue", alpha=1.1)

    def test_alpha_setter_invalid_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="alpha must be between"):
            item.alpha = -0.5

    # ---- edge_colour / edge_width ----

    def test_edge_colour_default_none(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.edge_colour is None

    def test_edge_colour_construction(self):
        item = AtomLegendItem(key="Na", colour="blue", edge_colour="red")
        assert item.edge_colour == (1.0, 0.0, 0.0)

    def test_edge_colour_setter(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.edge_colour = (0.1, 0.2, 0.3)
        assert item.edge_colour == (0.1, 0.2, 0.3)

    def test_edge_colour_setter_none(self):
        item = AtomLegendItem(key="Na", colour="blue", edge_colour="red")
        item.edge_colour = None
        assert item.edge_colour is None

    def test_edge_width_default_none(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.edge_width is None

    def test_edge_width_construction(self):
        item = AtomLegendItem(key="Na", colour="blue", edge_width=2.0)
        assert item.edge_width == 2.0

    def test_edge_width_setter(self):
        item = AtomLegendItem(key="Na", colour="blue")
        item.edge_width = 1.5
        assert item.edge_width == 1.5

    def test_edge_width_negative_raises(self):
        with pytest.raises(ValueError, match="edge_width must be non-negative"):
            AtomLegendItem(key="Na", colour="blue", edge_width=-1.0)

    def test_edge_width_setter_negative_raises(self):
        item = AtomLegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="edge_width must be non-negative"):
            item.edge_width = -0.5


class TestAtomLegendItem:
    def test_marker_type(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.marker_type == "atom"

    def test_marker(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item.marker == "o"

    def test_repr(self):
        item = AtomLegendItem(key="Na", colour="blue")
        r = repr(item)
        assert r.startswith("AtomLegendItem(")
        assert "key='Na'" in r
        assert "colour=(0.0, 0.0, 1.0)" in r
        assert "label" not in r
        assert "radius" not in r

    def test_repr_full(self):
        item = AtomLegendItem(
            key="Na", colour="blue", label="Sodium", radius=6.0,
        )
        r = repr(item)
        assert "label='Sodium'" in r
        assert "radius=6.0" in r

    def test_equality(self):
        a = AtomLegendItem(key="Na", colour="blue", label="Sodium", radius=6.0)
        b = AtomLegendItem(key="Na", colour="blue", label="Sodium", radius=6.0)
        assert a == b

    def test_equality_normalised_colours(self):
        a = AtomLegendItem(key="Na", colour="blue")
        b = AtomLegendItem(key="Na", colour=(0.0, 0.0, 1.0))
        assert a == b

    def test_inequality_different_colour(self):
        a = AtomLegendItem(key="Na", colour="blue")
        b = AtomLegendItem(key="Na", colour="red")
        assert a != b

    def test_inequality_not_legend_item(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert item != "not an item"

    def test_inequality_different_subclass(self):
        atom = AtomLegendItem(key="X", colour="red")
        poly = PolyhedronLegendItem(key="X", colour="red", shape="octahedron")
        assert atom != poly

    def test_repr_with_gap_after(self):
        item = AtomLegendItem(key="Na", colour="blue", gap_after=10.0)
        assert "gap_after=10.0" in repr(item)

    def test_repr_without_gap_after(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert "gap_after" not in repr(item)

    def test_repr_with_alpha(self):
        item = AtomLegendItem(key="Na", colour="blue", alpha=0.5)
        assert "alpha=0.5" in repr(item)

    def test_repr_without_alpha(self):
        item = AtomLegendItem(key="Na", colour="blue")
        assert "alpha" not in repr(item)

    def test_repr_with_edge_fields(self):
        item = AtomLegendItem(
            key="Na", colour="blue",
            edge_colour="red", edge_width=2.0,
        )
        r = repr(item)
        assert "edge_colour=" in r
        assert "edge_width=" in r

    def test_repr_without_edge_fields(self):
        item = AtomLegendItem(key="Na", colour="blue")
        r = repr(item)
        assert "edge_colour" not in r
        assert "edge_width" not in r

    def test_equality_with_edge_fields(self):
        a = AtomLegendItem(key="Na", colour="blue", edge_colour="red", edge_width=1.0)
        b = AtomLegendItem(key="Na", colour="blue", edge_colour="red", edge_width=1.0)
        assert a == b

    def test_inequality_different_edge_colour(self):
        a = AtomLegendItem(key="Na", colour="blue", edge_colour="red")
        b = AtomLegendItem(key="Na", colour="blue", edge_colour="green")
        assert a != b

    def test_inequality_different_edge_width(self):
        a = AtomLegendItem(key="Na", colour="blue", edge_width=1.0)
        b = AtomLegendItem(key="Na", colour="blue", edge_width=2.0)
        assert a != b

    def test_equality_with_gap_after(self):
        a = AtomLegendItem(key="Na", colour="blue", gap_after=5.0)
        b = AtomLegendItem(key="Na", colour="blue", gap_after=5.0)
        assert a == b

    def test_inequality_different_gap_after(self):
        a = AtomLegendItem(key="Na", colour="blue", gap_after=5.0)
        b = AtomLegendItem(key="Na", colour="blue", gap_after=10.0)
        assert a != b

    def test_equality_with_alpha(self):
        a = AtomLegendItem(key="Na", colour="blue", alpha=0.5)
        b = AtomLegendItem(key="Na", colour="blue", alpha=0.5)
        assert a == b

    def test_inequality_different_alpha(self):
        a = AtomLegendItem(key="Na", colour="blue", alpha=0.5)
        b = AtomLegendItem(key="Na", colour="blue", alpha=0.8)
        assert a != b


class TestPolygonLegendItem:
    def test_marker_type(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6)
        assert item.marker_type == "polygon"

    def test_sides_construction(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=8)
        assert item.sides == 8

    def test_rotation_default(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6)
        assert item.rotation == 0.0

    def test_rotation_construction(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        assert item.rotation == 30.0

    def test_sides_setter_valid(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=6)
        item.sides = 4
        assert item.sides == 4

    def test_sides_setter_too_small_raises(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=6)
        with pytest.raises(ValueError, match="sides must be >= 3"):
            item.sides = 2

    def test_sides_construction_too_small_raises(self):
        with pytest.raises(ValueError, match="sides must be >= 3"):
            PolygonLegendItem(key="Na", colour="blue", sides=1)

    def test_sides_bool_raises(self):
        with pytest.raises(TypeError, match="sides must be an int"):
            PolygonLegendItem(key="Na", colour="blue", sides=True)

    def test_sides_float_raises(self):
        with pytest.raises(TypeError, match="sides must be an int"):
            PolygonLegendItem(key="Na", colour="blue", sides=6.0)

    def test_sides_setter_bool_raises(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=6)
        with pytest.raises(TypeError, match="sides must be an int"):
            item.sides = True

    def test_sides_setter_float_raises(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=6)
        with pytest.raises(TypeError, match="sides must be an int"):
            item.sides = 4.0

    def test_rotation_setter(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=6)
        item.rotation = 45.0
        assert item.rotation == 45.0

    def test_rotation_coerced_to_float(self):
        item = PolygonLegendItem(key="Oct", colour="blue", sides=4, rotation=30)
        assert item.rotation == 30.0
        assert isinstance(item.rotation, float)

    def test_marker(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=15.0)
        assert item.marker == (6, 0, 15.0)

    def test_repr_with_sides(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6)
        r = repr(item)
        assert r.startswith("PolygonLegendItem(")
        assert "sides=6" in r
        assert "rotation" not in r

    def test_repr_with_sides_and_rotation(self):
        item = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        r = repr(item)
        assert "sides=6" in r
        assert "rotation=30.0" in r

    def test_equality(self):
        a = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        b = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        assert a == b

    def test_inequality_different_sides(self):
        a = PolygonLegendItem(key="Oct", colour="red", sides=6)
        b = PolygonLegendItem(key="Oct", colour="red", sides=4)
        assert a != b

    def test_inequality_different_rotation(self):
        a = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=0.0)
        b = PolygonLegendItem(key="Oct", colour="red", sides=6, rotation=45.0)
        assert a != b

    def test_inequality_different_subclass(self):
        polygon = PolygonLegendItem(key="X", colour="red", sides=6)
        atom = AtomLegendItem(key="X", colour="red")
        assert polygon != atom


class TestPolyhedronLegendItem:
    def test_marker_type(self):
        item = PolyhedronLegendItem(key="Oct", colour="red", shape="octahedron")
        assert item.marker_type == "polyhedron"

    def test_shape_construction(self):
        item = PolyhedronLegendItem(key="Oct", colour="red", shape="octahedron")
        assert item.shape == "octahedron"

    def test_shape_tetrahedron(self):
        item = PolyhedronLegendItem(
            key="Tet", colour="green", shape="tetrahedron",
        )
        assert item.shape == "tetrahedron"

    def test_shape_setter_valid(self):
        item = PolyhedronLegendItem(
            key="X", colour="blue", shape="octahedron",
        )
        item.shape = "tetrahedron"
        assert item.shape == "tetrahedron"

    def test_shape_unknown_raises(self):
        with pytest.raises(ValueError, match="shape must be one of"):
            PolyhedronLegendItem(key="Na", colour="blue", shape="cube")

    def test_shape_setter_unknown_raises(self):
        item = PolyhedronLegendItem(
            key="X", colour="blue", shape="octahedron",
        )
        with pytest.raises(ValueError, match="shape must be one of"):
            item.shape = "dodecahedron"

    def test_shape_non_string_raises(self):
        with pytest.raises(TypeError, match="shape must be a string"):
            PolyhedronLegendItem(key="Na", colour="blue", shape=6)

    def test_shape_setter_non_string_raises(self):
        item = PolyhedronLegendItem(
            key="X", colour="blue", shape="octahedron",
        )
        with pytest.raises(TypeError, match="shape must be a string"):
            item.shape = 4

    def test_rotation_default_none(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
        )
        assert item.rotation is None

    def test_rotation_matrix(self):
        rot = np.eye(3)
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron", rotation=rot,
        )
        np.testing.assert_array_equal(item.rotation, rot)

    def test_rotation_tuple(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
            rotation=(10.0, -15.0),
        )
        assert item.rotation is not None
        assert item.rotation.shape == (3, 3)

    def test_rotation_setter_none(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
            rotation=(10.0, -15.0),
        )
        item.rotation = None
        assert item.rotation is None

    def test_rotation_setter_matrix(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
        )
        rot = np.eye(3)
        item.rotation = rot
        np.testing.assert_array_equal(item.rotation, rot)

    def test_rotation_setter_tuple(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
        )
        item.rotation = (10.0, -15.0)
        assert item.rotation is not None
        assert item.rotation.shape == (3, 3)

    def test_rotation_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape.*3, 3"):
            PolyhedronLegendItem(
                key="Oct", colour="red", shape="octahedron",
                rotation=np.eye(4),
            )

    def test_rotation_bad_type_raises(self):
        with pytest.raises(TypeError, match="rotation must be"):
            PolyhedronLegendItem(
                key="Oct", colour="red", shape="octahedron",
                rotation="bad",
            )

    def test_repr(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
        )
        r = repr(item)
        assert r.startswith("PolyhedronLegendItem(")
        assert "shape='octahedron'" in r
        assert "rotation" not in r

    def test_repr_with_rotation(self):
        item = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
            rotation=(10.0, -15.0),
        )
        assert "rotation=" in repr(item)

    def test_equality(self):
        a = PolyhedronLegendItem(key="Oct", colour="red", shape="octahedron")
        b = PolyhedronLegendItem(key="Oct", colour="red", shape="octahedron")
        assert a == b

    def test_equality_with_rotation(self):
        rot = np.eye(3)
        a = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron", rotation=rot,
        )
        b = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron", rotation=rot,
        )
        assert a == b

    def test_inequality_different_shape(self):
        a = PolyhedronLegendItem(key="X", colour="red", shape="octahedron")
        b = PolyhedronLegendItem(key="X", colour="red", shape="tetrahedron")
        assert a != b

    def test_inequality_none_vs_set_rotation(self):
        a = PolyhedronLegendItem(key="Oct", colour="red", shape="octahedron")
        b = PolyhedronLegendItem(
            key="Oct", colour="red", shape="octahedron",
            rotation=np.eye(3),
        )
        assert a != b

    def test_inequality_different_subclass(self):
        poly = PolyhedronLegendItem(key="X", colour="red", shape="octahedron")
        atom = AtomLegendItem(key="X", colour="red")
        assert poly != atom


class TestPolyhedronLegendItemFromSpec:
    """Tests for PolyhedronLegendItem.from_polyhedron_spec()."""

    def test_basic(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour=(0.5, 0.5, 0.8))
        item = PolyhedronLegendItem.from_polyhedron_spec(spec, "octahedron")
        assert item.key == "Ti"
        assert item.colour == (0.5, 0.5, 0.8)
        assert item.alpha == 0.4  # spec default
        assert item.shape == "octahedron"
        assert item.label is None
        assert isinstance(item, PolyhedronLegendItem)

    def test_colour_override(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour=(0.5, 0.5, 0.8))
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", colour="red",
        )
        assert item.colour == (1.0, 0.0, 0.0)

    def test_none_colour_raises(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti")  # colour=None (inherits)
        with pytest.raises(ValueError, match="colour must be provided"):
            PolyhedronLegendItem.from_polyhedron_spec(spec, "octahedron")

    def test_none_spec_colour_with_explicit_colour(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti")
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "tetrahedron", colour="green",
        )
        assert item.colour == (0.0, 0.5019607843137255, 0.0)
        assert item.shape == "tetrahedron"

    def test_custom_key_and_label(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour="blue")
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", key="TiO6", label="Octahedral",
        )
        assert item.key == "TiO6"
        assert item.label == "Octahedral"

    def test_alpha_override(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour="blue", alpha=0.4)
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", alpha=0.8,
        )
        assert item.alpha == 0.8

    def test_radius_and_gap_after(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour="blue")
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", radius=8.0, gap_after=12.0,
        )
        assert item.radius == 8.0
        assert item.gap_after == 12.0

    def test_inherits_edge_colour(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(
            centre="Ti", colour="blue", edge_colour=(0.1, 0.2, 0.3),
        )
        item = PolyhedronLegendItem.from_polyhedron_spec(spec, "octahedron")
        assert item.edge_colour == (0.1, 0.2, 0.3)

    def test_inherits_edge_width(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(
            centre="Ti", colour="blue", edge_width=2.5,
        )
        item = PolyhedronLegendItem.from_polyhedron_spec(spec, "octahedron")
        assert item.edge_width == 2.5

    def test_edge_colour_override(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(
            centre="Ti", colour="blue", edge_colour=(0.1, 0.2, 0.3),
        )
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", edge_colour="red",
        )
        assert item.edge_colour == (1.0, 0.0, 0.0)

    def test_edge_width_override(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(
            centre="Ti", colour="blue", edge_width=2.5,
        )
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", edge_width=0.5,
        )
        assert item.edge_width == 0.5

    def test_rotation_passthrough(self):
        from hofmann.model import PolyhedronSpec
        spec = PolyhedronSpec(centre="Ti", colour="blue")
        item = PolyhedronLegendItem.from_polyhedron_spec(
            spec, "octahedron", rotation=(10.0, -15.0),
        )
        assert item.rotation is not None
        assert item.rotation.shape == (3, 3)


class TestLegendStyle:
    def test_defaults(self):
        style = LegendStyle()
        assert style.corner is WidgetCorner.BOTTOM_RIGHT
        assert style.margin == 0.15
        assert style.font_size == 10.0
        assert style.circle_radius == 5.0
        assert style.spacing == 2.5
        assert style.species is None

    def test_custom_values(self):
        style = LegendStyle(
            corner="top_left",
            margin=0.2,
            font_size=12.0,
            circle_radius=8.0,
            spacing=4.0,
            species=("Na", "Cl"),
        )
        assert style.corner is WidgetCorner.TOP_LEFT
        assert style.margin == 0.2
        assert style.font_size == 12.0
        assert style.circle_radius == 8.0
        assert style.spacing == 4.0
        assert style.species == ("Na", "Cl")

    def test_is_frozen(self):
        style = LegendStyle()
        with pytest.raises(AttributeError):
            style.font_size = 14.0

    def test_corner_string_coercion(self):
        style = LegendStyle(corner="top_left")
        assert style.corner is WidgetCorner.TOP_LEFT

    def test_negative_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            LegendStyle(font_size=-1.0)

    def test_zero_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size must be positive"):
            LegendStyle(font_size=0)

    def test_negative_circle_radius_raises(self):
        with pytest.raises(ValueError, match="circle_radius must be positive"):
            LegendStyle(circle_radius=-1.0)

    def test_zero_circle_radius_raises(self):
        with pytest.raises(ValueError, match="circle_radius must be positive"):
            LegendStyle(circle_radius=0)

    def test_negative_spacing_raises(self):
        with pytest.raises(ValueError, match="spacing must be non-negative"):
            LegendStyle(spacing=-0.1)

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin must be non-negative"):
            LegendStyle(margin=-0.1)

    def test_corner_tuple(self):
        style = LegendStyle(corner=(0.1, 0.9))
        assert style.corner == (0.1, 0.9)

    def test_corner_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="corner tuple must have 2 elements"):
            LegendStyle(corner=(0.1, 0.2, 0.3))

    def test_species_tuple(self):
        style = LegendStyle(species=("Na", "Cl"))
        assert style.species == ("Na", "Cl")

    def test_empty_species_raises(self):
        with pytest.raises(ValueError, match="species must be non-empty"):
            LegendStyle(species=())

    # ---- circle_radius variants ----

    def test_circle_radius_range_tuple(self):
        style = LegendStyle(circle_radius=(3.0, 8.0))
        assert style.circle_radius == (3.0, 8.0)

    def test_circle_radius_dict(self):
        style = LegendStyle(circle_radius={"Na": 5.0, "Cl": 8.0})
        assert style.circle_radius == {"Na": 5.0, "Cl": 8.0}

    def test_circle_radius_range_non_positive_min_raises(self):
        with pytest.raises(ValueError, match="circle_radius range values must be positive"):
            LegendStyle(circle_radius=(0.0, 8.0))

    def test_circle_radius_range_non_positive_max_raises(self):
        with pytest.raises(ValueError, match="circle_radius range values must be positive"):
            LegendStyle(circle_radius=(-1.0, 8.0))

    def test_circle_radius_range_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="circle_radius min must not exceed max"):
            LegendStyle(circle_radius=(10.0, 3.0))

    def test_circle_radius_dict_non_positive_raises(self):
        with pytest.raises(ValueError, match="circle_radius dict values must be positive"):
            LegendStyle(circle_radius={"Na": 0.0})

    def test_circle_radius_dict_empty_raises(self):
        with pytest.raises(ValueError, match="circle_radius dict must be non-empty"):
            LegendStyle(circle_radius={})

    # ---- label_gap ----

    def test_label_gap_default(self):
        assert LegendStyle().label_gap == 5.0

    def test_label_gap_negative_raises(self):
        with pytest.raises(ValueError, match="label_gap must be non-negative"):
            LegendStyle(label_gap=-1.0)

    # ---- labels ----

    def test_labels_default_none(self):
        assert LegendStyle().labels is None

    def test_labels_dict(self):
        labels = {"Ti": "$\\mathrm{Ti^{4+}}$", "O": "$\\mathrm{O^{2-}}$"}
        style = LegendStyle(labels=labels)
        assert style.labels == labels

    # ---- items ----

    def test_items_default_none(self):
        assert LegendStyle().items is None

    def test_items_tuple(self):
        items = (
            AtomLegendItem(key="oct", colour="blue"),
            AtomLegendItem(key="tet", colour="red"),
        )
        style = LegendStyle(items=items)
        assert style.items == items

    def test_empty_items_raises(self):
        with pytest.raises(ValueError, match="items must be non-empty"):
            LegendStyle(items=())

    def test_items_non_legend_item_raises(self):
        with pytest.raises(TypeError, match=r"items\[0\] must be a LegendItem"):
            LegendStyle(items=({"key": "Na", "colour": "blue"},))
