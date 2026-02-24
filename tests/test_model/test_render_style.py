"""Tests for RenderStyle, SlabClipMode, CellEdgeStyle, WidgetCorner, AxesStyle, LegendItem, and LegendStyle."""

import pytest

from hofmann.model.render_style import (
    AxesStyle,
    CellEdgeStyle,
    LegendItem,
    LegendStyle,
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


class TestLegendItem:
    def test_required_fields(self):
        item = LegendItem(key="Na", colour="blue")
        assert item.key == "Na"
        assert item.colour == (0.0, 0.0, 1.0)
        assert item.label is None
        assert item.radius is None

    def test_all_fields(self):
        item = LegendItem(key="Na", colour=(0.2, 0.4, 0.8), label="Sodium", radius=6.0)
        assert item.key == "Na"
        assert item.colour == (0.2, 0.4, 0.8)
        assert item.label == "Sodium"
        assert item.radius == 6.0

    def test_colour_setter_valid(self):
        item = LegendItem(key="Na", colour="blue")
        item.colour = "red"
        assert item.colour == (1.0, 0.0, 0.0)

    def test_colour_setter_invalid_raises(self):
        item = LegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError):
            item.colour = "not_a_colour"

    def test_radius_setter_valid(self):
        item = LegendItem(key="Na", colour="blue")
        item.radius = 3.0
        assert item.radius == 3.0

    def test_radius_setter_none(self):
        item = LegendItem(key="Na", colour="blue", radius=5.0)
        item.radius = None
        assert item.radius is None

    def test_radius_setter_negative_raises(self):
        item = LegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="radius must be positive"):
            item.radius = -1.0

    def test_radius_setter_zero_raises(self):
        item = LegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="radius must be positive"):
            item.radius = 0.0

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key must be non-empty"):
            LegendItem(key="", colour="blue")

    def test_invalid_colour_raises(self):
        with pytest.raises(ValueError):
            LegendItem(key="Na", colour="not_a_colour")

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            LegendItem(key="Na", colour="blue", radius=-1.0)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            LegendItem(key="Na", colour="blue", radius=0.0)

    def test_display_label_falls_back_to_key(self):
        item = LegendItem(key="Na", colour="blue")
        assert item.display_label == "Na"

    def test_display_label_returns_label(self):
        item = LegendItem(key="Na", colour="blue", label="Sodium")
        assert item.display_label == "Sodium"

    def test_label_is_mutable(self):
        item = LegendItem(key="Na", colour="blue")
        item.label = "Sodium"
        assert item.label == "Sodium"
        assert item.display_label == "Sodium"

    def test_repr_minimal(self):
        item = LegendItem(key="Na", colour="blue")
        r = repr(item)
        assert "key='Na'" in r
        assert "colour=(0.0, 0.0, 1.0)" in r
        assert "label" not in r
        assert "radius" not in r

    def test_repr_full(self):
        item = LegendItem(key="Na", colour="blue", label="Sodium", radius=6.0)
        r = repr(item)
        assert "label='Sodium'" in r
        assert "radius=6.0" in r

    def test_equality(self):
        a = LegendItem(key="Na", colour="blue", label="Sodium", radius=6.0)
        b = LegendItem(key="Na", colour="blue", label="Sodium", radius=6.0)
        assert a == b

    def test_equality_normalised_colours(self):
        a = LegendItem(key="Na", colour="blue")
        b = LegendItem(key="Na", colour=(0.0, 0.0, 1.0))
        assert a == b

    def test_inequality_different_colour(self):
        a = LegendItem(key="Na", colour="blue")
        b = LegendItem(key="Na", colour="red")
        assert a != b

    def test_inequality_not_legend_item(self):
        item = LegendItem(key="Na", colour="blue")
        assert item != "not an item"

    def test_not_hashable(self):
        item = LegendItem(key="Na", colour="blue")
        with pytest.raises(TypeError):
            hash(item)

    # ---- sides / rotation ----

    def test_sides_defaults_to_none(self):
        item = LegendItem(key="Na", colour="blue")
        assert item.sides is None

    def test_rotation_defaults_to_zero(self):
        item = LegendItem(key="Na", colour="blue")
        assert item.rotation == 0.0

    def test_sides_construction(self):
        item = LegendItem(key="Oct", colour="red", sides=8)
        assert item.sides == 8

    def test_rotation_construction(self):
        item = LegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        assert item.rotation == 30.0

    def test_sides_setter_valid(self):
        item = LegendItem(key="Na", colour="blue")
        item.sides = 4
        assert item.sides == 4

    def test_sides_setter_none(self):
        item = LegendItem(key="Na", colour="blue", sides=6)
        item.sides = None
        assert item.sides is None

    def test_sides_setter_too_small_raises(self):
        item = LegendItem(key="Na", colour="blue")
        with pytest.raises(ValueError, match="sides must be >= 3"):
            item.sides = 2

    def test_sides_construction_too_small_raises(self):
        with pytest.raises(ValueError, match="sides must be >= 3"):
            LegendItem(key="Na", colour="blue", sides=1)

    def test_rotation_setter(self):
        item = LegendItem(key="Na", colour="blue", sides=6)
        item.rotation = 45.0
        assert item.rotation == 45.0

    def test_rotation_coerced_to_float(self):
        item = LegendItem(key="Na", colour="blue", sides=4, rotation=30)
        assert item.rotation == 30.0
        assert isinstance(item.rotation, float)

    def test_marker_circle(self):
        item = LegendItem(key="Na", colour="blue")
        assert item.marker == "o"

    def test_marker_polygon(self):
        item = LegendItem(key="Oct", colour="red", sides=6, rotation=15.0)
        assert item.marker == (6, 0, 15.0)

    def test_repr_with_sides(self):
        item = LegendItem(key="Oct", colour="red", sides=6)
        r = repr(item)
        assert "sides=6" in r
        assert "rotation" not in r

    def test_repr_with_sides_and_rotation(self):
        item = LegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        r = repr(item)
        assert "sides=6" in r
        assert "rotation=30.0" in r

    def test_equality_with_sides(self):
        a = LegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        b = LegendItem(key="Oct", colour="red", sides=6, rotation=30.0)
        assert a == b

    def test_inequality_different_sides(self):
        a = LegendItem(key="Oct", colour="red", sides=6)
        b = LegendItem(key="Oct", colour="red", sides=4)
        assert a != b

    def test_inequality_different_rotation(self):
        a = LegendItem(key="Oct", colour="red", sides=6, rotation=0.0)
        b = LegendItem(key="Oct", colour="red", sides=6, rotation=45.0)
        assert a != b


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
            LegendItem(key="oct", colour="blue"),
            LegendItem(key="tet", colour="red"),
        )
        style = LegendStyle(items=items)
        assert style.items == items

    def test_empty_items_raises(self):
        with pytest.raises(ValueError, match="items must be non-empty"):
            LegendStyle(items=())
