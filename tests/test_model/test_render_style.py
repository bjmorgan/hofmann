"""Tests for RenderStyle, SlabClipMode, CellEdgeStyle, WidgetCorner, and AxesStyle."""

import pytest

from hofmann.model.render_style import (
    AxesStyle,
    CellEdgeStyle,
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
