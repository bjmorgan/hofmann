"""Tests for static rendering utilities — style resolution."""

from hofmann.model import RenderStyle
from hofmann.rendering.static import _DEFAULT_RENDER_STYLE, _resolve_style


class TestResolveStyle:
    def test_none_kwarg_preserves_base_style(self):
        """Passing None for a kwarg preserves the base style's value."""
        style = RenderStyle(atom_scale=0.8)
        resolved = _resolve_style(style, atom_scale=None)
        assert resolved.atom_scale == 0.8

    def test_explicit_kwarg_overrides_style(self):
        style = RenderStyle(atom_scale=0.5)
        resolved = _resolve_style(style, atom_scale=0.8)
        assert resolved.atom_scale == 0.8

    def test_absent_kwarg_keeps_style_value(self):
        style = RenderStyle(atom_scale=0.8)
        resolved = _resolve_style(style)
        assert resolved.atom_scale == 0.8

    def test_default_style_not_mutated_by_override(self):
        """Overriding with style=None should not mutate the module-level default."""
        original = RenderStyle()
        _resolve_style(None, atom_scale=0.8)
        assert _DEFAULT_RENDER_STYLE == original

    def test_default_style_not_mutated_by_caller(self):
        """Mutating the returned style should not affect the module-level default."""
        resolved = _resolve_style(None)
        resolved.atom_scale = 99.0
        assert _DEFAULT_RENDER_STYLE.atom_scale == RenderStyle().atom_scale
