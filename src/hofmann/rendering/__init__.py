"""Rendering: depth-sorted matplotlib output (static and interactive)."""

from hofmann.rendering.animation import render_animation
from hofmann.rendering.interactive import render_mpl_interactive
from hofmann.rendering.static import render_mpl

__all__ = [
    "render_animation",
    "render_mpl",
    "render_mpl_interactive",
]
