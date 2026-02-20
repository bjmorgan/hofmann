"""Rendering: depth-sorted matplotlib output (static and interactive)."""

from hofmann.rendering.interactive import render_mpl_interactive
from hofmann.rendering.static import render_mpl

__all__ = [
    "render_mpl",
    "render_mpl_interactive",
]
