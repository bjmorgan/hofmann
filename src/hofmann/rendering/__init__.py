"""Rendering: depth-sorted matplotlib output (static and interactive)."""

from hofmann.rendering.interactive import render_mpl_interactive
from hofmann.rendering.static import render_mpl


def render_animation(*args, **kwargs):
    """Render a trajectory animation to a GIF or MP4 file.

    Lazily imports :mod:`hofmann.rendering.animation` so that
    optional dependencies (Pillow) are only required when animation
    rendering is actually used.

    See :func:`hofmann.rendering.animation.render_animation` for
    full documentation.
    """
    from hofmann.rendering.animation import render_animation as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "render_animation",
    "render_mpl",
    "render_mpl_interactive",
]
