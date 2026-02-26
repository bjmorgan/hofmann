"""Compare cuboctahedron alpha values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "_static"))
from generate_images import perovskite_plain_scene  # noqa: E402

from hofmann import LegendStyle, PolyhedronLegendItem  # noqa: E402
from hofmann.rendering.static import render_legend  # noqa: E402

OUT = Path(__file__).resolve().parent
scene = perovskite_plain_scene()

for alpha in [0.4, 0.6, 0.8, 1.0]:
    items = (
        PolyhedronLegendItem(key="oct", colour="steelblue",
                             label="Octahedral", shape="octahedron",
                             alpha=alpha),
        PolyhedronLegendItem(key="tet", colour="goldenrod",
                             label="Tetrahedral", shape="tetrahedron",
                             alpha=alpha),
        PolyhedronLegendItem(key="cuboct", colour="mediumseagreen",
                             label="Cuboctahedral",
                             shape="cuboctahedron", alpha=alpha),
    )
    name = f"alpha_{alpha:.1f}".replace(".", "")
    render_legend(scene, OUT / f"check_{name}.png",
                  legend_style=LegendStyle(items=items), dpi=200)
print("done")
