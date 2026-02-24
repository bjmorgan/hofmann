Rendering
=========

Controlling the view
--------------------

The :class:`~hofmann.ViewState` controls rotation, zoom, perspective,
and slab clipping.

Rotation
~~~~~~~~

Set the viewing direction with :meth:`~hofmann.ViewState.look_along`:

.. code-block:: python

   scene.view.look_along([1, 1, 0])  # View along [110]

Or set the rotation matrix directly:

.. code-block:: python

   import numpy as np
   scene.view.rotation = np.eye(3)  # Identity (default)

Zoom
~~~~

.. code-block:: python

   scene.view.zoom = 1.5  # Zoom in

Perspective
~~~~~~~~~~~

.. code-block:: python

   scene.view.perspective = 0.3  # Mild perspective
   scene.view.perspective = 0.0  # Orthographic (default)

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/perovskite_ortho.svg

          Orthographic (``perspective=0.0``)

     - .. figure:: _static/perovskite_perspective.svg

          Perspective (``perspective=0.5``)

Slab clipping
~~~~~~~~~~~~~

Restrict the visible depth range to show a slice through the structure:

.. code-block:: python

   scene.view.slab_near = -2.0
   scene.view.slab_far = 2.0

See :meth:`~hofmann.ViewState.slab_mask` for how slab visibility is
computed.


Render styles
-------------

:class:`~hofmann.RenderStyle` groups all visual appearance settings
independent of the structure data.  You can pass a full style object
or use convenience keyword arguments:

.. code-block:: python

   from hofmann import RenderStyle

   # Via a style object:
   style = RenderStyle(
       atom_scale=0.8,
       show_outlines=False,
       half_bonds=False,
   )
   scene.render_mpl("clean.svg", style=style)

   # Or as convenience kwargs:
   scene.render_mpl("clean.svg", atom_scale=0.8,
                     show_outlines=False, half_bonds=False)

Any :class:`~hofmann.RenderStyle` field can be passed as a keyword
argument to :meth:`~hofmann.StructureScene.render_mpl`.  Unknown
keyword names raise :class:`TypeError`.

Here is the same SrTiO\ :sub:`3` perovskite rendered with different styles:

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/perovskite_plain.svg

          Ball-and-stick

     - .. figure:: _static/perovskite.svg

          With polyhedra

   * - .. figure:: _static/perovskite_spacefill.svg

          Space-filling (``atom_scale=1.0``)

     - .. figure:: _static/perovskite_no_outlines.svg

          Outlines disabled (``show_outlines=False``)

Key style options:

- ``atom_scale`` -- ``0.5`` for ball-and-stick, ``1.0`` for space-filling
- ``half_bonds`` -- colour each bond half to match the nearest atom
- ``show_bonds`` / ``show_polyhedra`` -- toggle bond or polyhedra drawing
- ``show_outlines`` -- toggle atom and bond outlines
- ``show_cell`` -- toggle unit cell edges (auto-detected by default;
  see :ref:`unit-cell` below)
- ``cell_style`` -- :class:`~hofmann.CellEdgeStyle` for cell edge
  colour, width, and linestyle
- ``show_axes`` -- toggle axes orientation widget (auto-detected by
  default; see :ref:`axes-widget` below)
- ``axes_style`` -- :class:`~hofmann.AxesStyle` for widget
  colour, labels, corner, and sizing
- ``show_legend`` -- toggle species legend (off by default; see
  :ref:`legend-widget` below)
- ``legend_style`` -- :class:`~hofmann.LegendStyle` for legend
  corner, sizing, and species selection
- ``slab_clip_mode`` -- how slab clipping interacts with polyhedra
  (see :ref:`slab-clipping` below)
- ``circle_segments`` / ``arc_segments`` -- polygon resolution for
  static output (defaults are publication quality)
- ``interactive_circle_segments`` / ``interactive_arc_segments`` --
  polygon resolution for the interactive viewer (lower defaults
  for responsive redraws)

Half-bonds
~~~~~~~~~~

When ``half_bonds=True`` (the default), each bond is split at the
midpoint and each half is coloured to match the nearest atom.  With
``half_bonds=False``, bonds use the colour from their
:class:`~hofmann.BondSpec`.

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/octahedron_half_bonds.svg

          ``half_bonds=True`` (default)

     - .. figure:: _static/octahedron_no_half_bonds.svg

          ``half_bonds=False``


Polyhedra shading
-----------------

The ``polyhedra_shading`` setting controls diffuse (Lambertian) shading
on polyhedra faces.  At ``0.0`` all faces are flat; at ``1.0`` (the
default) faces pointing towards the viewer are bright and edge-on faces
are dimmed.

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/octahedron_shading_flat.svg

          ``polyhedra_shading=0.0`` (flat)

     - .. figure:: _static/octahedron_shading_full.svg

          ``polyhedra_shading=1.0`` (Lambertian)

.. _slab-clipping:

Slab clipping and polyhedra
---------------------------

The ``slab_clip_mode`` setting on :class:`~hofmann.RenderStyle`
controls how polyhedra at the slab boundary are handled:

- ``"per_face"`` (default) -- drop individual faces whose vertices
  are outside the slab
- ``"clip_whole"`` -- hide the entire polyhedron if any vertex is
  clipped
- ``"include_whole"`` -- force the complete polyhedron to be visible
  when its centre atom is within the slab

Here is the LLZO garnet with a depth slab that clips through several
ZrO\ :sub:`6` octahedra, rendered with each mode:

.. list-table::
   :widths: 33 33 33

   * - .. figure:: _static/llzo_clip_whole.svg

          ``"clip_whole"``

     - .. figure:: _static/llzo_clip_per_face.svg

          ``"per_face"``

     - .. figure:: _static/llzo_clip_include_whole.svg

          ``"include_whole"``


.. _unit-cell:

Unit cell
---------

For scenes created from pymatgen ``Structure`` objects, the unit cell
wireframe is drawn automatically.  The 12 cell edges are
depth-interleaved with atoms, bonds, and polyhedra so they correctly
occlude and are occluded.

Disable cell edges or customise their appearance via
:class:`~hofmann.RenderStyle`:

.. code-block:: python

   # Disable cell edges:
   scene.render_mpl("output.svg", show_cell=False)

   # Custom cell edge style:
   from hofmann import CellEdgeStyle, RenderStyle

   style = RenderStyle(
       cell_style=CellEdgeStyle(
           colour="blue",
           line_width=1.2,
           linestyle="dashed",
       ),
   )
   scene.render_mpl("output.svg", style=style)

Available linestyles: ``"solid"`` (default), ``"dashed"``,
``"dotted"``, and ``"dashdot"``.

Scenes loaded from XBS files have no lattice information, so cell
edges are not drawn.  You can set a lattice manually:

.. code-block:: python

   import numpy as np
   scene = StructureScene.from_xbs("structure.bs")
   scene.lattice = np.diag([5.43, 5.43, 5.43])  # Cubic, 5.43 A


.. _axes-widget:

Axes orientation widget
-----------------------

For periodic structures, an axes orientation widget shows the
crystallographic **a**, **b**, **c** lattice directions as lines in
a corner of the figure.  The widget is drawn automatically when a
lattice is present (the same auto-detection as unit cell edges) and
rotates in sync with the structure.

Disable or customise the widget via :class:`~hofmann.RenderStyle`:

.. code-block:: python

   # Disable the axes widget:
   scene.render_mpl("output.svg", show_axes=False)

   # Custom widget style:
   from hofmann import AxesStyle, RenderStyle

   style = RenderStyle(
       axes_style=AxesStyle(
           corner="top_right",
           colours=("red", "green", "blue"),
           labels=("x", "y", "z"),
       ),
   )
   scene.render_mpl("output.svg", style=style)


.. _legend-widget:

Species legend
--------------

A species legend maps each atom species to its coloured circle.  Unlike
cell edges and the axes widget, the legend is off by default — enable it
with ``show_legend=True``:

.. code-block:: python

   scene.render_mpl("output.svg", show_legend=True)

.. figure:: _static/legend_perovskite.svg
   :align: center

   SrTiO\ :sub:`3` perovskite with ``show_legend=True``.

Customise the legend via :class:`~hofmann.LegendStyle`:

.. code-block:: python

   from hofmann import LegendStyle, RenderStyle

   style = RenderStyle(
       show_legend=True,
       legend_style=LegendStyle(
           corner="top_right",
           font_size=12.0,
       ),
   )
   scene.render_mpl("output.svg", style=style)

By default the legend auto-detects species from the scene in first-seen
order, filtering to those with ``visible=True``.  To control which
species appear and in what order, pass an explicit ``species`` tuple:

.. code-block:: python

   LegendStyle(species=("O", "Ti", "Sr"))

The ``spacing`` and ``label_gap`` parameters control the vertical gap
between entries and the horizontal gap between each circle and its
label, respectively (both in points):

.. code-block:: python

   LegendStyle(spacing=5.0, label_gap=8.0)

Custom labels
~~~~~~~~~~~~~

Pass a ``labels`` dict to override the display text for any species.
Common chemical notation is auto-formatted: trailing charges become
superscripts (``Sr2+`` → Sr²⁺) and embedded digits become subscripts
(``TiO6`` → TiO₆).  Labels containing ``$`` are passed through as
explicit matplotlib mathtext.

.. code-block:: python

   LegendStyle(labels={
       "Sr": "Sr2+",       # auto superscript
       "Ti": "TiO6",       # auto subscript
       "O":  r"$\mathrm{O^{2\!-}}$",  # explicit mathtext
   })

Species not in the dict use their name as-is.

.. figure:: _static/legend_labels.svg
   :align: center

Circle sizing
~~~~~~~~~~~~~

The ``circle_radius`` parameter controls the size of the legend
circles and accepts three forms:

- **float** — uniform radius for all entries (the default, ``5.0``
  points).
- **tuple (min, max)** — proportional sizing.  Each species' circle
  is scaled linearly between *min* and *max* based on its
  ``AtomStyle.radius`` relative to the smallest and largest radii in
  the legend.  When all atom radii are equal, *max* is used.
- **dict** — explicit per-species radii in points.  Species not
  present in the dict fall back to the default (5.0 points).

.. code-block:: python

   # Proportional: smaller atoms get smaller circles.
   LegendStyle(circle_radius=(3.0, 8.0))

   # Explicit per-species:
   LegendStyle(circle_radius={"O": 4.0, "Ti": 6.0, "Sr": 8.0})

.. list-table::
   :widths: 33 33 33

   * - .. figure:: _static/legend_uniform.svg
          :align: center

          Uniform (``5.0``)

     - .. figure:: _static/legend_proportional.svg
          :align: center

          Proportional (``(3.0, 7.0)``)

     - .. figure:: _static/legend_dict.svg
          :align: center

          Dict (per-species)

Custom legend items
~~~~~~~~~~~~~~~~~~~

When colouring atoms by custom data (via ``colour_by``), the default
species-based legend may not reflect the active colouring.  Pass a
tuple of :class:`~hofmann.LegendItem` instances to display a fully
custom legend:

.. code-block:: python

   from hofmann import LegendItem, LegendStyle

   style = LegendStyle(items=(
       LegendItem(key="octahedral", colour="blue", label="Octahedral"),
       LegendItem(key="tetrahedral", colour="red", label="Tetrahedral"),
   ))
   scene.render_mpl("output.svg", show_legend=True, legend_style=style)

When ``items`` is provided, the ``species`` and ``labels`` parameters
are ignored — each item carries its own key, colour, and optional
label.  Items with ``radius=None`` fall back to ``circle_radius``
when that is a plain float, or to 5.0 points otherwise (the
proportional and per-species dict modes do not apply).

:class:`~hofmann.LegendItem` is mutable, so items can be tweaked
after creation:

.. code-block:: python

   item = LegendItem(key="oct", colour="blue")
   item.label = "Octahedral"
   item.radius = 8.0

Standalone legend
~~~~~~~~~~~~~~~~~

Use :func:`~hofmann.rendering.static.render_legend` to produce a
legend-only image — useful for composing figures manually in Inkscape,
Illustrator, or LaTeX:

.. code-block:: python

   from hofmann.rendering.static import render_legend

   render_legend(scene, "legend.svg")

   # With proportional sizing:
   from hofmann import LegendStyle
   render_legend(scene, "legend.svg",
                 legend_style=LegendStyle(circle_radius=(3.0, 8.0)))

The figure is cropped tightly to the legend entries and has no axes or
structure.  See :func:`~hofmann.rendering.static.render_legend` for the
full parameter list.


Rendering into existing axes
-----------------------------

By default :meth:`~hofmann.StructureScene.render_mpl` creates its own
figure.  Pass the ``ax`` parameter to draw into an existing matplotlib
axes instead — useful for multi-panel figures or combining a structure
with other plots:

.. code-block:: python

   import matplotlib.pyplot as plt
   from hofmann import StructureScene

   scene = StructureScene.from_xbs("structure.bs")

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
   ax1.plot(x, y)              # Your own data
   scene.render_mpl(ax=ax2)    # Structure alongside
   fig.savefig("panel.pdf", bbox_inches="tight")

When ``ax`` is provided, the caller retains full control of the parent
figure — the *output*, *figsize*, *dpi*, *background*, and *show*
parameters are ignored.

As an example, here is rutile TiO\ :sub:`2` viewed along [100] and
[001], showing the distinct projections perpendicular and parallel to
the *c* axis:

.. image:: _static/multi_panel_projections.svg

.. code-block:: python

   import matplotlib.pyplot as plt
   from hofmann import StructureScene

   scene = StructureScene.from_pymatgen(structure, bond_specs)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
   for ax, direction, label in zip(
       [ax1, ax2], [[1, 0, 0], [0, 0, 1]], ["[100]", "[001]"],
   ):
       scene.view.look_along(direction)
       scene.title = label
       scene.render_mpl(ax=ax)

   fig.tight_layout()
   fig.savefig("projections.pdf", bbox_inches="tight")
