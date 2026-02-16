User guide
==========

Scenes and frames
-----------------

A :class:`~hofmann.StructureScene` is the central object in hofmann.  It
holds everything needed to render a structure:

- **species** -- one label per atom (e.g. ``["C", "H", "H", "H", "H"]``)
- **frames** -- one or more :class:`~hofmann.Frame` coordinate snapshots
- **atom_styles** -- mapping from species to :class:`~hofmann.AtomStyle`
  (radius and colour)
- **bond_specs** -- declarative :class:`~hofmann.BondSpec` rules
- **polyhedra** -- optional :class:`~hofmann.PolyhedronSpec` rules
- **view** -- a :class:`~hofmann.ViewState` controlling the camera

Scenes are typically created via :meth:`~hofmann.StructureScene.from_xbs`
or :meth:`~hofmann.StructureScene.from_pymatgen`, but you can also
construct one directly from data.

Here is a simple CH4 molecule loaded from an XBS file:

.. plot::
   :context: reset

   import hofmann
   from pathlib import Path
   from hofmann import StructureScene

   pkg_dir = Path(hofmann.__file__).resolve().parent
   fixture = pkg_dir.parent.parent / "tests" / "fixtures" / "ch4.bs"
   scene = StructureScene.from_xbs(fixture)
   scene.render_mpl(show=False, figsize=(4, 4))


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

Here is the same SrTiO3 perovskite rendered with different styles:

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


Bonds
-----

Bonds are detected at render time from declarative
:class:`~hofmann.BondSpec` rules.  Each rule specifies a species pair,
a length range, a display radius, and a colour:

.. code-block:: python

   from hofmann import BondSpec

   spec = BondSpec(
       species=("C", "H"),
       min_length=0.0,
       max_length=1.2,
       radius=0.1,
       colour=0.8,  # Grey
   )

Species matching supports wildcards:

.. code-block:: python

   # Match any bond between any species:
   BondSpec(species=("*", "*"), min_length=0.0, max_length=2.5,
            radius=0.1, colour="grey")

When no bond specs are provided, :func:`~hofmann.from_pymatgen`
generates sensible defaults from :data:`~hofmann.COVALENT_RADII`.


Polyhedra
---------

Coordination polyhedra are built from the bond graph: for each atom
whose species matches the ``centre`` pattern, a convex hull is
constructed from its bonded neighbours.

.. code-block:: python

   from hofmann import PolyhedronSpec

   spec = PolyhedronSpec(
       centre="Ti",
       colour=(0.5, 0.7, 1.0),
       alpha=0.3,
   )
   scene = StructureScene.from_pymatgen(
       structure, bonds, polyhedra=[spec], pbc=True,
   )

Polyhedra shading
~~~~~~~~~~~~~~~~~

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

Vertex draw order
~~~~~~~~~~~~~~~~~

The ``polyhedra_vertex_mode`` setting controls how vertex atoms are
layered relative to polyhedral faces.  ``"in_front"`` (the default)
draws each vertex on top of the faces it belongs to.

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/octahedron_vertex_in_front.svg

          Opaque polyhedra

     - .. figure:: _static/octahedron_vertex_in_front_transparent.svg

          Transparent polyhedra

.. _slab-clipping:

Slab clipping and polyhedra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``slab_clip_mode`` setting on :class:`~hofmann.RenderStyle`
controls how polyhedra at the slab boundary are handled:

- ``"per_face"`` (default) -- drop individual faces whose vertices
  are outside the slab
- ``"clip_whole"`` -- hide the entire polyhedron if any vertex is
  clipped
- ``"include_whole"`` -- force the complete polyhedron to be visible
  when its centre atom is within the slab

Here is the LLZO garnet with a depth slab that clips through several
ZrO6 octahedra, rendered with each mode:

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

The widget also rotates interactively in
:meth:`~hofmann.StructureScene.render_mpl_interactive`.


Interactive viewer
------------------

Open an interactive matplotlib window with mouse rotation and zoom:

.. code-block:: python

   view = scene.render_mpl_interactive()

   # Reuse the adjusted view for static output:
   scene.view = view
   scene.render_mpl("adjusted.svg")

Left-drag rotates; scroll zooms.  The returned
:class:`~hofmann.ViewState` captures the final orientation.
