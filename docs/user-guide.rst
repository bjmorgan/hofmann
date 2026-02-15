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
   scene.render_mpl("clean.svg", style=style, show=False)

   # Or as convenience kwargs:
   scene.render_mpl("clean.svg", atom_scale=0.8,
                     show_outlines=False, half_bonds=False,
                     show=False)

Any :class:`~hofmann.RenderStyle` field can be passed as a keyword
argument to :meth:`~hofmann.StructureScene.render_mpl`.  Unknown
keyword names raise :class:`TypeError`.

Key style options:

- ``atom_scale`` -- ``0.5`` for ball-and-stick, ``1.0`` for space-filling
- ``half_bonds`` -- colour each bond half to match the nearest atom
- ``show_bonds`` / ``show_polyhedra`` -- toggle bond or polyhedra drawing
- ``show_outlines`` -- toggle atom and bond outlines
- ``circle_segments`` / ``arc_segments`` -- polygon resolution for
  vector output (increase for publication quality)


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

Coordination polyhedra are specified with
:class:`~hofmann.PolyhedronSpec`:

.. code-block:: python

   from hofmann import PolyhedronSpec

   spec = PolyhedronSpec(
       centre_species="Ti",
       vertex_species="O",
       max_distance=2.5,
       colour=(0.5, 0.7, 1.0),
       alpha=0.3,
   )
   scene = StructureScene.from_pymatgen(
       structure, bonds, polyhedra=[spec], pbc=True,
   )

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


Interactive viewer
------------------

Open an interactive matplotlib window with mouse rotation and zoom:

.. code-block:: python

   view = scene.render_mpl_interactive()

   # Reuse the adjusted view for static output:
   scene.view = view
   scene.render_mpl("adjusted.svg", show=False)

Left-drag rotates; scroll zooms.  The returned
:class:`~hofmann.ViewState` captures the final orientation.
