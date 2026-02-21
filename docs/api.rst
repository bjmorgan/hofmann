API reference
=============

.. module:: hofmann

Scene construction
------------------

.. autoclass:: StructureScene
   :members:

.. autofunction:: from_xbs

.. autofunction:: from_pymatgen


Data model
----------

.. autoclass:: Frame
   :members:

.. autoclass:: AtomStyle
   :members:

.. autoclass:: BondSpec
   :members:

.. autoclass:: Bond
   :members:

.. autoclass:: PolyhedronSpec
   :members:

.. autoclass:: Polyhedron
   :members:

.. autoclass:: ViewState
   :members:


Rendering
---------

.. autoclass:: RenderStyle
   :members:

.. autoclass:: CellEdgeStyle
   :members:

.. autoclass:: AxesStyle
   :members:

.. autoclass:: WidgetCorner
   :members:

.. autoclass:: SlabClipMode
   :members:

.. autofunction:: hofmann.rendering.static.render_mpl

.. autofunction:: hofmann.rendering.interactive.render_mpl_interactive


Colours and defaults
--------------------

.. data:: Colour
   :type: str | float | tuple[float, float, float] | list[float]

   A colour specification accepted throughout hofmann.

   Can be any of:

   - A CSS colour name or hex string (e.g. ``"red"``, ``"#ff0000"``).
   - A single float for grey (``0.0`` = black, ``1.0`` = white).
   - An RGB tuple or list with values in ``[0, 1]``
     (e.g. ``(1.0, 0.0, 0.0)``).

   See :func:`normalise_colour` for conversion to a normalised RGB tuple.

.. autofunction:: normalise_colour

.. autofunction:: resolve_atom_colours

.. data:: CmapSpec

   Type alias for colourmap specifications accepted by
   :func:`resolve_atom_colours` and the ``cmap`` parameter of render
   methods.  See the type definition in :mod:`hofmann.model` for details.

.. data:: ELEMENT_COLOURS
   :type: dict[str, tuple[float, float, float]]

   Mapping from element symbols to muted, publication-friendly RGB colours.
   Common elements use hand-picked colours; less common elements use
   desaturated tones grouped by periodic table region.  Values are
   normalised to the ``[0, 1]`` range.

.. data:: COVALENT_RADII
   :type: dict[str, float]

   Covalent radii in angstroms, from Cordero *et al.*, Dalton Trans. 2008.
   Used by :func:`default_atom_style` for display radii.

.. autofunction:: default_atom_style

.. autofunction:: default_bond_specs


Style I/O
---------

.. autoclass:: StyleSet
   :members:

.. autofunction:: save_styles

.. autofunction:: load_styles


Bond and polyhedra computation
------------------------------

.. autofunction:: compute_bonds

.. autofunction:: compute_polyhedra
