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

.. autofunction:: hofmann.render_mpl.render_mpl

.. autofunction:: hofmann.render_mpl.render_mpl_interactive


Colours and defaults
--------------------

.. autodata:: Colour

.. autofunction:: normalise_colour

.. data:: ELEMENT_COLOURS
   :type: dict[str, tuple[float, float, float]]

   Mapping from element symbols to muted, publication-friendly RGB colours.
   Common elements use hand-picked colours; less common elements use
   desaturated tones grouped by periodic table region.  Values are
   normalised to the ``[0, 1]`` range.

.. data:: COVALENT_RADII
   :type: dict[str, float]

   Covalent radii in angstroms, from Cordero *et al.*, Dalton Trans. 2008.
   Used by :func:`default_bond_specs` to estimate bond length ranges.

.. autofunction:: default_atom_style

.. autofunction:: default_bond_specs


Bond and polyhedra computation
------------------------------

.. autofunction:: compute_bonds

.. autofunction:: compute_polyhedra
