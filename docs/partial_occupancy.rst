.. _partial_occupancy:

Partial and mixed occupancy
============================

.. image:: _static/partial_occupancy.svg
 :width: 400px
 :align: center
 :alt: TiOF2 unit cell with disordered O/F anion sites rendered as wedges

Hofmann can render sites with partial or mixed crystallographic
occupancy.  Such *mixed sites* are drawn as pie wedges; one wedge
per species, with each wedge's angle equal to that species'
occupancy.

Constructing a mixed site
--------------------------

Mixed sites are expressed by passing a :class:`Composition` value in
the ``species`` list, in place of a plain string label::

    import numpy as np
    from hofmann import Composition, StructureScene, Frame

    anion = Composition({"O": 2 / 3, "F": 1 / 3})
    scene = StructureScene(
        species=["Ti", anion, anion, anion],
        frames=[Frame(coords=np.array([
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [0.0, 1.9, 0.0],
            [0.0, 0.0, 1.9],
        ]))],
    )

Vacancies
---------

When a :class:`Composition`'s occupancies sum to less than one, the
missing fraction is treated as a vacancy fraction::

    fe_partial = Composition({"Fe": 0.7})  # 70% Fe, 30% vacancy

.. image:: _static/partial_occupancy_vacancy.svg
   :width: 220px
   :align: center
   :alt: Single Fe site at 70 percent occupancy with a vacancy wedge

The vacancy wedge is filled with the canvas background colour by
default.  A custom vacancy colour can be set by passing the
:attr:`~hofmann.RenderStyle.vacancy_colour` field on
:class:`~hofmann.RenderStyle`.

From a pymatgen Structure
-------------------------

A pymatgen :class:`~pymatgen.core.Structure` represents partial
occupancy and species disorder natively.
:func:`~hofmann.from_pymatgen` reads this directly: any site with
more than one species, or with a single species at occupancy below
one, becomes a :class:`Composition` in the resulting scene.

.. code-block:: python

   from hofmann import StructureScene

   scene = StructureScene.from_pymatgen(structure)
   scene.render_mpl("output.svg")

Loading from a CIF file
-----------------------

CIFs can be loaded with pymatgen and passed through to
:func:`~hofmann.from_pymatgen`.  For example:

.. literalinclude:: ../examples/disordered_site.cif
   :language: text

.. code-block:: python

   from pymatgen.core import Structure
   from hofmann import StructureScene

   structure = Structure.from_file("disordered_site.cif")
   scene = StructureScene.from_pymatgen(structure)
   scene.render_mpl("disordered.svg")

.. image:: _static/partial_occupancy_minimal.svg
   :width: 220px
   :align: center
   :alt: Single mixed-occupancy Fe/Mn site rendered as a two-wedge atom

Customising appearance
----------------------

Each wedge takes its colour from the corresponding species'
:attr:`~hofmann.AtomStyle.colour`.  The radius of the whole site is
the occupancy-weighted average of its constituents' radii, with the
average computed over the occupied species only -- so a half-vacant
site is drawn at the same size as a fully occupied one.

Three :class:`~hofmann.RenderStyle` fields control the wedge layout:

- :attr:`~hofmann.RenderStyle.wedge_start_angle` sets the starting
  orientation (default 12 o'clock).
- :attr:`~hofmann.RenderStyle.show_wedge_edges` toggles radial edges
  between wedges (default on; set to ``False`` to stroke only the
  outer arc).
- :attr:`~hofmann.RenderStyle.vacancy_colour` overrides the vacancy
  fill (default: canvas background colour).

For example::

    from hofmann import RenderStyle

    style = RenderStyle(
        wedge_start_angle=0.0,        # start at 3 o'clock
        show_wedge_edges=False,       # outer arc only, no radial edges
        vacancy_colour="lightgrey",   # explicit vacancy fill
    )
    scene.render_mpl("structure.svg", style=style)

For more specific styling -- for example, colouring all mixed sites
the same way to flag disordered positions -- attach per-site data
with :meth:`~hofmann.StructureScene.set_atom_data` and use the
``colour_by`` parameter when rendering.  See :doc:`colouring`.

Visibility of constituent species
---------------------------------

Setting :attr:`~hofmann.AtomStyle.visible` to ``False`` hides every
site whose species is given as a plain label.  It does **not** hide
that species when it appears as a constituent of a
:class:`Composition`: a mixed site is always drawn with all of its
constituents, regardless of any per-species ``visible`` flag.

Bonding and polyhedra
---------------------

A :class:`BondSpec` or :class:`PolyhedronSpec` rule applies to a
mixed site whenever any of its constituent species matches the rule.
Each pair of atoms still produces at most one bond, even when several
rules match: for a 70 / 30 Fe / Mn site bonded to an O neighbour,
defining both ``("Fe", "O")`` and ``("Mn", "O")`` rules gives one
bond, not two.  Vacancies never form bonds.

The half-bond at the mixed-site end is drawn in the colour of the
dominant species -- the constituent with the highest occupancy, with
ties broken alphabetically.
