.. _partial_occupancy:

Partial and mixed occupancy
============================

Some crystal structures have sites whose atomic identity is not
fixed across the lattice -- either a species occupies a site in only
some fraction of unit cells (partial occupancy), or several species
share the site (a solid solution).  Each individual site still
contains at most one atom; what varies is which atom, or whether one
is present at all, from one unit cell to the next.  Hofmann draws
such sites as pie wedges -- one wedge per species, with each wedge's
angle equal to that species' crystallographic occupancy.

.. image:: _static/partial_occupancy.svg
   :width: 400px
   :align: center
   :alt: TiOF2 unit cell with disordered O/F anion sites rendered as wedges

The figure above shows TiOF\ :sub:`2`, which adopts the cubic
ReO\ :sub:`3` structure with Ti at the cell corners and one anion
site at each edge midpoint.  Across the crystal, two-thirds of these
positions hold an F atom and one-third hold an O.  The wedges show
that average: an individual site contains one atom, but the rendering
draws each site as a 2/3 F, 1/3 O mix.

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

When several sites share the same composition -- as the three anion
sites do above -- define the :class:`Composition` once and pass the
same value at each position, rather than constructing a fresh
identical mix every time.

Loading from a CIF file
-----------------------

CIF files express mixed sites by listing each species on its own row
at the same fractional position, with the row's occupancy giving that
species' fraction.  pymatgen reads these directly, and
:func:`~hofmann.from_pymatgen` turns each such site into a
:class:`Composition` in the resulting scene -- you don't need to
preprocess the structure.

Here is a minimal example -- a single Fe / Mn site filling a 2.8 Å
cubic cell:

.. literalinclude:: ../examples/disordered_site.cif
   :language: text
   :caption: ``disordered_site.cif``

Load it and render:

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

The same conversion applies to any pymatgen :class:`~pymatgen.core.Structure`,
whether you read it from a CIF or built it in code.  The rule is
simple: any site with more than one species, or with a single species
at occupancy below one, becomes a :class:`Composition`.  Everything
else stays as a plain species label.

Vacancies
---------

When a :class:`Composition`'s occupancies sum to less than one, the
missing fraction is treated as a vacancy::

    fe_partial = Composition({"Fe": 0.7})  # 70% Fe, 30% vacancy

The vacancy is drawn as an opaque wedge filled with the canvas
background colour, so the site still reads as a solid circle with a
"missing" slice rather than a hole.  Override the fill with
:attr:`~hofmann.RenderStyle.vacancy_colour` -- for example,
``"lightgrey"`` on a white canvas if you want vacancies to be
visible at a glance.

Customising appearance
----------------------

Each wedge takes its colour from the species'
:attr:`~hofmann.AtomStyle.colour`.  The radius of the whole site is
the occupancy-weighted average of its constituents' radii, with the
average computed over the occupied species only -- so a half-vacant
site is drawn at the same size as a fully occupied one, not half the
size.

For more specific styling -- for example, colouring all mixed sites
the same way to flag disordered positions -- attach per-site data
with :meth:`~hofmann.StructureScene.set_atom_data` and use the
``colour_by`` parameter when rendering.  This is the same workflow
described in :doc:`colouring`; mixed sites participate just like
single-species sites.

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

Visibility of constituent species
---------------------------------

Setting :attr:`~hofmann.AtomStyle.visible` to ``False`` hides every
site whose species is given as a plain label.  It does **not** hide
that species when it appears as a constituent of a
:class:`Composition`: a mixed site is always drawn with all of its
constituents, regardless of any per-species ``visible`` flag.

This is deliberate.  The constituents of a mixed site still
participate in bond rules and polyhedron rules, and rendering only
some of the wedges would leave the visible site out of step with the
bonds drawn around it.  To de-emphasise specific mixed sites, attach
per-site data with :meth:`~hofmann.StructureScene.set_atom_data` and
recolour them via ``colour_by``.

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
