.. _partial_occupancy:

Partial and mixed occupancy
============================

Hofmann can render sites that are partially occupied (a fraction of an
atom) or shared between multiple species (a solid solution).  These
sites appear as VESTA-style pie wedges, one wedge per constituent
species, with wedge angles proportional to occupancy.

.. image:: _static/partial_occupancy.svg
   :width: 400px
   :align: center
   :alt: TiOF2 unit cell with disordered O/F anion sites rendered as wedges

The figure above shows TiOF\ :sub:`2`, which adopts the cubic
ReO\ :sub:`3` structure with Ti at the cell corners and a single anion
site at each edge midpoint.  Every anion site is occupied by O and F
in a 2:1 ratio, so each appears as a two-wedge mixed site.

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

CIF files express mixed sites by listing each species at the same
fractional position with its own occupancy.  pymatgen reads these
files directly and :func:`~hofmann.from_pymatgen` propagates the
occupancies through to a :class:`Composition`-bearing scene with no
manual processing.

Here is the smallest possible example -- a single Fe / Mn site
filling a 2.8 Å cubic cell:

.. literalinclude:: ../examples/disordered_site.cif
   :language: text
   :caption: ``disordered_site.cif``

Loading and rendering takes two lines:

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

The same pattern applies to any pymatgen :class:`~pymatgen.core.Structure`,
whether it was read from a CIF file or built programmatically: any
:class:`pymatgen.core.PeriodicSite` whose ``species`` mapping has
more than one entry, or a single entry at occupancy less than one,
becomes a :class:`Composition` in the resulting scene.

Vacancies
---------

A :class:`Composition` whose occupancies sum to less than one carries
an implicit vacancy fraction::

    fe_partial = Composition({"Fe": 0.7})  # 70% Fe + 30% vacancy

The vacancy fraction renders as an opaque wedge filled with the
canvas background colour by default, so partial sites still read as
solid circles with a "missing" slice.  Set
:attr:`~hofmann.RenderStyle.vacancy_colour` to override with an
explicit colour (for example, ``"lightgrey"`` on a white canvas to
make the vacancy stand out).

Customising appearance
----------------------

By default a mixed site is drawn at a radius equal to the
occupancy-weighted average of its constituents'
:attr:`~hofmann.AtomStyle.radius`, normalised by the total species
occupancy so that vacancy fractions do not shrink the site.  Each
wedge uses its species' :attr:`~hofmann.AtomStyle.colour`.
For more specific styling — for example, colouring all mixed sites the
same way to highlight disordered positions — use
:meth:`~hofmann.StructureScene.set_atom_data` and the ``colour_by``
parameter of the render methods, exactly as for pure sites.

The wedge layout itself is controlled by three render-style fields:

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

:attr:`~hofmann.AtomStyle.visible` is a per-species flag — setting it
to ``False`` hides every pure-string site of that species.  It does
**not** apply to constituents of a :class:`Composition`: a mixed site
is always drawn with all of its constituents, regardless of any
constituent's ``visible`` flag.  This avoids the visually inconsistent
state where a constituent is hidden in the wedge rendering but its
species still attracts bonds and matches rule lookups.  To
de-emphasise specific mixed sites, use
:meth:`~hofmann.StructureScene.set_atom_data` with ``colour_by`` to
recolour them at the row level.

Bonding and polyhedra
---------------------

:class:`BondSpec` and :class:`PolyhedronSpec` rules fire on a mixed
site whenever any constituent species satisfies the rule.  A 70 / 30
Fe / Mn site with both ``("Fe", "O")`` and ``("Mn", "O")`` bond rules
defined produces exactly one bond per neighbour, drawn at whichever
matching cutoff is permissive enough.  Vacancy fractions never
participate in bonding.

The half-bond at a mixed-site end uses the dominant species'
colour — the species with the highest occupancy in the composition,
with alphabetical tiebreak.  The wedges, not the bond, communicate the
full composition.
