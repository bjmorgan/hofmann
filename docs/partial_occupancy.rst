.. _partial_occupancy:

Partial and mixed occupancy
============================

Hofmann can render sites that are partially occupied (a fraction of an
atom) or shared between multiple species (a solid solution).  These
sites appear as VESTA-style pie wedges, one wedge per constituent
species, with wedge angles proportional to occupancy.

Constructing a mixed site
--------------------------

Mixed sites are expressed by passing a :class:`Composition` value in
the ``species`` list, in place of a plain string label::

    import numpy as np
    from hofmann import Composition, StructureScene, Frame

    fe_mn = Composition({"Fe": 0.7, "Mn": 0.3})
    scene = StructureScene(
        species=["Fe", fe_mn, fe_mn, "O"],
        frames=[Frame(coords=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]))],
    )

Reusing a single ``Composition`` value across many rows is the
recommended way to keep authoring concise.

Vacancies
---------

A :class:`Composition` whose occupancies sum to less than one carries
an implicit vacancy fraction::

    fe_partial = Composition({"Fe": 0.7})  # 70% Fe + 30% vacancy

The vacancy fraction renders as a gap by default (canvas background
shows through).  Set
:attr:`~hofmann.RenderStyle.vacancy_colour` to fill the gap with an
explicit colour.

Loading from pymatgen
---------------------

:func:`~hofmann.from_pymatgen` propagates partial occupancies
directly: any :class:`pymatgen.core.PeriodicSite` whose ``species``
mapping has more than one entry, or a single entry at occupancy less
than one, becomes a :class:`Composition` in the resulting scene.  No
preprocessing or manual handling is required.

Customising appearance
----------------------

By default a mixed site is drawn at a radius equal to the occupancy-
weighted average of its constituents' :attr:`~hofmann.AtomStyle.radius`,
with each wedge using its species' :attr:`~hofmann.AtomStyle.colour`.
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
- :attr:`~hofmann.RenderStyle.vacancy_colour` fills the vacancy gap.

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
