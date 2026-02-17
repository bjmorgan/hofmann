Scenes and structures
=====================

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

Polyhedra can also inherit per-atom colours from ``colour_by``
data attached to their centre atoms.  See :doc:`colouring` for
details on per-atom colouring, custom colouring functions, and
polyhedra colour inheritance.
