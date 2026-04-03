Animations
==========

hofmann can render multi-frame trajectories as GIF or MP4 animations
using :meth:`~hofmann.StructureScene.render_animation`.  The same
per-frame rendering pipeline used for static output is applied to each
frame, so all visual options (bonds, polyhedra, colouring, styles)
work identically.


Basic usage
-----------

Build a scene from a trajectory and call ``render_animation()``:

.. code-block:: python

   from ase.io import read
   from hofmann import StructureScene

   traj = read("trajectory.traj", index=":")
   scene = StructureScene.from_ase(traj)
   scene.render_animation("output.gif")

The output format is determined by the file extension — ``.gif`` for
GIF or ``.mp4`` for MP4.  Animation rendering requires the
``animation`` optional extra::

   pip install "hofmann[animation]"

The examples on this page use ASE for trajectory loading, which
requires the ``ase`` extra::

   pip install "hofmann[ase]"


Frame selection
---------------

By default all frames are rendered.  Use the ``frames`` parameter to
select a subset:

.. code-block:: python

   # Every 5th frame:
   scene.render_animation("output.gif", frames=range(0, 100, 5))

   # Specific frames:
   scene.render_animation("output.gif", frames=[0, 10, 20, 50])


Resolution and size
-------------------

Control the output resolution with ``dpi`` and ``figsize``:

.. code-block:: python

   scene.render_animation(
       "output.gif",
       fps=10,               # default: 30
       dpi=100,              # default: 150
       figsize=(6.0, 6.0),   # default: (5.0, 5.0)
   )


Example: CH\ :sub:`4` vibration
-------------------------------

A vibrating methane molecule styled to match the
:doc:`Getting Started <getting-started>` example:

.. code-block:: python

   from ase.io import read
   from hofmann import AtomStyle, BondSpec, StructureScene

   traj = read("ch4_md.traj", index=":")

   bonds = [BondSpec(species=("C", "H"), max_length=1.5, radius=0.055, colour=1.0)]
   atom_styles = {
       "C": AtomStyle(radius=0.5, colour=0.7),
       "H": AtomStyle(radius=0.35, colour=1.0),
   }

   scene = StructureScene.from_ase(traj, bond_specs=bonds, atom_styles=atom_styles)
   scene.render_animation("ch4_md.gif", fps=15, dpi=100, figsize=(4, 4))

.. image:: _static/ch4_md.gif
   :width: 300px
   :align: center
   :alt: CH4 vibration animation


Example: SrTiO\ :sub:`3` perovskite MD
---------------------------------------

A single octahedral layer from a 4x4x4 SrTiO\ :sub:`3` supercell at 1000 K.
The trajectory has been pre-filtered to one Sr plane, one Ti plane,
and their coordinating O.  The ``hide_vertices`` option on the
polyhedron spec hides the oxygen atoms to emphasise the polyhedral
network:

.. code-block:: python

   from ase.io import read
   from hofmann import (
       AtomStyle, BondSpec, PolyhedronSpec, StructureScene,
   )

   traj = read("srtio3_md.traj", index=":")

   bonds = [BondSpec(species=("Ti", "O"), max_length=2.5)]
   polyhedra = [
       PolyhedronSpec(
           centre="Ti", colour="steelblue", alpha=0.4,
           hide_centre=True, hide_bonds=True, hide_vertices=True,
       ),
   ]
   atom_styles = {
       "Sr": AtomStyle(radius=1.2, colour="forestgreen"),
   }

   scene = StructureScene.from_ase(
       traj, bond_specs=bonds, polyhedra=polyhedra,
       atom_styles=atom_styles,
   )
   scene.render_animation(
       "srtio3_md.gif", fps=10, dpi=100, figsize=(6, 6),
       show_axes=False, show_cell=False,
       pbc_padding=1.0,
   )

.. image:: _static/srtio3_md.gif
   :width: 450px
   :align: center
   :alt: SrTiO3 perovskite MD animation


Style keyword arguments
-----------------------

Any :class:`~hofmann.RenderStyle` field can be passed as a keyword
argument to ``render_animation()``, just as with ``render_mpl()``:

.. code-block:: python

   scene.render_animation(
       "output.gif",
       show_bonds=False,
       show_polyhedra=True,
       pbc_padding=1.0,
   )


Interactive trajectory viewing
------------------------------

Multi-frame scenes can also be explored interactively.  See
:doc:`interactive` for keyboard controls including frame navigation
(``[`` / ``]``, ``{`` / ``}``), frame indicator (``f``),
go-to-frame (``g``), and step size (``s``).
