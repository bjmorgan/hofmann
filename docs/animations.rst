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
   scene.render_animation("output.gif", fps=10)

The output format is determined by the file extension — ``.gif`` for
GIF (requires ``imageio``) or ``.mp4`` for MP4 (requires
``imageio-ffmpeg``).  Install the optional dependencies with::

   pip install imageio imageio-ffmpeg


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
       fps=10,
       dpi=150,              # 150 DPI (default)
       figsize=(6.0, 6.0),   # 6 x 6 inches -> 900 x 900 px
   )


Example: CH4 vibration
----------------------

A minimal molecular example — a vibrating methane molecule rendered
with C-H bonds:

.. image:: _static/ch4_md.gif
   :width: 300px
   :align: center
   :alt: CH4 vibration animation

.. code-block:: python

   from ase.io import read
   from hofmann import AtomStyle, BondSpec, StructureScene

   traj = read("ch4_md.traj", index=":")

   bonds = [BondSpec(species=("C", "H"), max_length=1.3)]
   atom_styles = {
       "C": AtomStyle(radius=0.6, colour="dimgrey"),
       "H": AtomStyle(radius=0.35, colour="lightgrey"),
   }

   scene = StructureScene.from_ase(
       traj, bond_specs=bonds, atom_styles=atom_styles,
   )
   scene.render_animation("ch4_md.gif", fps=15, dpi=100, figsize=(4, 4))

See ``examples/generate_ch4_trajectory.py`` and
``examples/render_ch4_animation.py`` for the full runnable scripts.


Example: SrTiO3 perovskite MD
------------------------------

A periodic structure with TiO6 octahedral polyhedra.  Bond completion
ensures octahedra are drawn correctly across periodic boundaries:

.. image:: _static/srtio3_md.gif
   :width: 450px
   :align: center
   :alt: SrTiO3 perovskite MD animation

.. code-block:: python

   from ase.io import read
   from hofmann import (
       AtomStyle, BondSpec, PolyhedronSpec, StructureScene,
   )

   traj = read("srtio3_md.traj", index="::2")

   bonds = [
       BondSpec(species=("Ti", "O"), max_length=2.5, complete="*"),
       BondSpec(species=("Sr", "O"), max_length=3.2),
   ]
   polyhedra = [
       PolyhedronSpec(
           centre="Ti",
           colour="steelblue",
           alpha=0.4,
           hide_centre=True,
           hide_bonds=True,
           hide_vertices=True,
       ),
   ]
   atom_styles = {
       "Sr": AtomStyle(radius=1.2, colour="forestgreen"),
       "Ti": AtomStyle(radius=0.8, colour="steelblue"),
       "O": AtomStyle(radius=0.6, colour="firebrick"),
   }

   scene = StructureScene.from_ase(
       traj,
       bond_specs=bonds,
       polyhedra=polyhedra,
       atom_styles=atom_styles,
   )
   scene.render_animation(
       "srtio3_md.gif", fps=10, dpi=100, figsize=(6, 6),
       pbc_padding=1.0,
   )

See ``examples/generate_srtio3_trajectory.py`` and
``examples/render_srtio3_animation.py`` for the full runnable scripts.


Style keyword arguments
-----------------------

Any :class:`~hofmann.RenderStyle` field can be passed as a keyword
argument to ``render_animation()``, just as with ``render_mpl()``:

.. code-block:: python

   scene.render_animation(
       "output.gif",
       show_bonds=False,
       show_polyhedra=True,
       pbc_padding=2.0,
   )


Interactive trajectory viewing
------------------------------

Multi-frame scenes can also be explored interactively.  See
:doc:`interactive` for keyboard controls including frame navigation
(``[`` / ``]``), frame indicator (``f``), go-to-frame (``g``), and
step size (``s``).
