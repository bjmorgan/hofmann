XBS file format
===============

hofmann can read the ``.bs`` and ``.mv`` file formats from the original
XBS_ ball-and-stick viewer by Methfessel (1995).

.. _XBS: https://www.ccl.net/cca/software/X-WINDOW/xbs/


The ``.bs`` format
------------------

A ``.bs`` file defines atoms, species styles, and bond rules as a
sequence of keyword lines.  Blank lines and lines starting with ``*``
are comments.

``atom``
~~~~~~~~

Defines an atom position::

   atom  <species>  <x>  <y>  <z>

Example::

   atom  C   0.000  0.000  0.000
   atom  H   1.155  1.155  1.155

``spec``
~~~~~~~~

Sets the display radius and colour for a species::

   spec  <species>  <radius>  <colour>

Colour can be:

- A single float for grey (``0.0`` = black, ``1.0`` = white)
- Three floats for RGB (each in ``[0, 1]``)
- A CSS colour name

Example::

   spec  C  1.000  0.7
   spec  H  0.700  1.00
   spec  O  0.900  1.0  0.0  0.0

``bonds``
~~~~~~~~~

Declares a bond detection rule between two species::

   bonds  <sp1>  <sp2>  <min_len>  <max_len>  <radius>  <colour>

Bonds are detected between all atom pairs of the given species whose
distance falls within ``[min_len, max_len]``.

Example::

   bonds  C  H  0.000  3.400  0.109  1.00
   bonds  H  H  0.000  2.800  0.109  1.00

``polyhedra``
~~~~~~~~~~~~~

Declares a coordination polyhedron rule::

   polyhedra  <centre_sp>  <vertex_sp>  <max_dist>  <colour>  [alpha]  [edge_colour]  [edge_width]

Example::

   polyhedra  Ti  O  2.5  0.5 0.7 1.0  0.3


Complete example (CH4)
~~~~~~~~~~~~~~~~~~~~~~

::

   atom      C       0.000      0.000      0.000
   atom      H       1.155      1.155      1.155
   atom      H      -1.155     -1.155      1.155
   atom      H       1.155     -1.155     -1.155
   atom      H      -1.155      1.155     -1.155

   spec      C      1.000   0.7
   spec      H      0.700   1.00

   bonds     C     H    0.000    3.400    0.109   1.00
   bonds     H     H    0.000    2.800    0.109   1.00


The ``.mv`` format
------------------

The ``.mv`` format stores multi-frame trajectories (e.g. from molecular
dynamics).  Each frame is introduced by a ``frame`` line, followed by
coordinates for all atoms listed sequentially:

::

   frame <label>
   <x1> <y1> <z1> <x2> <y2> <z2> ...

Numbers may be split across lines.  Lines starting with ``*`` and blank
lines are comments.

Example::

   frame step_0
   0.0 0.0 0.0  1.155 1.155 1.155  -1.155 -1.155 1.155
   1.155 -1.155 -1.155  -1.155 1.155 -1.155

   frame step_1
   0.01 0.0 0.0  1.16 1.16 1.16  -1.16 -1.16 1.16
   1.16 -1.16 -1.16  -1.16 1.16 -1.16
