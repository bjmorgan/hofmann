Styles and presets
==================

hofmann stores visual appearance settings in a handful of dataclasses --
:class:`~hofmann.AtomStyle`, :class:`~hofmann.BondSpec`,
:class:`~hofmann.PolyhedronSpec`, and :class:`~hofmann.RenderStyle`.
This page describes how to save and reuse styles, how to customise
them at construction time, and the underlying JSON format.


Saving and loading styles
--------------------------

The simplest way to save styles is directly from a scene.  Once you
have a scene whose atom colours, bond rules, and polyhedra look the
way you want, save everything to a JSON file:

.. code-block:: python

   # Save the scene's current styles to a file.
   scene.save_styles("my_styles.json")

To reuse those styles in another session or on a different structure,
load them back:

.. code-block:: python

   # Apply saved styles to a new scene.
   other_scene.load_styles("my_styles.json")

:meth:`~hofmann.StructureScene.load_styles` merges atom styles
(existing species keep their styles unless overridden by the file) and
replaces bond specs and polyhedra entirely.  If the file contains a
``render_style`` section it is ignored -- use
:func:`~hofmann.load_styles` to retrieve it separately and pass it to
the renderer.

Choosing what to save
~~~~~~~~~~~~~~~~~~~~~

The module-level :func:`~hofmann.save_styles` function gives you full
control over which sections are included.  You can save a subset of
a scene's styles, or include render style (which the scene convenience
method does not):

.. code-block:: python

   from hofmann import save_styles, RenderStyle

   # Save only atom styles from the current scene.
   save_styles("atom_colours.json", atom_styles=scene.atom_styles)

   # Save everything including a render style.
   save_styles(
       "full_preset.json",
       atom_styles=scene.atom_styles,
       bond_specs=scene.bond_specs,
       polyhedra=scene.polyhedra,
       render_style=RenderStyle(atom_scale=0.8, show_outlines=False),
   )

In other words, ``scene.save_styles("out.json")`` is shorthand for
``save_styles("out.json", atom_styles=scene.atom_styles,
bond_specs=scene.bond_specs, polyhedra=scene.polyhedra)``.  It does
not include ``render_style`` because render style is passed to the
renderer, not stored on the scene.

Loading with ``load_styles``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module-level :func:`~hofmann.load_styles` returns a
:class:`~hofmann.StyleSet` whose fields are ``None`` for sections not
present in the file.  You can then pick out whichever parts you need:

.. code-block:: python

   from hofmann import load_styles

   styles = load_styles("my_styles.json")
   styles.atom_styles    # dict or None
   styles.bond_specs     # list or None
   styles.polyhedra      # list or None
   styles.render_style   # RenderStyle or None

This is useful when you want to combine loaded styles with a new scene
or pass the render style to the renderer separately:

.. code-block:: python

   styles = load_styles("my_styles.json")
   scene = StructureScene.from_pymatgen(
       structure, bonds,
       atom_styles=styles.atom_styles,
   )
   scene.render_mpl("output.svg", style=styles.render_style)

See also :ref:`construction-time-styles` in the scenes page for
passing styles directly to :func:`~hofmann.from_pymatgen`.


Serialising individual styles
------------------------------

Every style class has ``to_dict()`` and ``from_dict()`` methods for
converting to and from plain Python dictionaries.  These are the
building blocks used by ``save_styles`` / ``load_styles`` internally,
but they are also useful on their own for programmatic manipulation:

.. code-block:: python

   from hofmann import AtomStyle, BondSpec

   style = AtomStyle(radius=1.4, colour=(0.5, 1.0, 0.5))
   d = style.to_dict()
   # {'radius': 1.4, 'colour': [0.5, 1.0, 0.5]}

   restored = AtomStyle.from_dict(d)

   spec = BondSpec(species=("Ti", "O"), max_length=2.5, radius=0.12)
   d = spec.to_dict()
   # {'species': ['O', 'Ti'], 'max_length': 2.5, 'radius': 0.12}

   restored = BondSpec.from_dict(d)

Fields at their default values are omitted from the output to keep
dictionaries compact.  For :class:`~hofmann.BondSpec`, ``radius`` and
``colour`` are only included when explicitly set (not when using the
class-level default).

The following classes support ``to_dict()`` / ``from_dict()``:

- :class:`~hofmann.AtomStyle`
- :class:`~hofmann.BondSpec`
- :class:`~hofmann.PolyhedronSpec`
- :class:`~hofmann.RenderStyle` (including nested
  :class:`~hofmann.CellEdgeStyle` and :class:`~hofmann.AxesStyle`)
- :class:`~hofmann.CellEdgeStyle`
- :class:`~hofmann.AxesStyle`


.. _style-json-format:

Style JSON file format
-----------------------

Style files are plain JSON with up to four optional top-level
sections.  A file can contain any combination -- just
``atom_styles``, just ``render_style``, or all four:

.. code-block:: json

   {
     "atom_styles": {
       "Zr": {"radius": 1.4, "colour": [0.5, 1.0, 0.5]},
       "O":  {"radius": 0.8, "colour": [1.0, 0.0, 0.0]}
     },
     "bond_specs": [
       {
         "species": ["O", "Ti"],
         "max_length": 2.5,
         "radius": 0.12,
         "colour": [0.4, 0.4, 0.4]
       }
     ],
     "polyhedra": [
       {"centre": "Ti", "alpha": 0.3}
     ],
     "render_style": {
       "atom_scale": 0.8,
       "show_outlines": false,
       "half_bonds": true
     }
   }

Section reference
~~~~~~~~~~~~~~~~~

``atom_styles``
   A mapping from species label to an object with:

   - ``radius`` (float, required) -- display radius in angstroms.
   - ``colour`` (required) -- CSS name, hex string, grey float, or
     ``[r, g, b]`` list.
   - ``visible`` (bool, optional) -- omit or set ``true`` for the
     default; set ``false`` to hide atoms.

``bond_specs``
   A list of objects, each with:

   - ``species`` (list of two strings, required) -- species pair.
   - ``max_length`` (float, required) -- maximum bond length threshold.
   - ``min_length`` (float, optional) -- minimum bond length (default
     ``0.0``).
   - ``radius`` (float, optional) -- bond cylinder radius.  Omit to
     use the class default.
   - ``colour`` (optional) -- bond colour.  Omit to use the class
     default.
   - ``complete`` (string or ``false``, optional) -- bond completion
     mode.
   - ``recursive`` (bool, optional) -- recursive bond search.

``polyhedra``
   A list of objects, each with:

   - ``centre`` (string, required) -- species pattern for the centre
     atom.
   - ``colour`` (optional) -- face colour, or omit to inherit from
     the centre atom.
   - ``alpha`` (float, optional) -- face transparency (default
     ``0.4``).
   - ``edge_colour`` (optional) -- wireframe edge colour.
   - ``edge_width`` (float, optional) -- wireframe edge width.
   - ``hide_centre``, ``hide_bonds``, ``hide_vertices`` (bool,
     optional) -- visibility flags.
   - ``min_vertices`` (int or ``null``, optional) -- minimum vertex
     count.

``render_style``
   An object with any :class:`~hofmann.RenderStyle` field as a key.
   Nested ``cell_style`` and ``axes_style`` are sub-objects with their
   respective fields.  Enum values are serialised as strings (e.g.
   ``"per_face"``).  See the :class:`~hofmann.RenderStyle` API docs
   for the full list of fields.
