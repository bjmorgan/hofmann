Changelog
=========

0.7.0
-----

- :func:`~hofmann.from_pymatgen` (and the
  :meth:`~hofmann.StructureScene.from_pymatgen` classmethod) now
  accept ``atom_styles``, ``title``, ``view``, and ``atom_data``
  keyword arguments, allowing styles to be configured at construction
  time rather than requiring post-hoc mutation.

- New JSON-based style persistence.  All style classes gain
  ``to_dict()`` / ``from_dict()`` methods for serialisation, and
  module-level :func:`~hofmann.save_styles` /
  :func:`~hofmann.load_styles` functions write and read style files
  containing any combination of ``atom_styles``, ``bond_specs``,
  ``polyhedra``, and ``render_style`` sections.
  :meth:`~hofmann.StructureScene.save_styles` and
  :meth:`~hofmann.StructureScene.load_styles` provide convenience
  methods on the scene itself.  See :doc:`styles` for details.

- New :class:`~hofmann.StyleSet` dataclass returned by
  :func:`~hofmann.load_styles`.

0.6.0
-----

- :class:`~hofmann.BondSpec` now only requires ``species`` and
  ``max_length``.  ``min_length`` defaults to ``0.0``; ``radius`` and
  ``colour`` default to class-level values
  (``BondSpec.default_radius = 0.1``,
  ``BondSpec.default_colour = 0.5``) which can be changed to set
  project-wide defaults.  The ``repr()`` shows ``<default ...>`` for
  values that have not been explicitly set.

0.5.0
-----

- :meth:`~hofmann.StructureScene.render_mpl` now accepts an ``ax``
  parameter to render into an existing matplotlib axes, enabling
  multi-panel figures and composition with other plots.

0.4.0
-----

- Default bond detection now uses VESTA bond length cutoffs (bundled
  as JSON, sourced from `pymatgen <https://github.com/materialsproject/pymatgen>`_)
  instead of the covalent-radii-sum heuristic.  Self-bonds (e.g. C-C)
  are included automatically when present in the VESTA data.  The
  ``tolerance`` and ``self_bonds`` parameters on
  :func:`~hofmann.default_bond_specs` have been removed.

0.3.0
-----

- Bond completion across periodic boundaries.
  :class:`~hofmann.BondSpec` gains a ``complete`` flag for
  single-pass completion of bonds at cell boundaries, and a
  ``recursive`` flag for iterative search that follows chains of
  bonds across periodic images.  See :doc:`scenes` for details.

- :class:`~hofmann.AtomStyle` gains a ``visible`` flag (default
  ``True``).  Setting it to ``False`` hides atoms of that species
  and suppresses their bonds without removing them from the scene.

- ``BondSpec.complete`` now validates the species name against the
  bond spec's species pair, catching typos that previously resulted
  in a silent no-op.

- Removed ``PolyhedraVertexMode`` enum and the
  ``polyhedra_vertex_mode`` field on :class:`~hofmann.RenderStyle`.
  Vertex atoms are now always drawn in front of their connected
  polyhedral faces (the previous default behaviour).

0.2.2
-----

- Add dodecahedron project logo to README and repository.

0.2.1
-----

- Static renderer now uses per-axis viewport extents, producing
  tightly cropped output for non-square scenes.

0.2.0
-----

- Per-atom metadata colouring via colourmaps.  Use
  :meth:`~hofmann.StructureScene.set_atom_data` and the ``colour_by``
  parameter on :meth:`~hofmann.StructureScene.render_mpl` to map
  numerical or categorical data to atom colours.
- Multiple ``colour_by`` layers with priority merging.  Pass a list
  of keys to apply different colouring rules to different atom
  subsets; the first non-missing value wins for each atom.
- Polyhedra without an explicit colour now inherit the resolved
  ``colour_by`` colour of their centre atom.
- New public API: :func:`~hofmann.resolve_atom_colours` for
  programmatic colour resolution, and :data:`~hofmann.CmapSpec` type
  alias for colourmap specifications.

0.1.0
-----

Initial release.

- Ball-and-stick rendering via matplotlib with depth-sorted painter's
  algorithm.
- Publication-quality vector output (SVG, PDF).
- XBS ``.bs`` and ``.mv`` file format support.
- pymatgen ``Structure`` interoperability (optional dependency).
- Periodic boundary condition expansion with bond-aware and
  polyhedra-vertex-aware image generation.
- Coordination polyhedra with configurable slab clipping modes.
- Interactive matplotlib viewer with mouse rotation and zoom.
