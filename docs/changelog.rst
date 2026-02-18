Changelog
=========

0.4.0
-----

- Default bond detection now uses VESTA bond length cutoffs instead
  of the covalent-radii-sum heuristic.  Self-bonds (e.g. C-C) are
  included automatically when present in the VESTA data.  The
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
