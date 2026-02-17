Changelog
=========

0.1.1
-----

- Per-atom metadata colouring via colourmaps.  Use
  :meth:`~hofmann.StructureScene.set_atom_data` and the ``colour_by``
  parameter on :meth:`~hofmann.StructureScene.render_mpl` to map
  numerical or categorical data to atom colours.
- Multiple ``colour_by`` layers with priority merging.  Pass a list
  of keys to apply different colouring rules to different atom
  subsets; the first non-missing value wins for each atom.

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
