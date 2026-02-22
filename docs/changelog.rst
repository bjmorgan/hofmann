Changelog
=========

0.10.0
------

- **Species legend widget.**  New ``show_legend`` option on
  :class:`~hofmann.RenderStyle` draws a vertical column of coloured
  circles with species labels.  Customise placement, font size, circle
  sizing, spacing, and label gap via :class:`~hofmann.LegendStyle`.

- **Flexible legend circle sizing.**  ``LegendStyle.circle_radius``
  accepts three forms: a uniform float, a ``(min, max)`` tuple for
  proportional sizing based on ``AtomStyle.radius``, or a per-species
  dict for explicit control.

- **Standalone legend rendering.**
  :func:`~hofmann.rendering.static.render_legend` produces a
  tightly-cropped legend image without any structure, useful for
  composing figures in external tools.  Supports ``figsize`` for
  fixed output dimensions and ``transparent`` backgrounds.

- New ``LegendStyle.label_gap`` parameter controls the horizontal gap
  between legend circles and species labels (default 5.0 points).

0.9.0
-----

- **Render-time periodic boundary pipeline.**  Periodic boundary
  handling has moved from scene construction to render time.
  :func:`~hofmann.from_pymatgen` no longer expands image atoms
  at construction; instead, :class:`~hofmann.model.Bond` carries an
  ``image`` field recording which lattice translation the bond
  crosses, and a new ``RenderingSet`` pipeline materialises image
  atoms on demand during rendering.  This means the same scene can be
  rendered with different PBC settings without reconstruction.

  The pipeline applies five stages in order:

  1. Single-pass completion (``complete`` on :class:`~hofmann.BondSpec`)
  2. Recursive expansion (``recursive`` on :class:`~hofmann.BondSpec`)
  3. Geometric padding (``pbc_padding`` on :class:`~hofmann.RenderStyle`)
  4. Polyhedra vertex completion
  5. Molecule deduplication (``deduplicate_molecules`` on
     :class:`~hofmann.RenderStyle`)

  See :doc:`scenes` for details.

- **PBC options move to RenderStyle.**  ``pbc``, ``pbc_padding``,
  ``max_recursive_depth``, and ``deduplicate_molecules`` are now
  fields on :class:`~hofmann.RenderStyle` rather than
  :class:`~hofmann.StructureScene`, so rendering style is fully
  separated from structure data.

- **Two-tier periodic bond computation.**  Bond detection for periodic
  structures dispatches based on the inscribed sphere radius of the
  unit cell.  When all bond lengths are shorter than the inscribed
  sphere radius (the common case), the minimum image convention (MIC)
  gives an efficient single-pass computation.  When bond lengths are
  comparable to cell dimensions, all 27 image offsets are checked
  iteratively.  Both paths use O(n^2) peak memory.

- **Molecule deduplication heuristics.**  Recursive expansion can
  produce duplicate molecular fragments.  The deduplication stage
  detects extended structures (slabs, frameworks) that wrap the unit
  cell and protects them from removal, removes non-wrapped components
  whose source atoms are a subset of a wrapped component, and selects
  a canonical copy among remaining duplicates using an unwrapped
  fractional centre-of-mass tie-breaker.

- :class:`~hofmann.StructureScene` now validates that ``view`` is a
  :class:`~hofmann.ViewState` on assignment, with a helpful hint when
  a tuple from :func:`~hofmann.render_mpl_interactive` is
  accidentally assigned without unpacking.

- Passing ``None`` for a nullable style keyword argument (e.g.
  ``pbc_padding=None``) in :func:`~hofmann.render_mpl` now correctly
  passes through as an explicit override rather than being silently
  dropped.

0.8.0
-----

- **Restructured package layout.**  The monolithic ``model.py`` (1888
  lines) and ``render_mpl.py`` (2517 lines) have been split into three
  sub-packages organised by architectural layer:

  - ``hofmann.model`` -- data types (colour utilities, atom/bond/polyhedron
    specs, rendering styles, view state, and the scene container).
  - ``hofmann.construction`` -- scene building (file parsing, bond and
    polyhedra computation, element defaults, style I/O, and pymatgen/XBS
    scene constructors).
  - ``hofmann.rendering`` -- display (projection, bond geometry, cell edge
    rendering, the painter's algorithm, static output, and the interactive
    viewer).

  All public import paths are preserved: ``from hofmann import BondSpec``
  and ``from hofmann.model import BondSpec`` continue to work.

0.7.1
-----

- Avoid quadratic array growth in ``_merge_expansions`` during
  periodic boundary expansion.  Accepted image coordinates are now
  collected in a list with O(1) hash-based deduplication and
  concatenated once at the end, matching the approach already used by
  ``_expand_neighbour_shells``.

- :class:`~hofmann.StructureScene` now validates that every frame has
  the same number of atoms as the ``species`` list at construction
  time, raising :class:`ValueError` immediately instead of failing
  with a confusing error during rendering.

- :meth:`~hofmann.ViewState.look_along` now returns ``self``, enabling
  one-liner construction such as
  ``ViewState(centre=centroid).look_along([1, 1, 1])``.

- Passing ``None`` for a style keyword argument in
  :func:`~hofmann.render_mpl` now resets that field to the
  :class:`~hofmann.RenderStyle` class default instead of being silently
  ignored.

- :class:`~hofmann.BondSpec`, :class:`~hofmann.AtomStyle`,
  :class:`~hofmann.PolyhedronSpec`, and :class:`~hofmann.ViewState` now
  validate their numeric fields at construction time, raising
  :class:`ValueError` for out-of-range values (e.g. negative radii,
  ``min_length > max_length``, ``alpha`` outside ``[0, 1]``,
  non-positive ``zoom`` or ``view_distance``).

- :func:`~hofmann.render_mpl`, :func:`~hofmann.render_mpl_interactive`,
  :meth:`~hofmann.StructureScene.centre_on`, and
  :func:`~hofmann.from_pymatgen` now raise descriptive
  :class:`ValueError` messages for out-of-range index arguments
  (``frame_index``, ``atom_index``, ``centre_atom``) instead of
  leaking bare :class:`IndexError` exceptions.

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
