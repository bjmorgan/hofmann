Changelog
=========

0.13.0
------

- **Breaking:** :class:`~hofmann.LegendItem` is now an abstract base class.
  Use the concrete subclasses :class:`~hofmann.AtomLegendItem` (circle
  markers), :class:`~hofmann.PolygonLegendItem` (regular-polygon
  markers with ``sides`` and ``rotation``), or
  :class:`~hofmann.PolyhedronLegendItem` (miniature 3D icons with
  ``shape`` and optional ``rotation``).

  Migration:

  - ``LegendItem(key=..., colour=...)`` becomes
    ``AtomLegendItem(key=..., colour=...)``.
  - ``LegendItem(key=..., colour=..., sides=6)`` becomes
    ``PolygonLegendItem(key=..., colour=..., sides=6)``.
  - ``LegendItem(key=..., colour=..., polyhedron="octahedron")``
    becomes
    ``PolyhedronLegendItem(key=..., colour=..., shape="octahedron")``.
  - ``LegendItem.from_polyhedron_spec(...)`` becomes
    ``PolyhedronLegendItem.from_polyhedron_spec(...)``.

  Serialisation: ``to_dict()`` now includes a ``"type"`` discriminator;
  ``LegendItem.from_dict()`` dispatches to the correct subclass.
  Saved style files from 0.12.x that contain polygon or polyhedron
  legend items must be re-saved; only plain atom-style dicts (no
  ``sides``/``polyhedron`` fields) are handled without a ``"type"`` key.

- :class:`~hofmann.PolyhedronLegendItem` gains a ``rotation``
  parameter accepting a ``(3, 3)`` rotation matrix or an ``(Rx, Ry)``
  tuple of angles in degrees.  When ``None`` (the default), the
  standard oblique legend viewing angle is used.

- Fix legend edge width scaling: legend marker outlines were
  incorrectly multiplied by the widget display-space scaling factor,
  making them appear thicker than the corresponding edges in the scene.

0.12.0
------

- :class:`~hofmann.LegendItem` gains a ``polyhedron`` field.  Set it to
  ``"octahedron"``, ``"tetrahedron"``, or ``"cuboctahedron"`` to render
  a miniature 3D-shaded polyhedron icon in the legend instead of a flat
  circle or polygon marker.  Polyhedron icons default to twice the
  flat-marker radius so that the 3D shading is legible at typical figure
  sizes.

- :class:`~hofmann.LegendItem` gains per-item ``edge_colour`` and
  ``edge_width`` fields.  When set, these override the scene-level
  outline settings; when unset, items fall back to the scene's
  ``outline_colour`` and ``outline_width``.  Setting
  ``show_outlines=False`` disables edges only for items that do not
  define their own edge styling.

- New :meth:`~hofmann.LegendItem.from_polyhedron_spec` classmethod
  creates a legend item from a :class:`~hofmann.PolyhedronSpec`,
  inheriting colour, alpha, and edge settings without duplication.

- :func:`~hofmann.rendering.static.render_legend` gains a
  ``polyhedra_shading`` parameter controlling the shading strength
  of 3D polyhedron icons (0 = flat, 1 = full).

- Fix bounding box computation in :func:`~hofmann.rendering.static.render_legend`
  to account for edge linewidth, preventing clipped outlines on
  tightly-cropped legend images.

0.11.1
------

- Internal: extract legend rendering from ``painter.py`` into a dedicated
  ``legend.py`` module, and move the shared widget scaling constant into
  ``_widget_scale.py``.

0.11.0
------

- Internal: legend drawing now runs through :class:`~hofmann.LegendItem`
  objects.  A new ``_build_legend_items`` helper assembles items from the
  scene's species and atom styles, and ``_draw_legend_widget`` consumes
  the resulting list.

- New :class:`~hofmann.LegendItem` class bundles per-entry legend data
  (key, colour, optional label, optional radius) with validated property
  setters following the :class:`~hofmann.BondSpec` pattern.

- :class:`~hofmann.LegendStyle` gains an ``items`` parameter.  Pass a
  tuple of :class:`~hofmann.LegendItem` instances to display a fully
  custom legend (e.g. for ``colour_by`` data) instead of the default
  species-based entries.

- :class:`~hofmann.LegendItem` supports regular-polygon markers via
  ``sides`` and ``rotation`` fields.  Set ``sides`` (>= 3) to draw
  a polygon instead of a circle, and ``rotation`` to rotate it in
  degrees.  Useful for indicating polyhedra types in the legend.

- :class:`~hofmann.LegendItem` gains a ``gap_after`` field for
  non-uniform vertical spacing.  Each item can override the
  style-level ``spacing`` for the gap below it; ``None`` falls back
  to ``LegendStyle.spacing``.

- :class:`~hofmann.LegendItem` gains an ``alpha`` field (0.0--1.0,
  default 1.0) for semi-transparent marker faces.  Marker outlines
  remain fully opaque, matching the visual style of polyhedra.

0.10.2
------

- Internal: replaced four parallel per-polyhedron lists in
  ``_PrecomputedScene`` (``poly_base_colours``, ``poly_alphas``,
  ``poly_edge_colours``, ``poly_edge_widths``) with a single list of
  frozen ``_PolyhedronRenderData`` dataclass instances, making the
  coupling between colour, alpha, and edge style explicit.

0.10.1
------

- :class:`~hofmann.BondSpec` validation is now extracted into
  per-field private methods, removing duplication between
  ``__init__`` and property setters.  The ``colour`` setter now
  validates its input via :func:`~hofmann.model.colour.normalise_colour`,
  closing a gap where invalid colours were silently accepted
  post-construction.

0.10.0
------

- New ``show_legend`` option on :class:`~hofmann.RenderStyle` draws a
  vertical column of coloured circles with species labels.  Customise
  placement, font size, circle sizing, spacing, and label gap via
  :class:`~hofmann.LegendStyle`.

- ``LegendStyle.circle_radius`` accepts three forms: a uniform float,
  a ``(min, max)`` tuple for proportional sizing based on
  ``AtomStyle.radius``, or a per-species dict for explicit control.

- :func:`~hofmann.rendering.static.render_legend` produces a
  tightly-cropped legend image without any structure, useful for
  composing figures in external tools.  Supports ``figsize`` for
  fixed output dimensions and ``transparent`` backgrounds.

- ``LegendStyle.label_gap`` controls the horizontal gap between legend
  circles and species labels (default 5.0 points).

- ``LegendStyle.labels`` accepts a dict mapping species names to
  display strings.  Common chemical notation is auto-formatted:
  trailing charges become superscripts with tight kerning (``"Sr2+"``
  renders as Sr^2+), embedded digits become subscripts (``"TiO6"``
  renders as TiO_6), and strings containing ``$`` are passed through
  as explicit matplotlib mathtext.

- Legend entry spacing is automatically widened when labels contain
  super/subscripts, unless the user has explicitly set a custom
  ``spacing`` value.

0.9.0
-----

- Periodic boundary handling has moved from scene construction to
  render time.  :func:`~hofmann.from_pymatgen` no longer expands image
  atoms at construction; instead, :class:`~hofmann.model.Bond` carries
  an ``image`` field recording which lattice translation the bond
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

- ``pbc``, ``pbc_padding``, ``max_recursive_depth``, and
  ``deduplicate_molecules`` are now fields on
  :class:`~hofmann.RenderStyle` rather than
  :class:`~hofmann.StructureScene`, so rendering style is fully
  separated from structure data.

- Bond detection for periodic structures dispatches based on the
  inscribed sphere radius of the unit cell.  When all bond lengths are
  shorter than the inscribed sphere radius (the common case), the
  minimum image convention (MIC) gives an efficient single-pass
  computation.  When bond lengths are comparable to cell dimensions,
  all 27 image offsets are checked iteratively.  Both paths use
  O(n^2) peak memory.

- Recursive expansion can produce duplicate molecular fragments.  The
  deduplication stage detects extended structures (slabs, frameworks)
  that wrap the unit cell and protects them from removal, removes
  non-wrapped components whose source atoms are a subset of a wrapped
  component, and selects a canonical copy among remaining duplicates
  using an unwrapped fractional centre-of-mass tie-breaker.

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

- The monolithic ``model.py`` (1888 lines) and ``render_mpl.py``
  (2517 lines) have been split into three sub-packages organised by
  architectural layer:

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
