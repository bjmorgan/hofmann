from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hofmann.model.atom_style import AtomStyle
from hofmann.model.bond_spec import BondSpec
from hofmann.model.colour import Colour, CmapSpec
from hofmann.model.frame import Frame
from hofmann.model.polyhedron_spec import PolyhedronSpec
from hofmann.model.render_style import RenderStyle
from hofmann.model.view_state import ViewState

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass
class StructureScene:
    """Top-level scene holding atoms, frames, styles, bond rules, and view.

    Attributes:
        species: One label per atom.
        frames: List of coordinate snapshots.
        atom_styles: Mapping from species label to visual style.
        bond_specs: Declarative bond detection rules.
        polyhedra: Declarative polyhedron rendering rules.
        view: Camera / projection state.
        title: Scene title for display.
        lattice: Unit cell lattice matrix, shape ``(3, 3)`` with rows
            as lattice vectors, or ``None`` for non-periodic structures.
            When a lattice is present, the renderer can use it for
            periodic bond computation and image-atom expansion
            (controlled by :attr:`RenderStyle.pbc`).
        atom_data: Per-atom metadata arrays, keyed by name.  Each value
            must be a 1-D array of length ``n_atoms``.  Use
            :meth:`set_atom_data` to populate this and ``colour_by``
            on the render methods to visualise it.
    """

    species: list[str]
    frames: list[Frame]
    atom_styles: dict[str, AtomStyle] = field(default_factory=dict)
    bond_specs: list[BondSpec] = field(default_factory=list)
    polyhedra: list[PolyhedronSpec] = field(default_factory=list)
    view: ViewState = field(default_factory=ViewState)
    title: str = ""
    lattice: np.ndarray | None = None
    atom_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "view" and not isinstance(value, ViewState):
            hint = ""
            if isinstance(value, tuple):
                hint = (
                    " (hint: render_mpl_interactive() returns a"
                    " (ViewState, RenderStyle) tuple — did you forget"
                    " to unpack it?)"
                )
            raise TypeError(
                f"view must be a ViewState, got {type(value).__name__}"
                + hint
            )
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        if self.lattice is not None:
            self.lattice = np.asarray(self.lattice, dtype=float)
            if self.lattice.shape != (3, 3):
                raise ValueError(
                    f"lattice must have shape (3, 3), got {self.lattice.shape}"
                )
        n_atoms = len(self.species)
        for i, frame in enumerate(self.frames):
            if frame.coords.shape[0] != n_atoms:
                raise ValueError(
                    f"species has {n_atoms} atoms but frame {i} has "
                    f"{frame.coords.shape[0]}"
                )
        for key, arr in self.atom_data.items():
            arr = np.asarray(arr)
            if arr.ndim != 1 or len(arr) != n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length {n_atoms}, "
                    f"got shape {arr.shape}"
                )
            self.atom_data[key] = arr

    @classmethod
    def from_xbs(
        cls,
        bs_path: str | Path,
        mv_path: str | Path | None = None,
    ) -> StructureScene:
        """Create a StructureScene from XBS ``.bs`` (and optional ``.mv``) files.

        Args:
            bs_path: Path to the ``.bs`` structure file.
            mv_path: Optional path to a ``.mv`` trajectory file.  When
                provided, the scene will contain multiple frames.

        Returns:
            A fully configured StructureScene with styles and bond
            specs parsed from the file.

        See Also:
            :func:`hofmann.construction.scene_builders.from_xbs`
        """
        from hofmann.construction.scene_builders import from_xbs

        return from_xbs(bs_path, mv_path)

    @classmethod
    def from_pymatgen(
        cls,
        structure: Structure | Sequence[Structure],
        bond_specs: list[BondSpec] | None = None,
        *,
        polyhedra: list[PolyhedronSpec] | None = None,
        centre_atom: int | None = None,
        atom_styles: dict[str, AtomStyle] | None = None,
        title: str = "",
        view: ViewState | None = None,
        atom_data: dict[str, np.ndarray] | None = None,
    ) -> StructureScene:
        """Create a StructureScene from pymatgen ``Structure`` object(s).

        Fractional coordinates are wrapped to ``[0, 1)`` and stored as
        Cartesian coordinates.  Periodic boundary handling (image-atom
        expansion, recursive bond depth, molecule deduplication) is
        controlled at render time via :class:`RenderStyle`.

        Args:
            structure: A single pymatgen ``Structure`` or a sequence of
                structures (e.g. from an MD trajectory).
            bond_specs: Bond detection rules.  ``None`` generates
                sensible defaults from VESTA bond length cutoffs;
                pass an empty list to disable bonds.
            polyhedra: Polyhedron rendering rules.  ``None`` disables
                polyhedra.
            centre_atom: Index of the atom to centre the unit cell on.
                Fractional coordinates are shifted so this atom sits
                at (0.5, 0.5, 0.5).  If *view* is also provided, the
                explicit view takes precedence and only the fractional-
                coordinate shift is applied.
            atom_styles: Per-species style overrides.  When provided,
                these are merged on top of the auto-generated defaults
                so you only need to specify the species you want to
                customise.
            title: Scene title for display.
            view: Camera / projection state.  When ``None`` (the
                default), the view is auto-centred on the structure.
            atom_data: Per-atom metadata arrays, keyed by name.

        Returns:
            A StructureScene with default element styles.

        Raises:
            ImportError: If pymatgen is not installed.

        See Also:
            :func:`hofmann.construction.scene_builders.from_pymatgen`
        """
        from hofmann.construction.scene_builders import from_pymatgen

        return from_pymatgen(
            structure, bond_specs, polyhedra=polyhedra,
            centre_atom=centre_atom,
            atom_styles=atom_styles, title=title, view=view,
            atom_data=atom_data,
        )

    def save_styles(self, path: str | Path) -> None:
        """Save the scene's styles to a JSON file.

        Writes ``atom_styles``, ``bond_specs``, and ``polyhedra``
        sections.  Render style is not included (it belongs to the
        render call, not the scene).

        Args:
            path: Destination file path.
        """
        from hofmann.construction.styles import save_styles

        save_styles(
            path,
            atom_styles=self.atom_styles,
            bond_specs=self.bond_specs,
            polyhedra=self.polyhedra,
        )

    def load_styles(self, path: str | Path) -> None:
        """Load styles from a JSON file and apply them to the scene.

        Atom styles are merged (existing species keep their styles
        unless overridden).  Bond specs and polyhedra are replaced
        entirely.  The ``render_style`` section, if present in the
        file, is ignored — pass it to the render call instead.

        Args:
            path: Source file path.
        """
        from hofmann.construction.styles import load_styles

        style_set = load_styles(path)
        if style_set.atom_styles is not None:
            self.atom_styles.update(style_set.atom_styles)
        if style_set.bond_specs is not None:
            self.bond_specs = style_set.bond_specs
        if style_set.polyhedra is not None:
            self.polyhedra = style_set.polyhedra

    def centre_on(self, atom_index: int, *, frame: int = 0) -> None:
        """Centre the view on a specific atom.

        Sets :attr:`view.centre` to the Cartesian position of the atom
        at *atom_index* in the given frame.

        Args:
            atom_index: Index of the atom to centre on.
            frame: Frame index to read coordinates from.
        """
        n_frames = len(self.frames)
        if not 0 <= frame < n_frames:
            raise ValueError(
                f"frame {frame} out of range for scene "
                f"with {n_frames} frame(s)"
            )
        n_atoms = len(self.species)
        if not 0 <= atom_index < n_atoms:
            raise ValueError(
                f"atom_index {atom_index} out of range for scene "
                f"with {n_atoms} atom(s)"
            )
        self.view.centre = self.frames[frame].coords[atom_index].copy()

    def set_atom_data(
        self,
        key: str,
        values: np.ndarray | Sequence[float] | Sequence[str] | dict[int, object],
    ) -> None:
        """Set per-atom metadata for colourmap-based rendering.

        Args:
            key: Name for this metadata (e.g. ``"charge"``,
                ``"site"``).
            values: Either an array-like of length ``n_atoms``, or a
                dict mapping atom indices to values.  When a dict is
                given, the fill value for missing atoms is inferred
                from the first entry: ``NaN`` for numeric values or
                ``""`` for string values.  All values in a dict must
                be of compatible types (all numeric or all strings).

        Raises:
            ValueError: If an array-like has the wrong length, or a
                dict contains indices outside the valid range.
            TypeError: If a dict contains a mixture of string and
                numeric values.
        """
        n_atoms = len(self.species)

        if isinstance(values, dict):
            if not values:
                raise ValueError("values dict must not be empty")
            for idx in values:
                if not 0 <= idx < n_atoms:
                    raise ValueError(
                        f"atom index {idx} out of range for "
                        f"{n_atoms} atoms"
                    )
            sample = next(iter(values.values()))
            is_str = isinstance(sample, str)
            for idx, val in values.items():
                if isinstance(val, str) != is_str:
                    raise TypeError(
                        f"atom_data dict values must all be the same "
                        f"type (string or numeric), but index {idx} "
                        f"has type {type(val).__name__!r}"
                    )
            if is_str:
                arr = np.array([""] * n_atoms, dtype=object)
                for idx, val in values.items():
                    arr[idx] = val
            else:
                arr = np.full(n_atoms, np.nan)
                for idx, val in values.items():
                    arr[idx] = val
        else:
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) != n_atoms:
                raise ValueError(
                    f"atom_data[{key!r}] must have length {n_atoms}, "
                    f"got shape {arr.shape}"
                )

        self.atom_data[key] = arr

    def render_mpl(
        self,
        output: str | Path | None = None,
        *,
        ax: Axes | None = None,
        style: RenderStyle | None = None,
        frame_index: int = 0,
        figsize: tuple[float, float] = (5.0, 5.0),
        dpi: int = 150,
        background: Colour = "white",
        show: bool | None = None,
        colour_by: str | list[str] | None = None,
        cmap: CmapSpec | list[CmapSpec] = "viridis",
        colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
        **style_kwargs: object,
    ) -> Figure:
        """Render the scene as a static matplotlib figure.

        Args:
            output: Optional file path to save the figure.  The format
                is inferred from the extension (``.svg``, ``.pdf``,
                ``.png``).  Ignored when *ax* is provided.
            ax: Optional matplotlib
                :class:`~matplotlib.axes.Axes` to draw into.  When
                provided, the caller is responsible for saving and
                closing the figure.  The *output*, *figsize*, *dpi*,
                *background*, and *show* parameters are ignored.
            style: A :class:`RenderStyle` controlling visual appearance.
                Any :class:`RenderStyle` field name may also be passed
                as a keyword argument to override individual fields.
            frame_index: Which frame to render (default 0).
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution for raster output formats.
            background: Background colour.
            show: Whether to call ``plt.show()``.  Defaults to
                ``True`` when *output* is ``None``, ``False`` when
                saving to a file.
            colour_by: Key (or list of keys) into :attr:`atom_data`
                to colour atoms by.  When ``None`` (the default),
                species-based colouring is used.  When a list, layers
                are tried in priority order and the first non-missing
                value determines the atom's colour.
            cmap: A :type:`CmapSpec` — matplotlib colourmap name
                (e.g. ``"viridis"``), ``Colormap`` object, or callable
                mapping a float in ``[0, 1]`` to an ``(r, g, b)``
                tuple.  When *colour_by* is a list, *cmap* may also
                be a list of the same length (one per layer).
            colour_range: Explicit ``(vmin, vmax)`` for normalising
                numerical data.  ``None`` auto-ranges from the data.
                When *colour_by* is a list, may also be a list of the
                same length.
            **style_kwargs: Any :class:`RenderStyle` field name as a
                keyword argument (e.g. ``show_bonds=False``).

        Returns:
            The matplotlib :class:`~matplotlib.figure.Figure`.

        See Also:
            :func:`hofmann.rendering.static.render_mpl`
        """
        from hofmann.rendering.static import render_mpl

        return render_mpl(
            self, output, ax=ax, style=style, frame_index=frame_index,
            figsize=figsize, dpi=dpi, background=background,
            show=show, colour_by=colour_by, cmap=cmap,
            colour_range=colour_range, **style_kwargs,
        )

    def render_mpl_interactive(
        self,
        *,
        style: RenderStyle | None = None,
        frame_index: int = 0,
        figsize: tuple[float, float] = (5.0, 5.0),
        dpi: int = 150,
        background: Colour = "white",
        colour_by: str | list[str] | None = None,
        cmap: CmapSpec | list[CmapSpec] = "viridis",
        colour_range: tuple[float, float] | None | list[tuple[float, float] | None] = None,
        **style_kwargs: object,
    ) -> tuple[ViewState, RenderStyle]:
        """Open an interactive matplotlib viewer with mouse and keyboard controls.

        Left-drag rotates, scroll zooms, and keyboard shortcuts control
        rotation, pan, perspective, display toggles, and frame navigation.
        Press **h** to show a help overlay listing all keybindings.

        When the window is closed the updated :class:`ViewState` and
        :class:`RenderStyle` are returned so they can be reused for
        static rendering::

            view, style = scene.render_mpl_interactive()
            scene.view = view
            scene.render_mpl("output.svg", style=style)

        Args:
            style: A :class:`RenderStyle` controlling visual appearance.
                Any :class:`RenderStyle` field name may also be passed
                as a keyword argument to override individual fields.
            frame_index: Which frame to render initially.
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution.
            background: Background colour.
            colour_by: Key (or list of keys) into :attr:`atom_data`
                to colour atoms by.  When a list, layers are tried in
                priority order.
            cmap: A :type:`CmapSpec` — colourmap name, object, or
                callable.  When *colour_by* is a list, may also be a
                list of the same length.
            colour_range: Explicit ``(vmin, vmax)`` for numerical
                data.  When *colour_by* is a list, may also be a list
                of the same length.
            **style_kwargs: Any :class:`RenderStyle` field name as a
                keyword argument (e.g. ``show_bonds=False``).

        Returns:
            A ``(ViewState, RenderStyle)`` tuple reflecting any view
            and style changes applied during the interactive session.

        See Also:
            :func:`hofmann.rendering.interactive.render_mpl_interactive`
        """
        from hofmann.rendering.interactive import render_mpl_interactive

        return render_mpl_interactive(
            self, style=style, frame_index=frame_index,
            figsize=figsize, dpi=dpi, background=background,
            colour_by=colour_by, cmap=cmap, colour_range=colour_range,
            **style_kwargs,
        )
