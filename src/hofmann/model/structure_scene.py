from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hofmann.model.atom_data import _AtomData
from hofmann.model.atom_style import AtomStyle
from hofmann.model.bond_spec import BondSpec
from hofmann.model.colour import Colour, CmapSpec
from hofmann.model.frame import Frame
from hofmann.model.polyhedron_spec import PolyhedronSpec
from hofmann.model.render_style import RenderStyle
from hofmann.model.view_state import ViewState

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure


class StructureScene:
    """Top-level scene holding atoms, frames, styles, bond rules, and view.

    Attributes:
        species: One label per atom.
        frames: List of coordinate snapshots.  Each :class:`Frame` may
            carry its own ``lattice`` for variable-cell trajectories.
        atom_styles: Mapping from species label to visual style.
        bond_specs: Declarative bond detection rules.
        polyhedra: Declarative polyhedron rendering rules.
        view: Camera / projection state.
        title: Scene title for display.
        atom_data: Per-atom metadata container: read-only view with a
            Mapping-style interface.  See :meth:`set_atom_data`,
            :meth:`del_atom_data`, and :meth:`clear_2d_atom_data` for
            modifications.
    """

    def __init__(
        self,
        species: list[str],
        frames: list[Frame],
        atom_styles: dict[str, AtomStyle] | None = None,
        bond_specs: list[BondSpec] | None = None,
        polyhedra: list[PolyhedronSpec] | None = None,
        view: ViewState | None = None,
        title: str = "",
        atom_data: dict[str, ArrayLike] | None = None,
    ) -> None:
        self.species = species
        self.frames = frames
        self.atom_styles = atom_styles if atom_styles is not None else {}
        self.bond_specs = bond_specs if bond_specs is not None else []
        self.polyhedra = polyhedra if polyhedra is not None else []
        self.view = view if view is not None else ViewState()
        self.title = title

        # Validate frames.
        n_atoms = len(species)
        for i, frame in enumerate(frames):
            if frame.coords.shape[0] != n_atoms:
                raise ValueError(
                    f"species has {n_atoms} atoms but frame {i} has "
                    f"{frame.coords.shape[0]}"
                )
        if frames:
            has_lattice = [f.lattice is not None for f in frames]
            if any(has_lattice) and not all(has_lattice):
                raise ValueError(
                    "all frames must have a lattice or none must"
                )

        # Build validated _AtomData container.
        self._atom_data = _AtomData(n_atoms=n_atoms)
        if atom_data is not None:
            for key, arr_like in atom_data.items():
                arr = np.asarray(arr_like)
                if arr.ndim == 2 and arr.shape[0] != len(self.frames):
                    raise ValueError(
                        f"atom_data[{key!r}] has {arr.shape[0]} rows "
                        f"but scene has {len(self.frames)} frames"
                    )
                self._atom_data._set(key, arr)

    @property
    def view(self) -> ViewState:
        """Camera / projection state."""
        return self._view

    @view.setter
    def view(self, value: ViewState) -> None:
        if not isinstance(value, ViewState):
            hint = ""
            if isinstance(value, tuple):
                hint = (
                    " (hint: render_mpl_interactive() returns a"
                    " (ViewState, RenderStyle) tuple; did you forget"
                    " to unpack it?)"
                )
            raise TypeError(
                f"view must be a ViewState, got {type(value).__name__}"
                + hint
            )
        self._view = value

    @property
    def lattice(self) -> np.ndarray | None:
        """Lattice matrix of the first frame.

        Convenience accessor equivalent to ``self.frames[0].lattice``.
        Returns ``None`` when the scene has no frames or the first
        frame has no lattice (non-periodic structure).
        """
        if not self.frames:
            return None
        return self.frames[0].lattice

    @lattice.setter
    def lattice(self, value: object) -> None:
        raise AttributeError(
            "lattice is a read-only property; "
            "set lattice on individual frames instead"
        )

    @property
    def atom_data(self) -> _AtomData:
        """Per-atom metadata container: read-only Mapping-style view.

        Each value is either a 1-D array of length ``n_atoms`` (same
        every frame) or a 2-D array of shape ``(n_frames, n_atoms)``
        (per-frame values).  Stored arrays are returned read-only.
        The property has no setter; ``scene.atom_data = ...`` raises
        ``AttributeError``.  The container itself exposes only
        :class:`~collections.abc.Mapping` reads, so
        ``scene.atom_data[key] = ...`` is not supported either.  Use
        ``colour_by`` on the render methods to visualise a key, and
        see :meth:`set_atom_data`, :meth:`del_atom_data`, and
        :meth:`clear_2d_atom_data` for all modifications.
        """
        return self._atom_data

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
    def from_ase(
        cls,
        atoms: Atoms | Sequence[Atoms],
        bond_specs: list[BondSpec] | None = None,
        *,
        polyhedra: list[PolyhedronSpec] | None = None,
        centre_atom: int | None = None,
        atom_styles: dict[str, AtomStyle] | None = None,
        title: str = "",
        view: ViewState | None = None,
        atom_data: dict[str, ArrayLike] | None = None,
    ) -> StructureScene:
        """Create a StructureScene from ASE ``Atoms`` object(s).

        For periodic systems, fractional coordinates are wrapped to
        ``[0, 1)`` and stored as Cartesian coordinates.  For non-periodic
        systems, Cartesian positions are stored directly and
        :attr:`lattice` is ``None``.

        Args:
            atoms: A single ASE ``Atoms`` object or a sequence of
                ``Atoms`` (e.g. from an MD trajectory or
                ``ase.io.Trajectory``).
            bond_specs: Bond detection rules.  ``None`` generates
                sensible defaults from VESTA bond length cutoffs;
                pass an empty list to disable bonds.
            polyhedra: Polyhedron rendering rules.  ``None`` disables
                polyhedra.
            centre_atom: Index of the atom to centre the unit cell on.
                Fractional coordinates are shifted so this atom sits
                at (0.5, 0.5, 0.5).  Only valid for periodic systems.
                If *view* is also provided, the explicit view takes
                precedence and only the fractional-coordinate shift is
                applied.
            atom_styles: Per-species style overrides.  When provided,
                these are merged on top of the auto-generated defaults
                so you only need to specify the species you want to
                customise.
            title: Scene title for display.
            view: Camera / projection state.  When ``None`` (the
                default), the view is auto-centred on the centre atom
                (if set) or the centroid of all atoms.
            atom_data: Per-atom metadata arrays, keyed by name.
                Each value is a 1-D array of length ``n_atoms``
                (same every frame) or a 2-D array of shape
                ``(n_frames, n_atoms)`` (per-frame values).

        Returns:
            A StructureScene with default element styles.

        Raises:
            ImportError: If ASE is not installed.
            ValueError: If *atoms* is an empty sequence, if
                *centre_atom* is out of range, if *centre_atom* is
                used with a non-periodic system, or if frames in a
                trajectory have inconsistent species, atom counts,
                or periodicity.

        See Also:
            :func:`hofmann.construction.scene_builders.from_ase`
        """
        from hofmann.construction.scene_builders import from_ase

        return from_ase(
            atoms, bond_specs, polyhedra=polyhedra,
            centre_atom=centre_atom,
            atom_styles=atom_styles, title=title, view=view,
            atom_data=atom_data,
        )

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
        atom_data: dict[str, ArrayLike] | None = None,
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
                Each value is a 1-D array of length ``n_atoms``
                (same every frame) or a 2-D array of shape
                ``(n_frames, n_atoms)`` (per-frame values).

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
        file, is ignored; pass it to the render call instead.

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
        values: ArrayLike | dict[int, object],
    ) -> None:
        """Set per-atom metadata for colourmap-based rendering.

        Canonical write entry point for per-atom metadata.  The
        container is otherwise read-only: to remove a single entry use
        :meth:`del_atom_data`, and to bulk-drop all 2-D entries (e.g.
        after extending the trajectory) use
        :meth:`clear_2d_atom_data`.

        A 2-D *values* array is validated against ``len(self.frames)``
        before storage, in addition to the cross-entry 2-D consistency
        check performed by the underlying container.

        Args:
            key: Name for this metadata (e.g. ``"charge"``,
                ``"site"``).
            values: Either a 1-D array-like of length ``n_atoms`` (same
                value every frame), a 2-D array-like of shape
                ``(n_frames, n_atoms)`` (per-frame values), or a dict
                mapping atom indices to values (always 1-D).  When a
                dict is given, the fill value for missing atoms is
                inferred from the first entry: ``NaN`` for numeric
                values or ``""`` for string values.  All values in a
                dict must be of compatible types (all numeric or all
                strings).

        Raises:
            ValueError: If an array-like has the wrong length or shape,
                if a 2-D array's leading dimension does not match
                ``len(self.frames)``, or if a dict contains indices
                outside the valid range.
            TypeError: If a dict contains a mixture of string and
                numeric values.

        See Also:
            :meth:`del_atom_data`: Remove a single entry.
            :meth:`clear_2d_atom_data`: Remove all 2-D entries.
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

        if arr.ndim == 2 and arr.shape[0] != len(self.frames):
            raise ValueError(
                f"atom_data[{key!r}] has {arr.shape[0]} rows but "
                f"scene has {len(self.frames)} frames"
            )

        self._atom_data._set(key, arr)

    def del_atom_data(self, key: str) -> None:
        """Remove a per-atom metadata entry.

        Args:
            key: The metadata key to remove.

        Raises:
            KeyError: If *key* is not present in :attr:`atom_data`.

        See Also:
            :meth:`set_atom_data`: Canonical write entry point.
            :meth:`clear_2d_atom_data`: Remove all 2-D entries at once.
        """
        self._atom_data._del(key)

    def clear_2d_atom_data(self) -> None:
        """Remove all 2-D per-atom metadata entries, preserving 1-D.

        Typical use after extending the trajectory when stored 2-D
        data no longer matches the frame count::

            scene.frames.append(new_frame)
            scene.clear_2d_atom_data()
            scene.set_atom_data("energy", new_energy_2d)
            scene.render_mpl(...)

        See Also:
            :meth:`set_atom_data`: Canonical write entry point.
            :meth:`del_atom_data`: Remove a single entry.
        """
        self._atom_data.clear_2d()

    def _validate_for_render(self) -> None:
        """Raise if atom_data is incompatible with the current trajectory.

        Called as the first action of every public ``render_*``
        method.  Delegates to the container's
        ``_check_frames_compatibility`` with ``len(self.frames)`` as
        the expected frame count.  Fires exactly once per user-facing
        render invocation, before any per-frame work begins, so a
        stale scene fails fast without partially initialising figures
        or animation writers.

        Raises:
            ValueError: If the container has a 2-D entry whose
                ``shape[0]`` does not equal ``len(self.frames)``.
        """
        self._atom_data._check_frames_compatibility(len(self.frames))

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
            cmap: A :type:`CmapSpec`: matplotlib colourmap name
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
        self._validate_for_render()

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
            cmap: A :type:`CmapSpec`: colourmap name, object, or
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
        self._validate_for_render()

        from hofmann.rendering.interactive import render_mpl_interactive

        return render_mpl_interactive(
            self, style=style, frame_index=frame_index,
            figsize=figsize, dpi=dpi, background=background,
            colour_by=colour_by, cmap=cmap, colour_range=colour_range,
            **style_kwargs,
        )

    def render_animation(
        self,
        output: str | Path,
        *,
        style: RenderStyle | None = None,
        frames: range | Sequence[int] | None = None,
        fps: int = 30,
        figsize: tuple[float, float] = (5.0, 5.0),
        dpi: int = 150,
        background: Colour = "white",
        colour_by: str | list[str] | None = None,
        cmap: CmapSpec | list[CmapSpec] = "viridis",
        colour_range: (
            tuple[float, float]
            | None
            | list[tuple[float, float] | None]
        ) = None,
        **style_kwargs: object,
    ) -> Path:
        """Render a trajectory animation to a GIF or MP4 file.

        Loops over the specified frames, rendering each with the
        per-frame pipeline and writing it to the output file.

        Args:
            output: Destination file path.  Extension determines the
                format (e.g. ``.gif``, ``.mp4``).
            style: A :class:`RenderStyle` controlling visual appearance.
                Any :class:`RenderStyle` field name may also be passed
                as a keyword argument to override individual fields.
            frames: Which frame indices to render, in order.  ``None``
                renders all frames.  Accepts ``range(0, 100, 5)`` or
                an arbitrary sequence of indices.
            fps: Frames per second in the output file.
            figsize: Figure size in inches ``(width, height)``.
            dpi: Resolution in dots per inch.
            background: Background colour.
            colour_by: Key (or list of keys) into :attr:`atom_data`
                to colour atoms by.
            cmap: A :type:`CmapSpec`: colourmap name, object, or
                callable.
            colour_range: Explicit ``(vmin, vmax)`` for numerical
                data.
            **style_kwargs: Any :class:`RenderStyle` field name as a
                keyword argument (e.g. ``show_bonds=False``).

        Returns:
            The output file path as a :class:`~pathlib.Path`.

        Raises:
            ValueError: If *frames* is empty or contains out-of-range
                indices, if *fps* is less than 1, or if *output*
                has an unsupported file extension (must be ``.gif``
                or ``.mp4``).
            ImportError: If ``imageio`` is not installed.

        See Also:
            :func:`hofmann.rendering.animation.render_animation`
        """
        self._validate_for_render()

        from hofmann.rendering.animation import render_animation

        return render_animation(
            self, output, style=style, frames=frames, fps=fps,
            figsize=figsize, dpi=dpi, background=background,
            colour_by=colour_by, cmap=cmap,
            colour_range=colour_range, **style_kwargs,
        )
