"""Style set save/load for JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hofmann.model import (
    AtomStyle,
    BondSpec,
    PolyhedronSpec,
    RenderStyle,
)

_VALID_SECTIONS = frozenset({
    "atom_styles", "bond_specs", "polyhedra", "render_style",
})


@dataclass
class StyleSet:
    """A collection of style settings loaded from or saved to a file.

    All fields are optional.  A ``StyleSet`` loaded from a file that
    only contains ``"atom_styles"`` will have ``bond_specs``,
    ``polyhedra``, and ``render_style`` set to ``None``.

    Attributes:
        atom_styles: Per-species visual styles, keyed by species label.
        bond_specs: Bond detection and appearance rules.
        polyhedra: Polyhedron rendering rules.
        render_style: Global rendering parameters.
    """

    atom_styles: dict[str, AtomStyle] | None = None
    bond_specs: list[BondSpec] | None = None
    polyhedra: list[PolyhedronSpec] | None = None
    render_style: RenderStyle | None = None


def save_styles(
    path: str | Path,
    *,
    atom_styles: dict[str, AtomStyle] | None = None,
    bond_specs: list[BondSpec] | None = None,
    polyhedra: list[PolyhedronSpec] | None = None,
    render_style: RenderStyle | None = None,
) -> None:
    """Save style settings to a JSON file.

    Only sections that are not ``None`` are written.  The file is
    human-readable with two-space indentation.

    Args:
        path: Destination file path.
        atom_styles: Per-species visual styles.
        bond_specs: Bond detection and appearance rules.
        polyhedra: Polyhedron rendering rules.
        render_style: Global rendering parameters.
    """
    data: dict = {}
    if atom_styles is not None:
        data["atom_styles"] = {
            sp: style.to_dict() for sp, style in atom_styles.items()
        }
    if bond_specs is not None:
        data["bond_specs"] = [spec.to_dict() for spec in bond_specs]
    if polyhedra is not None:
        data["polyhedra"] = [spec.to_dict() for spec in polyhedra]
    if render_style is not None:
        data["render_style"] = render_style.to_dict()

    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_styles(path: str | Path) -> StyleSet:
    """Load style settings from a JSON file.

    All sections are optional.  Unknown top-level keys raise
    :class:`ValueError`.

    Args:
        path: Source file path.

    Returns:
        A :class:`StyleSet` with the parsed sections.

    Raises:
        ValueError: If the file contains unknown top-level keys.
    """
    data = json.loads(Path(path).read_text())

    unknown = set(data) - _VALID_SECTIONS
    if unknown:
        raise ValueError(
            f"unknown top-level keys in style file: {sorted(unknown)}"
        )

    atom_styles = None
    if "atom_styles" in data:
        atom_styles = {
            sp: AtomStyle.from_dict(d)
            for sp, d in data["atom_styles"].items()
        }

    bond_specs = None
    if "bond_specs" in data:
        bond_specs = [BondSpec.from_dict(d) for d in data["bond_specs"]]

    polyhedra = None
    if "polyhedra" in data:
        polyhedra = [
            PolyhedronSpec.from_dict(d) for d in data["polyhedra"]
        ]

    render_style = None
    if "render_style" in data:
        render_style = RenderStyle.from_dict(data["render_style"])

    return StyleSet(
        atom_styles=atom_styles,
        bond_specs=bond_specs,
        polyhedra=polyhedra,
        render_style=render_style,
    )
