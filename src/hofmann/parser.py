"""XBS .bs and .mv file parsers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hofmann.model import AtomStyle, BondSpec, Frame


def _read_source(source: str | Path) -> str:
    """Read file content from a path or return inline string content."""
    if isinstance(source, Path):
        return source.read_text()
    # If the string has no newlines and the path exists, read it.
    if "\n" not in source:
        path = Path(source)
        if path.is_file():
            return path.read_text()
    return source


def _parse_colour_tokens(tokens: list[str]) -> float | str | tuple[float, float, float]:
    """Parse colour from remaining tokens on a spec or bonds line.

    Returns a grey float, an RGB tuple, or a colour name string.
    """
    if len(tokens) == 1:
        try:
            return float(tokens[0])
        except ValueError:
            return tokens[0]  # Colour name.
    if len(tokens) == 3:
        try:
            return (float(tokens[0]), float(tokens[1]), float(tokens[2]))
        except ValueError:
            pass
    raise ValueError(f"Cannot parse colour from tokens: {tokens}")


def parse_bs(
    source: str | Path,
) -> tuple[list[str], Frame, dict[str, AtomStyle], list[BondSpec]]:
    """Parse an XBS .bs file.

    Args:
        source: Path to a ``.bs`` file, or the file content as a string.

    Returns:
        Tuple of ``(species, frame, atom_styles, bond_specs)``.
    """
    text = _read_source(source)

    species: list[str] = []
    coords: list[list[float]] = []
    atom_styles: dict[str, AtomStyle] = {}
    bond_specs: list[BondSpec] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        command = parts[0].lower()

        if command == "atom":
            species.append(parts[1])
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

        elif command == "spec":
            sp = parts[1]
            radius = float(parts[2])
            colour = _parse_colour_tokens(parts[3:])
            atom_styles[sp] = AtomStyle(radius=radius, colour=colour)

        elif command == "bonds":
            sp_a = parts[1]
            sp_b = parts[2]
            min_len = float(parts[3])
            max_len = float(parts[4])
            radius = float(parts[5])
            colour = _parse_colour_tokens(parts[6:])
            bond_specs.append(BondSpec(
                species=(sp_a, sp_b),
                min_length=min_len,
                max_length=max_len,
                radius=radius,
                colour=colour,
            ))

    frame = Frame(coords=np.array(coords), label="")
    return species, frame, atom_styles, bond_specs


def parse_mv(source: str | Path, n_atoms: int) -> list[Frame]:
    """Parse an XBS .mv trajectory file.

    Args:
        source: Path to a ``.mv`` file, or the file content as a string.
        n_atoms: Number of atoms per frame (needed to reshape coordinates).

    Returns:
        List of Frame objects, one per frame in the trajectory.
    """
    text = _read_source(source)

    frames: list[Frame] = []
    current_label = ""
    current_values: list[float] = []

    def _flush() -> None:
        if current_values:
            arr = np.array(current_values).reshape(n_atoms, 3)
            frames.append(Frame(coords=arr, label=current_label))

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue

        parts = line.split()
        if parts[0].lower() == "frame":
            _flush()
            current_label = parts[1] if len(parts) > 1 else ""
            current_values = []
        else:
            current_values.extend(float(v) for v in parts)

    _flush()  # Final frame.
    return frames
