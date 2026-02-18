# hofmann

[![CI](https://github.com/bjmorgan/hofmann/actions/workflows/ci.yml/badge.svg)](https://github.com/bjmorgan/hofmann/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/hofmann/badge/?version=latest)](https://hofmann.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/hofmann)](https://pypi.org/project/hofmann/)

A modern Python reimagining of Methfessel's [XBS](https://www.ccl.net/cca/software/X-WINDOW/xbs/) ball-and-stick viewer (1995), named after [August Wilhelm von Hofmann](https://en.wikipedia.org/wiki/August_Wilhelm_von_Hofmann) who built the first ball-and-stick molecular models in 1865.

hofmann renders crystal and molecular structures as depth-sorted ball-and-stick images with static, publication-quality vector output (SVG, PDF) via matplotlib.

<p align="center">
  <img src="https://raw.githubusercontent.com/bjmorgan/hofmann/main/docs/_static/llzo.png" width="480" alt="LLZO garnet with ZrO6 polyhedra rendered with hofmann">
</p>

## Features

- Static publication-quality output (SVG, PDF, PNG) via matplotlib
- XBS `.bs` and `.mv` (trajectory) file formats
- Optional pymatgen `Structure` interoperability
- Periodic boundary conditions with automatic image expansion
- Coordination polyhedra with configurable shading and slab clipping
- Unit cell wireframe rendering
- Interactive viewer with mouse rotation, zoom, and keyboard controls
- Orthographic and perspective projection

## Installation

```bash
pip install hofmann
```

For pymatgen interoperability:

```bash
pip install "hofmann[pymatgen]"
```

### Requirements

- Python 3.11+
- numpy >= 1.24
- matplotlib >= 3.7
- scipy >= 1.10
- pymatgen >= 2024.1.1 (optional)

## Quick start

### From an XBS file

```python
from hofmann import StructureScene

scene = StructureScene.from_xbs("structure.bs")
scene.render_mpl("output.svg")
```

### From a pymatgen Structure

```python
from pymatgen.core import Lattice, Structure
from hofmann import StructureScene, BondSpec

lattice = Lattice.cubic(5.43)
structure = Structure(
    lattice, ["Si"] * 8,
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
     [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
     [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]],
)

bonds = [BondSpec(species=("Si", "Si"), max_length=2.8)]
scene = StructureScene.from_pymatgen(structure, bonds, pbc=True)
scene.render_mpl("si.pdf")
```

### Controlling the view

```python
scene.view.look_along([1, 1, 0])   # View along [110]
scene.view.zoom = 1.5              # Zoom in
scene.view.perspective = 0.3       # Mild perspective
scene.render_mpl("rotated.svg")
```

### Interactive viewer

```python
view, style = scene.render_mpl_interactive()

# Reuse the adjusted view for static output:
scene.view = view
scene.render_mpl("final.svg", style=style)
```

## Documentation

Full documentation is available at [hofmann.readthedocs.io](https://hofmann.readthedocs.io/), covering:

- [Getting started](https://hofmann.readthedocs.io/en/latest/getting-started.html) -- installation and first renders
- [Scenes and structures](https://hofmann.readthedocs.io/en/latest/scenes.html) -- scenes, frames, bonds, polyhedra
- [Rendering](https://hofmann.readthedocs.io/en/latest/rendering.html) -- views, render styles, unit cells, axes
- [Colouring](https://hofmann.readthedocs.io/en/latest/colouring.html) -- per-atom data colouring, custom functions, multiple layers
- [Interactive viewer](https://hofmann.readthedocs.io/en/latest/interactive.html) -- mouse and keyboard controls
- [XBS file format](https://hofmann.readthedocs.io/en/latest/xbs-format.html) -- `.bs` and `.mv` format reference
- [API reference](https://hofmann.readthedocs.io/en/latest/api.html) -- full autodoc API

## Citing hofmann

If you use hofmann in published work, please cite it:

> B. J. Morgan, *hofmann*, https://github.com/bjmorgan/hofmann

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE) for details.
