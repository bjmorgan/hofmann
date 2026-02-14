"""Demo script: load CH4 from XBS file and render with matplotlib."""

from pathlib import Path

from hofmann import StructureScene

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT = Path(__file__).resolve().parent / "ch4.pdf"


def main():
    scene = StructureScene.from_xbs(FIXTURES / "ch4.bs")
    print(f"Loaded scene: {len(scene.species)} atoms, {len(scene.frames)} frame(s)")
    print(f"Species: {scene.species}")
    print(f"Atom styles: {list(scene.atom_styles.keys())}")
    print(f"Bond specs: {len(scene.bond_specs)}")

    scene.render_mpl(output=OUTPUT, show=False, half_bonds=False)
    print(f"Rendered to {OUTPUT}")


if __name__ == "__main__":
    main()
