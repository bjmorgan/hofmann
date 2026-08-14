"""Tests for hofmann public API."""

import hofmann


class TestPublicAPI:
    def test_all_names_importable(self):
        for name in hofmann.__all__:
            assert hasattr(hofmann, name), f"{name} not importable from hofmann"

    def test_structure_scene_from_xbs(self, ch4_bs_path):
        scene = hofmann.StructureScene.from_xbs(ch4_bs_path)
        assert isinstance(scene, hofmann.StructureScene)
        assert len(scene.species) == 5

    def test_end_to_end_xbs_to_mpl(self, ch4_bs_path, tmp_path):
        scene = hofmann.StructureScene.from_xbs(ch4_bs_path)
        out = tmp_path / "ch4.png"
        scene.render_mpl(output=out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_composition_top_level_export():
    import hofmann
    import hofmann.model.composition

    assert hasattr(hofmann, "Composition")
    assert hofmann.Composition is hofmann.model.composition.Composition


def test_composition_in_all():
    import hofmann

    assert "Composition" in hofmann.__all__


def test_oblique_top_level_export():
    import hofmann
    import hofmann.model.view_state

    assert hofmann.Oblique is hofmann.model.view_state.Oblique


def test_oblique_in_all():
    import hofmann

    assert "Oblique" in hofmann.__all__


def test_orthographic_top_level_export():
    import hofmann
    import hofmann.model.view_state

    assert hofmann.Orthographic is hofmann.model.view_state.Orthographic


def test_orthographic_in_all():
    import hofmann

    assert "Orthographic" in hofmann.__all__


def test_perspective_top_level_export():
    import hofmann
    import hofmann.model.view_state

    assert hofmann.Perspective is hofmann.model.view_state.Perspective


def test_perspective_in_all():
    import hofmann

    assert "Perspective" in hofmann.__all__
