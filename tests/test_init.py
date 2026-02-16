"""Tests for hofmann public API."""

import pytest

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
