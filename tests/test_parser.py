"""Tests for hofmann.parser — XBS .bs and .mv file parsing."""

import numpy as np
import pytest

from hofmann.parser import parse_bs, parse_mv


class TestParseBs:
    def test_ch4_species(self, ch4_bs_path):
        species, frame, styles, specs = parse_bs(ch4_bs_path)
        assert species == ["C", "H", "H", "H", "H"]

    def test_ch4_coords_shape(self, ch4_bs_path):
        species, frame, styles, specs = parse_bs(ch4_bs_path)
        assert frame.coords.shape == (5, 3)

    def test_ch4_coords_values(self, ch4_bs_path):
        species, frame, styles, specs = parse_bs(ch4_bs_path)
        np.testing.assert_allclose(frame.coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(frame.coords[1], [1.155, 1.155, 1.155])

    def test_ch4_atom_styles(self, ch4_bs_path):
        species, frame, styles, specs = parse_bs(ch4_bs_path)
        assert len(styles) == 2
        assert styles["C"].radius == 1.0
        assert styles["C"].colour == pytest.approx(0.7)
        assert styles["H"].radius == 0.7
        assert styles["H"].colour == pytest.approx(1.0)

    def test_ch4_bond_specs(self, ch4_bs_path):
        species, frame, styles, specs = parse_bs(ch4_bs_path)
        assert len(specs) == 2
        assert specs[0].species == ("C", "H")
        assert specs[0].min_length == 0.0
        assert specs[0].max_length == 3.4
        assert specs[0].radius == 0.109

    def test_from_string(self):
        content = "atom X 1.0 2.0 3.0\nspec X 0.5 red\n"
        species, frame, styles, specs = parse_bs(content)
        assert species == ["X"]
        np.testing.assert_allclose(frame.coords[0], [1.0, 2.0, 3.0])
        assert styles["X"].colour == "red"

    def test_rgb_colour(self):
        content = "spec O 0.8 1.0 0.3 0.3\natom O 0.0 0.0 0.0\n"
        species, frame, styles, specs = parse_bs(content)
        assert styles["O"].colour == pytest.approx((1.0, 0.3, 0.3))

    def test_named_colour(self):
        content = "spec N 0.75 blue\natom N 0.0 0.0 0.0\n"
        species, frame, styles, specs = parse_bs(content)
        assert styles["N"].colour == "blue"

    def test_comments_ignored(self):
        content = "* This is a comment\natom X 1.0 2.0 3.0\n"
        species, frame, styles, specs = parse_bs(content)
        assert species == ["X"]

    def test_blank_lines_ignored(self):
        content = "\natom X 1.0 2.0 3.0\n\n"
        species, frame, styles, specs = parse_bs(content)
        assert species == ["X"]

    def test_bonds_rgb_colour(self):
        content = (
            "atom A 0.0 0.0 0.0\n"
            "atom B 1.0 0.0 0.0\n"
            "bonds A B 0.0 2.0 0.1 0.8 0.2 0.3\n"
        )
        species, frame, styles, specs = parse_bs(content)
        assert specs[0].colour == pytest.approx((0.8, 0.2, 0.3))


class TestParseMv:
    def test_ch4_frame_count(self, ch4_mv_path):
        frames = parse_mv(ch4_mv_path, n_atoms=5)
        assert len(frames) == 2

    def test_ch4_frame_labels(self, ch4_mv_path):
        frames = parse_mv(ch4_mv_path, n_atoms=5)
        assert frames[0].label == "frame_1"
        assert frames[1].label == "frame_2"

    def test_ch4_frame_coords_shape(self, ch4_mv_path):
        frames = parse_mv(ch4_mv_path, n_atoms=5)
        assert frames[0].coords.shape == (5, 3)
        assert frames[1].coords.shape == (5, 3)

    def test_ch4_frame_1_coords(self, ch4_mv_path):
        frames = parse_mv(ch4_mv_path, n_atoms=5)
        # First atom (C) at origin in frame 1.
        np.testing.assert_allclose(frames[0].coords[0], [0.0, 0.0, 0.0])
        # Second atom (H) at (1.155, 1.155, 1.155).
        np.testing.assert_allclose(frames[0].coords[1], [1.155, 1.155, 1.155])

    def test_ch4_frame_2_coords(self, ch4_mv_path):
        frames = parse_mv(ch4_mv_path, n_atoms=5)
        # First atom shifted to (0.1, 0.1, 0.1) in frame 2.
        np.testing.assert_allclose(frames[1].coords[0], [0.1, 0.1, 0.1])

    def test_comments_ignored(self):
        content = (
            "* comment\n"
            "frame f1\n"
            "1.0 2.0 3.0\n"
        )
        frames = parse_mv(content, n_atoms=1)
        assert len(frames) == 1
        np.testing.assert_allclose(frames[0].coords[0], [1.0, 2.0, 3.0])

    def test_from_string(self):
        content = "frame f1\n1.0 2.0 3.0 4.0 5.0 6.0\n"
        frames = parse_mv(content, n_atoms=2)
        assert len(frames) == 1
        np.testing.assert_allclose(frames[0].coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(frames[0].coords[1], [4.0, 5.0, 6.0])
