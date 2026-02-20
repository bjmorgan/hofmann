"""Tests for colour normalisation and atom colour resolution."""

import numpy as np
import pytest

from hofmann.model.atom_style import AtomStyle
from hofmann.model.colour import normalise_colour, resolve_atom_colours


class TestNormaliseColour:
    def test_css_name(self):
        assert normalise_colour("red") == (1.0, 0.0, 0.0)

    def test_hex_string(self):
        assert normalise_colour("#00FF00") == pytest.approx((0.0, 1.0, 0.0))

    def test_grey_float_zero(self):
        assert normalise_colour(0.0) == (0.0, 0.0, 0.0)

    def test_grey_float(self):
        assert normalise_colour(0.7) == pytest.approx((0.7, 0.7, 0.7))

    def test_grey_float_one(self):
        assert normalise_colour(1.0) == (1.0, 1.0, 1.0)

    def test_rgb_tuple(self):
        assert normalise_colour((0.5, 0.3, 0.1)) == pytest.approx(
            (0.5, 0.3, 0.1)
        )

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognised colour"):
            normalise_colour("notacolour")

    def test_grey_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Grey value"):
            normalise_colour(1.5)

    def test_rgb_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3 elements"):
            normalise_colour((0.5, 0.3))  # type: ignore[arg-type]

    def test_rgb_out_of_range_raises(self):
        with pytest.raises(ValueError, match="RGB component"):
            normalise_colour((0.5, 1.5, 0.0))

    def test_rgb_list(self):
        assert normalise_colour([0.5, 0.3, 0.1]) == pytest.approx(
            (0.5, 0.3, 0.1)
        )

    def test_rgb_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3 elements"):
            normalise_colour([0.5, 0.3])  # type: ignore[arg-type]


class TestResolveAtomColours:
    """Tests for resolve_atom_colours."""

    SPECIES = ["C", "H", "O"]
    STYLES: dict = {
        "C": AtomStyle(radius=1.0, colour=(0.4, 0.4, 0.4)),
        "H": AtomStyle(radius=0.7, colour=(1.0, 1.0, 1.0)),
        "O": AtomStyle(radius=0.8, colour=(0.6, 0.0, 0.0)),
    }

    def test_species_fallback(self):
        """colour_by=None returns species-based colours."""
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, {},
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_species_fallback_missing_style(self):
        """Missing species falls back to grey."""
        result = resolve_atom_colours(
            ["X"], {}, {},
        )
        assert result == [(0.5, 0.5, 0.5)]

    def test_numerical_viridis_endpoints(self):
        """Known endpoints of the viridis colourmap."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        expected_0 = cmap(0.0)[:3]
        expected_1 = cmap(1.0)[:3]

        data = {"val": np.array([0.0, 1.0])}
        result = resolve_atom_colours(
            ["A", "B"], self.STYLES, data,
            colour_by="val", cmap="viridis",
        )
        assert result[0] == pytest.approx(expected_0)
        assert result[1] == pytest.approx(expected_1)

    def test_numerical_custom_range(self):
        """Explicit colour_range normalises correctly."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        # value 5 with range (0, 10) -> normalised 0.5
        expected = cmap(0.5)[:3]

        data = {"val": np.array([5.0])}
        result = resolve_atom_colours(
            ["A"], self.STYLES, data,
            colour_by="val", colour_range=(0.0, 10.0),
        )
        assert result[0] == pytest.approx(expected)

    def test_numerical_constant_values(self):
        """All-same values should map to 0.5 (no division by zero)."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        expected = cmap(0.5)[:3]

        data = {"val": np.array([3.0, 3.0, 3.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        for c in result:
            assert c == pytest.approx(expected)

    def test_numerical_nan_falls_back(self):
        """NaN entries get their species colour."""
        data = {"val": np.array([0.0, np.nan, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)

    def test_numerical_all_nan(self):
        """All NaN returns species colours."""
        data = {"val": np.array([np.nan, np.nan, np.nan])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_categorical_distinct_colours(self):
        """Two categories get two different colours."""
        data = {"site": np.array(["4a", "8b", "4a"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Atoms 0 and 2 (both "4a") should have the same colour.
        assert result[0] == result[2]
        # Atom 1 ("8b") should differ.
        assert result[1] != result[0]

    def test_categorical_empty_falls_back(self):
        """Empty string entries get their species colour."""
        data = {"site": np.array(["4a", "", "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H, empty label) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)

    def test_callable_cmap(self):
        """A callable cmap is used directly."""
        def red_blue(val: float) -> tuple[float, float, float]:
            return (val, 0.0, 1.0 - val)

        data = {"val": np.array([0.0, 0.5, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap=red_blue,
        )
        assert result[0] == pytest.approx((0.0, 0.0, 1.0))
        assert result[1] == pytest.approx((0.5, 0.0, 0.5))
        assert result[2] == pytest.approx((1.0, 0.0, 0.0))

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, {},
                colour_by="nonexistent",
            )

    def test_list_colour_by_priority(self):
        """Non-overlapping layers: each atom gets its layer's colour."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[red, blue],
        )
        assert result[0] == (1.0, 0.0, 0.0)  # from layer "a"
        assert result[1] == (0.0, 0.0, 1.0)  # from layer "b"
        # Atom 2 has NaN in both — species fallback (O)
        assert result[2] == (0.6, 0.0, 0.0)

    def test_list_colour_by_first_wins(self):
        """When an atom has data in multiple layers, first wins."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([2.0, 2.0, np.nan]),  # atom 0 in both
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[red, blue],
        )
        # Atom 0: layer "a" has data, so red wins over blue.
        assert result[0] == (1.0, 0.0, 0.0)
        assert result[1] == (0.0, 0.0, 1.0)

    def test_list_colour_by_broadcast_cmap(self):
        """A single cmap string is broadcast to all layers."""
        data = {
            "a": np.array([0.0, np.nan, np.nan]),
            "b": np.array([np.nan, 1.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap="viridis",
        )
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        assert result[0] == pytest.approx(cmap(0.5)[:3])  # constant -> 0.5
        assert result[1] == pytest.approx(cmap(0.5)[:3])

    def test_list_colour_by_all_missing_falls_back(self):
        """Atom with NaN in all layers gets species colour."""
        data = {
            "a": np.array([np.nan, np.nan, np.nan]),
            "b": np.array([np.nan, np.nan, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"],
        )
        assert result == [
            (0.4, 0.4, 0.4),
            (1.0, 1.0, 1.0),
            (0.6, 0.0, 0.0),
        ]

    def test_list_colour_by_categorical(self):
        """List colour_by works with categorical layers."""
        def red(_v: float) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            "metal": np.array(["Fe", "", ""], dtype=object),
            "anion": np.array(["", "O", ""], dtype=object),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["metal", "anion"], cmap=[red, blue],
        )
        assert result[0] == (1.0, 0.0, 0.0)  # Fe
        assert result[1] == (0.0, 0.0, 1.0)  # O
        assert result[2] == (0.6, 0.0, 0.0)  # fallback

    def test_numerical_integer_array(self):
        """Integer arrays are coerced to float without error."""
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]

        data = {"val": np.array([1, 2, 3])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="val",
        )
        # 1->0.0, 2->0.5, 3->1.0 after normalisation
        assert result[0] == pytest.approx(cmap(0.0)[:3])
        assert result[1] == pytest.approx(cmap(0.5)[:3])
        assert result[2] == pytest.approx(cmap(1.0)[:3])

    def test_categorical_nan_falls_back(self):
        """np.nan in categorical data is treated as missing."""
        data = {"site": np.array(["4a", np.nan, "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)
        assert result[2] != (0.6, 0.0, 0.0)

    def test_categorical_none_falls_back(self):
        """None in categorical data is treated as missing."""
        data = {"site": np.array(["4a", None, "8b"], dtype=object)}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data, colour_by="site",
        )
        # Middle atom (H) should get species colour
        assert result[1] == (1.0, 1.0, 1.0)
        # Others should NOT be the species colour
        assert result[0] != (0.4, 0.4, 0.4)
        assert result[2] != (0.6, 0.0, 0.0)

    def test_list_colour_by_mismatched_cmap_length(self):
        """Mismatched cmap list length raises ValueError."""
        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        with pytest.raises(ValueError, match="cmap"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by=["a", "b"],
                cmap=["viridis", "plasma", "inferno"],
            )

    def test_list_colour_by_mismatched_colour_range_length(self):
        """Mismatched colour_range list length raises ValueError."""
        data = {
            "a": np.array([1.0, np.nan, np.nan]),
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        with pytest.raises(ValueError, match="colour_range"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by=["a", "b"],
                colour_range=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
            )

    def test_list_colour_by_fallback_colour_collision(self):
        """Merge uses data masks, not colour equality with fallback.

        If a cmap returns the same RGB as the species fallback, the
        atom should still get the cmap colour (not be skipped).
        """
        # C's species colour is (0.4, 0.4, 0.4).  Create a cmap that
        # always returns exactly (0.4, 0.4, 0.4).
        def grey_cmap(_v: float) -> tuple[float, float, float]:
            return (0.4, 0.4, 0.4)

        def blue(_v: float) -> tuple[float, float, float]:
            return (0.0, 0.0, 1.0)

        data = {
            # Layer "a" has data for atom 0 (C) — cmap will return same
            # as species colour.
            "a": np.array([1.0, np.nan, np.nan]),
            # Layer "b" has data for atom 1.
            "b": np.array([np.nan, 2.0, np.nan]),
        }
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by=["a", "b"], cmap=[grey_cmap, blue],
        )
        # Atom 0 has data in layer "a", so it should NOT fall through
        # to layer "b".  With colour-equality merge it would be treated
        # as missing because grey_cmap returns the same as fallback.
        assert result[0] == (0.4, 0.4, 0.4)  # from grey_cmap
        assert result[1] == (0.0, 0.0, 1.0)  # from blue
        assert result[2] == (0.6, 0.0, 0.0)  # fallback (O)

    def test_single_key_list_cmap_raises(self):
        """List cmap with single colour_by string raises ValueError."""
        data = {"val": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="cmap"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by="val", cmap=["viridis"],
            )

    def test_single_key_list_colour_range_raises(self):
        """List colour_range with single colour_by string raises ValueError."""
        data = {"val": np.array([1.0, 2.0, 3.0])}
        with pytest.raises(ValueError, match="colour_range"):
            resolve_atom_colours(
                self.SPECIES, self.STYLES, data,
                colour_by="val", colour_range=[(0.0, 1.0)],
            )

    def test_colormap_object_returns_rgb(self):
        """Passing a Colormap object produces 3-tuple (r, g, b) colours."""
        import matplotlib
        cmap_obj = matplotlib.colormaps["viridis"]

        data = {"val": np.array([0.0, 0.5, 1.0])}
        result = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap=cmap_obj,
        )
        for colour in result:
            assert len(colour) == 3
        # Check values match the string-based lookup.
        expected = resolve_atom_colours(
            self.SPECIES, self.STYLES, data,
            colour_by="val", cmap="viridis",
        )
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)
