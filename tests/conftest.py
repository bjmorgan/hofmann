"""Shared test fixtures for hofmann."""

# Select a non-interactive backend before any test imports pyplot, so
# the suite never tries to open a GUI window.
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def ch4_bs_path():
    """Return the path to the CH4 .bs fixture file."""
    return FIXTURES_DIR / "ch4.bs"


@pytest.fixture
def ch4_mv_path():
    """Return the path to the CH4 .mv fixture file."""
    return FIXTURES_DIR / "ch4.mv"
