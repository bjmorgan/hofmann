"""Shared test fixtures for hofmann."""

from pathlib import Path

import matplotlib
import pytest

# Force the non-interactive backend for the whole suite, before any
# test module imports pyplot: rendering tests must not open windows
# or depend on whichever GUI backend the local environment selects.
matplotlib.use("Agg")

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
